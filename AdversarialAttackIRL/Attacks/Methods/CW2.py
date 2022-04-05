import math
import os
import sys

import numpy as np
import torch

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))

from AttackUtils import tensor2variable
from attacks import Attack


class CW2Attack(Attack):

    def __init__(self, model=None, norm=2, kappa=0, init_const=0.001, lr=0.02, binary_search_steps=5, max_iters=10000,
                 lower_bound=0.0, upper_bound=1.0, num_classes=10):
        """
        :param model:
        :param kappa:
        :param init_const:
        :param lr:
        :param binary_search_steps:
        :param max_iters:
        :param lower_bound:
        :param upper_bound:
        """
        super(CW2Attack, self).__init__(model=model, num_classes=num_classes)
        self.model = model
        self.norm = norm
        self.kappa = kappa * 1.0
        self.learning_rate = lr
        self.init_const = init_const
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_iterations = max_iters
        self.binary_search_steps = binary_search_steps
        self.tau_for_L_inf = 1
        self.tau_decay = 0.9

    def perturbation(self, samples, ys_targets, batch_size, device):
        """

        :param samples: (batch_size, num_channels, width, height) of float, images
        :param ys_targets: (batch_size) of int, indicating target class
        :param batch_size: batch size
        :param device:
        :return: (batch_size, num_channels, width, height) of float, perturbed images
        """
        # assert len(samples) == batch_size, "the length of sample is not equal to the batch_size"

        # transform the samples [lower, upper] to [-1, 1] and then to the arctanh space
        mid_point = (self.upper_bound + self.lower_bound) * 0.5
        half_range = (self.upper_bound - self.lower_bound) * 0.5
        arctanh_samples = np.arctanh((samples - mid_point) / half_range * 0.9999)
        var_samples = tensor2variable(torch.from_numpy(arctanh_samples), device=device, requires_grad=True)

        # set the lower and upper bound for searching 'c' const in the CW2 attack
        const_origin = np.ones(shape=batch_size, dtype=float) * self.init_const
        c_upper_bound = np.full(batch_size, 1e10)
        c_lower_bound = np.zeros(batch_size)

        # convert targets to one hot encoder
        temp_one_hot_matrix = np.eye(self.num_classes)
        targets_in_one_hot = []
        for i in range(batch_size):
            current_target = temp_one_hot_matrix[int(ys_targets[i])]
            targets_in_one_hot.append(current_target)
        targets_in_one_hot = tensor2variable(torch.FloatTensor(np.array(targets_in_one_hot)), device=device)

        best_lp = np.full(batch_size, 1e10)
        best_perturbation = np.zeros(var_samples.size())
        current_prediction_class = np.full(batch_size, -1)

        def attack_achieved(pre_softmax, target_class):
            pre_softmax[target_class] -= self.kappa
            # only when target_class is at least kappa larger we take this attack to be successful
            return np.argmax(pre_softmax) == target_class

        self.model.eval()
        # Outer loop for linearly searching for c
        for search_for_c in range(self.binary_search_steps):

            modifier = torch.zeros(var_samples.size()).float()
            modifier = tensor2variable(modifier, device=device, requires_grad=True)
            optimizer = torch.optim.Adam([modifier], lr=self.learning_rate)
            var_const = tensor2variable(torch.FloatTensor(const_origin), device=device)
            print("\tbinary search step {}:".format(search_for_c), end='\r')

            for iteration_times in range(self.max_iterations):
                # inverse the transform tanh -> [0, 1]
                perturbed_images = torch.tanh(var_samples + modifier) * half_range + mid_point
                prediction = self.model(perturbed_images)  # strip output in order to fit correct dimension
                delta_imposed = torch.abs(perturbed_images - (torch.tanh(var_samples) * half_range + mid_point))
                lpdist = torch.sum(delta_imposed ** self.norm, [1, 2, 3])
                constraint_loss = torch.max(
                    (prediction - 1e10 * targets_in_one_hot).max(1)[0] - (prediction * targets_in_one_hot).sum(1),
                    torch.ones(batch_size, device=device) * self.kappa * -1)

                loss_f = var_const * constraint_loss
                loss = lpdist.sum() + loss_f.sum()  # minimize |r| + c * loss_f(x+r,l)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # update the best lp distance, current prediction class as well as the corresponding adversarial example
                for i, (dist, score, img) in enumerate(
                        zip(lpdist.data.cpu().numpy(), prediction.data.cpu().numpy(),
                            perturbed_images.data.cpu().numpy())):
                    if dist < best_lp[i] and attack_achieved(score, ys_targets[i]):
                        best_lp[i] = dist
                        current_prediction_class[i] = np.argmax(score)
                        best_perturbation[i] = img

            # update the best constant c for each sample in the batch
            for i in range(batch_size):
                if current_prediction_class[i] == ys_targets[i] and current_prediction_class[i] != -1:
                    c_upper_bound[i] = min(c_upper_bound[i], const_origin[i])
                    if c_upper_bound[i] < 1e10:
                        const_origin[i] = (c_lower_bound[i] + c_upper_bound[i]) / 2.0
                else:
                    c_lower_bound[i] = max(c_lower_bound[i], const_origin[i])
                    if c_upper_bound[i] < 1e10:
                        const_origin = (c_lower_bound[i] + c_upper_bound[i]) / 2.0
                    else:
                        const_origin[i] *= 10

        return np.array(best_perturbation)

    def batch_perturbation(self, xs, ys_target, batch_size, device):
        """

        :param xs:
        :param ys_target:
        :param batch_size:
        :param device:
        :return:
        """
        assert len(xs) == len(ys_target), "The lengths of samples and its ys should be equal"

        adv_sample = []
        number_batch = int(math.ceil(len(xs) / batch_size))
        for index in range(number_batch):
            start = index * batch_size
            end = min((index + 1) * batch_size, len(xs))
            print('\r===> in batch {:>2}, {:>4} ({:>4} in total) nature examples are perturbed ... '.format(index + 1,
                                                                                                            end - start,
                                                                                                            end))

            batch_adv_images = self.perturbation(xs[start:end], ys_target[start:end], batch_size, device)
            adv_sample.extend(batch_adv_images)
        return np.array(adv_sample)
