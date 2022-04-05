import math
import numpy as np
import os
import sys
import torch
import torch.nn as nn
from datetime import datetime

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))

from attacks import Attack

class CWINFAttack(Attack):

    def __init__(self, model=None, kappa=0, init_const=0.1, lr=0.02, binary_search_steps=5, max_iters=1000,
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
        super(CWINFAttack, self).__init__(model=model, num_classes=num_classes)
        self.model = model
        self.kappa = kappa * 1.0
        self.learning_rate = lr
        self.init_const = init_const
        self.const_factor = 2
        self.const_limit = 10
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_iterations = max_iters
        self.binary_search_steps = binary_search_steps
        self.tau = 1
        self.tau_decay = 0.9

    def attack_achieved(self, pre_softmax, target_class):
        pre_softmax[target_class] -= self.kappa
        # only when target_class is at least kappa larger we take this attack to be successful
        return np.argmax(pre_softmax) == target_class

    def l2attack_single(self, sample, target, valid_map, initial_const, device, tau):
        # Convert sample image from pixel [0, 1] to real (-inf, inf)
        mid_point = (self.upper_bound + self.lower_bound) * 0.5
        half_range = (self.upper_bound - self.lower_bound) * 0.5
        arctanh_sample = np.arctanh((sample - mid_point) / half_range * 0.9999)
        var_sample = torch.from_numpy(arctanh_sample).to(device)
        original_image = torch.tensor(sample, device=device)
        # convert targets to one-hot code
        target_onehot = torch.zeros(self.num_classes).to(device)
        target_onehot[target] = 1

        c = initial_const
        while c < self.const_limit:
            modifier = nn.Parameter(torch.zeros(sample.shape, device=device), requires_grad=True)
            optimizer = torch.optim.Adam([modifier], lr=self.learning_rate)
            for i in range(self.max_iterations):
                perturbed_image = torch.tanh(var_sample + modifier * valid_map.unsqueeze(0)) * half_range + mid_point
                delta_imposed = torch.abs(perturbed_image - original_image)
                prediction = self.model(perturbed_image.unsqueeze(0)).squeeze()
                constraint_loss = \
                    torch.max((prediction - 1e10 * target_onehot).max() - prediction[target] + self.kappa, 0).values
                linfdist = (delta_imposed.sum(dim=0) - tau).clamp(min=0).sum()
                loss_f = c * constraint_loss
                loss = linfdist + loss_f.sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Test whether l2 attack succeeded
                if self.attack_achieved(prediction.squeeze().detach().cpu().numpy(), target):
                    return perturbed_image, delta_imposed
            c *= self.const_factor

    def perturbation(self, samples: np.ndarray, targets: np.ndarray, batch_size, device):
        adv_samples = []
        for i in range(batch_size):
            adv_samples.append(self.linf_attack(samples[i], targets[i], device))
        return np.asarray(adv_samples)

    def linf_attack(self, sample: np.ndarray, target, device):
        best_ae = sample
        tau = 1
        while tau > 1. / 256.:
            valid_map = torch.ones(sample.shape[1:], device=device)
            perturbed_image, delta_imposed = self.l2attack_single(sample=sample, target=target, valid_map=valid_map,
                                                                  initial_const=self.init_const, device=device, tau=tau)
            if perturbed_image is None:
                break

            current_tau = torch.max(delta_imposed.sum(dim=0).flatten()).item()
            if tau > current_tau:
                tau = current_tau
            tau *= self.tau_decay

            best_ae = perturbed_image.detach().cpu().numpy()
        l_inf = np.max(np.abs((best_ae - sample)).sum(0))
        print('Tau = {}, L_INF distance = {}'.format(tau, l_inf))
        return best_ae

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
