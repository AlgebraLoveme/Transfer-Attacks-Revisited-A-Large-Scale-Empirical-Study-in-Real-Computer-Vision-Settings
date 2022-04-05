import math
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from attacks import Attack


class CW0Attack(Attack):

    def __init__(self, model, kappa=0, init_const=1e-4, lr=1e-3, max_iters=10000, lower_bound=0.0,
                 upper_bound=1.0, truncation=None, const_factor=2, largest_const=2e6):
        super(CW0Attack, self).__init__(model=model, truncation=truncation)
        self.model = model
        self.kappa = kappa * 1.0
        self.learning_rate = lr
        self.init_const = init_const
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_iterations = max_iters
        self.tau_decay = 0.9
        self.const_limit = largest_const
        self.const_factor = const_factor

    def attack_achieved(self, pre_softmax, target_class):
        pre_softmax[target_class] -= self.kappa
        # only when target_class is at least kappa larger we take this attack to be successful
        return np.argmax(pre_softmax) == target_class

    def l2attack_single(self, sample, target, valid_map, initial_const, device):
        mid_point = (self.upper_bound + self.lower_bound) * 0.5
        half_range = (self.upper_bound - self.lower_bound) * 0.5
        arctanh_sample = np.arctanh((sample - mid_point) / half_range * 0.9999)
        var_sample = torch.from_numpy(arctanh_sample).to(device)
        original_image = torch.tensor(sample, device=device)
        # convert targets to one-hot code
        target_onehot = torch.zeros(self.truncation).to(device)
        target_onehot[target] = 1
        c = initial_const
        while c < self.const_limit:
            modifier = nn.Parameter(torch.zeros(sample.shape, device=device), requires_grad=True)
            optimizer = torch.optim.Adam([modifier], lr=self.learning_rate)
            for i in range(self.max_iterations):
                perturbed_image = torch.tanh(var_sample + modifier * valid_map.unsqueeze(0)) * half_range + mid_point
                delta_imposed = torch.abs(perturbed_image - original_image)
                prediction = self.model(perturbed_image.unsqueeze(0))[0, :self.truncation]
                constraint_loss = \
                    torch.max((prediction - 1e10 * target_onehot).max() - prediction[target] + self.kappa, 0)[0]
                l2dist = torch.sum(delta_imposed)
                loss_f = torch.tensor(data=c, device=device) * constraint_loss
                loss = l2dist.sum() + loss_f.sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Test whether l2 attack succeeded
                if self.attack_achieved(prediction.squeeze().detach().cpu().numpy(), target):
                    grad = modifier.grad
                    return grad, perturbed_image
            c *= self.const_factor

    def perturbation(self, samples: np.ndarray, targets: np.ndarray, batch_size, device):
        adv_samples = []
        for i in range(batch_size):
            adv_samples.append(self.l0_attack(samples[i], targets[i], device))
        return np.asarray(adv_samples)

    def l0_attack(self, sample: np.ndarray, target, device):
        best_ae = sample
        while True:
            st = datetime.now()
            valid_map = torch.ones(sample.shape[1:], device=device)
            grad, perturbed_image = self.l2attack_single(sample=best_ae, target=target, valid_map=valid_map,
                                                         initial_const=self.init_const, device=device)
            print('Time elapsed for 1 l2 attack = {}'.format(datetime.now() - st))
            if perturbed_image is None:
                break
            delta_imposed = torch.abs(torch.from_numpy(sample).to(device) - perturbed_image)
            saliency_map = delta_imposed.sum(dim=0) * grad.sum(dim=0)
            ranking = np.argsort(saliency_map.flatten().cpu().detach().numpy())
            for idx in ranking:
                i = idx // sample.shape[1]
                j = idx % sample.shape[1]
                if valid_map[i, j]:
                    valid_map[i, j] = 0
                    break
            if valid_map.sum() == 0:
                break
            best_ae = perturbed_image
        return best_ae

    def batch_perturbation(self, xs, ys_target, batch_size, device):
        assert len(xs) == len(ys_target), "The lengths of samples and its ys should be equal"

        adv_sample = []
        number_batch = int(math.ceil(len(xs) / batch_size))
        for index in range(number_batch):
            start = index * batch_size
            end = min((index + 1) * batch_size, len(xs))
            batch_adv_images = self.perturbation(xs[start:end], ys_target[start:end], batch_size, device)
            adv_sample.extend(batch_adv_images)
        return np.array(adv_sample)
