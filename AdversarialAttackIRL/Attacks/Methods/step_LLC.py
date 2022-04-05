import math

import numpy as np
import torch

import sys
import os
sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))

from AttackUtils import tensor2variable
from attacks import Attack


class step_LLCAttack(Attack):

    def __init__(self, model=None, epsilon=None, num_classes=10):
        """

        :param model:
        :param epsilon:
        """
        super(step_LLCAttack, self).__init__(model, num_classes=num_classes)
        self.model = model
        self.epsilon = epsilon

    def perturbation(self, samples, device):
        """

        :param samples:
        :param ys_target:
        :param device:
        :return:
        """
        adv_samples = torch.from_numpy(np.copy(samples)).to(device)
        self.model.to(device)
        var_samples = tensor2variable(adv_samples, device=device, requires_grad=True)
        self.model.eval()
        preds = self.model(var_samples)
        _, target_label = torch.min(preds.data, 1)
        loss_fun = torch.nn.CrossEntropyLoss()
        loss = loss_fun(preds, target_label)
        loss.backward()
        gradient_sign = var_samples.grad.data.sign()

        adv_samples = adv_samples - self.epsilon * gradient_sign
        adv_samples = torch.clamp(adv_samples, 0.0, 1.0)

        return adv_samples.cpu().numpy()

    def batch_perturbation(self, xs, batch_size, device):
        """

        :param xs:
        :param ys_target:
        :param batch_size:
        :param device:
        :return:
        """

        adv_sample = []
        number_batch = int(math.ceil(len(xs) / batch_size))
        for index in range(number_batch):
            start = index * batch_size
            end = min((index + 1) * batch_size, len(xs))
            print('\r===> in batch {:>2}, {:>4} ({:>4} in total) nature examples are perturbed ... '.format(index+1, end - start, end), end=' ')

            batch_adv_images = self.perturbation(xs[start:end], device)
            adv_sample.extend(batch_adv_images)
        return np.array(adv_sample)
