import math
import os
import sys

import numpy as np
import torch

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))

from AttackUtils import tensor2variable
from attacks import Attack


class LLCAttack(Attack):

    def __init__(self, model=None, epsilon=None, num_steps=None, iter_epsilon=None, num_classes=10):
        """

        :param model:
        :param epsilon:
        """
        super(LLCAttack, self).__init__(model, num_classes=num_classes)
        self.model = model
        self.num_steps = num_steps
        self.epsilon = epsilon
        self.iter_epsilon = iter_epsilon

    def perturbation(self, samples, device):
        """

        :param samples:
        :param ys_target:
        :param device:
        :return:
        """
        adv_samples = np.copy(samples)
        self.model.to(device)
        for index in range(self.num_steps):
            var_samples = tensor2variable(torch.from_numpy(adv_samples), device=device, requires_grad=True)
            self.model.eval()
            preds = self.model(var_samples)
            target_label = torch.argmin(preds.data, 1)
            loss_fun = torch.nn.CrossEntropyLoss()
            loss = loss_fun(preds, target_label)
            loss.backward()
            gradient_sign = var_samples.grad.data.sign().cpu().numpy()

            adv_samples = adv_samples - self.iter_epsilon * gradient_sign
            adv_samples = np.clip(adv_samples, 0.0, 1.0)
            adv_samples = np.clip(adv_samples, samples - self.epsilon, samples + self.epsilon)

        return adv_samples

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
            print('\r===> in batch {:>2}, {:>4} ({:>4} in total) nature examples are perturbed ... '.format(index,
                                                                                                            end - start,
                                                                                                            end))

            batch_adv_images = self.perturbation(xs[start:end], device)
            adv_sample.extend(batch_adv_images)
        return np.array(adv_sample)
