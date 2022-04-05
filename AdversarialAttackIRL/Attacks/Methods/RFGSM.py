import math

import numpy as np
import torch
import sys
import os
sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))

from AttackUtils import tensor2variable
from attacks import Attack


class RFGSMAttack(Attack):

    def __init__(self, model=None, epsilon=None, alpha_ratio=None, num_classes=10):
        """

        :param model:
        :param eps:
        :param alpha:
        """
        super(RFGSMAttack, self).__init__(model, num_classes=num_classes)
        self.model = model

        self.epsilon = epsilon
        self.alpha_ratio = alpha_ratio

    def perturbation(self, samples, ys, device):
        """

        :param samples:
        :param ys:
        :param device:
        :return:
        """

        copy_samples = np.copy(samples)

        # add randomized single-step attack
        copy_samples = copy_samples + (self.alpha_ratio * self.epsilon * np.sign(np.random.randn(*copy_samples.shape)))
        copy_samples = np.clip(copy_samples, 0.0, 1.0).astype(np.float32)

        eps = (1.0 - self.alpha_ratio) * self.epsilon
        var_samples = tensor2variable(torch.from_numpy(copy_samples), device=device, requires_grad=True)
        var_ys = tensor2variable(torch.LongTensor(ys), device=device)

        self.model.eval()
        preds = self.model(var_samples)
        loss_fun = torch.nn.CrossEntropyLoss()
        loss = loss_fun(preds, var_ys)
        loss.backward()

        gradient_sign = var_samples.grad.data.cpu().sign().numpy()
        adv_samples = copy_samples + eps * gradient_sign

        adv_samples = np.clip(adv_samples, 0.0, 1.0)
        return adv_samples

    def batch_perturbation(self, xs, ys, batch_size, device):
        """

        :param xs:
        :param ys:
        :param batch_size:
        :param device:
        :return:
        """
        assert len(xs) == len(ys), "The lengths of samples and its ys should be equal"

        adv_sample = []
        number_batch = int(math.ceil(len(xs) / batch_size))
        for index in range(number_batch):
            start = index * batch_size
            end = min((index + 1) * batch_size, len(xs))
            print('\r===> in batch {:>2}, {:>4} ({:>4} in total) nature examples are perturbed ... '.format(index+1, end - start, end))

            batch_adv_images = self.perturbation(xs[start:end], ys[start:end], device)
            adv_sample.extend(batch_adv_images)
        return np.array(adv_sample)
