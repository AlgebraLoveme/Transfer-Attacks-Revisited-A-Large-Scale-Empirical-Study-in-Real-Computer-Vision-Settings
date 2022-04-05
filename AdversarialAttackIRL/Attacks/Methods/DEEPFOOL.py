import collections
import os
import sys

import numpy as np
import torch
# from torch.autograd.gradcheck import zero_gradients

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))

from AttackUtils import tensor2variable
from attacks import Attack


def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)


class DeepFoolAttack(Attack):

    def __init__(self, model=None, overshoot=0.02, max_iters=50, num_classes=10):
        """

        :param model:
        :param overshoot:
        :param max_iters:
        """
        super(DeepFoolAttack, self).__init__(model=model, num_classes=num_classes)
        self.model = model
        self.overshoot = overshoot
        self.max_iterations = max_iters

    def perturbation_single(self, sample, device):
        """

        :param sample:
        :param device:
        :return:
        """

        assert sample.shape[0] == 1, 'only perturbing one sample'
        copy_sample = np.copy(sample)
        var_sample = tensor2variable(torch.from_numpy(copy_sample), device=device, requires_grad=True).float()

        self.model.eval()
        prediction = self.model(var_sample)

        # indices of predication in descending order
        I = np.argsort(prediction.data.cpu().numpy() * -1)
        current = original = I[0, 0]

        perturbation_r_tot = np.zeros(copy_sample.shape, dtype=np.float32)
        iteration = 0
        while (original == current) and (iteration < self.max_iterations):

            # predication for the adversarial example in i-th iteration
            # var_sample.zero_grad()
            zero_gradients(var_sample)
            self.model.eval()
            f_kx = self.model(var_sample)
            current = torch.max(f_kx, 1)[1]
            # gradient of the original example
            f_kx[0, original].backward(retain_graph=True)
            grad_original = var_sample.grad.data.cpu().clone().detach()

            # calculate the w_k and f_k for every class label
            closest_dist = 1e15
            # for k in range(1, self.truncation):
            for k in range(1, self.num_classes):
                # gradient of adversarial example for k-th label
                # var_sample.zero_grad()
                zero_gradients(var_sample)
                f_kx[0, I[0, k]].backward(retain_graph=True)
                grad_current = var_sample.grad.data.cpu().clone().detach()
                # update w_k and f_k
                w_k = grad_current - grad_original
                f_k = (f_kx[0, I[0, k]] - f_kx[0, original]).detach().data.cpu()
                # find the closest distance and the corresponding w_k
                dist_k = np.abs(f_k) / (np.linalg.norm(w_k) + 1e-5)  # in case there is a divide-by-zero error
                if dist_k < closest_dist:
                    closest_dist = dist_k
                    closest_w = w_k

            # accumulation of perturbation
            try:
                r_i = (closest_dist + 1e-4) * closest_w / np.linalg.norm(closest_w)
            except:
                r_i = torch.zeros_like(w_k)
            perturbation_r_tot = perturbation_r_tot + r_i.numpy()

            tmp_sample = np.clip((1 + self.overshoot) * perturbation_r_tot + sample, 0.0, 1.0)
            var_sample = tensor2variable(torch.from_numpy(tmp_sample), device=device, requires_grad=True)

            iteration += 1

        adv = np.clip(sample + (1 + self.overshoot) * perturbation_r_tot, 0.0, 1.0)
        return adv, perturbation_r_tot, iteration

    def perturbation(self, xs, device):
        """

        :param xs: batch of samples
        :param device:
        :return: batch of adversarial samples
        """
        print('The DeepFool attack perturbs the samples one by one ......')
        adv_samples = []
        for i in range(len(xs)):
            print("Harassing sample {}/{}...".format(i + 1, len(xs)))
            adv_image, _, _ = self.perturbation_single(sample=xs[i: i + 1], device=device)
            adv_samples.extend(adv_image)
        return np.array(adv_samples)
