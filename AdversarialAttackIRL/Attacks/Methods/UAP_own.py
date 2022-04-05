import numpy as np
import os
import sys
import torch

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))

from Methods.DEEPFOOL import DeepFoolAttack
from attacks import Attack
from AttackUtils import predict


class UniversalAttack(Attack):
    def __init__(self, model=None, max_iter_universal=np.inf, fooling_rate=0.5, epsilon=0.1, overshoot=0.02,
                 max_iter_deepfool=10, num_classes=10):
        """

        :param model: the targeted attack model
        :param max_iter_universal: the maximum iterations for the Universal Adversarial Perturbation (UAP)
        :param fooling_rate: the desired fooling rate for all samples sampled from the training dataset
        :param epsilon: the magnitude of perturbation in L\inf norm
        :param overshoot: the overshoot parameter for DeepFool
        :param max_iter_deepfool: the maximum iterations for DeepFool
        """
        super(UniversalAttack, self).__init__(model=model, num_classes=num_classes)
        self.model = model

        # parameters for UAP
        self.max_iter_universal = max_iter_universal
        self.fooling_rate = fooling_rate
        self.epsilon = epsilon
        # parameters for DeepFool
        self.overshoot_deepfool = overshoot
        self.max_iter_deepfool = max_iter_deepfool

    def projection_linf(self, v, eps):
        """
        The perturbation v is projected on the l\inf norm of radius eps
        :param v: the perturbation
        :param eps: the magnitude of perturbation in L\inf norm
        :return: the projected perturbation
        """
        v = np.sign(v) * np.minimum(abs(v), eps)
        return v

    def universal_perturbation(self, dataset, device, true_labels):
        """

        :param dataset: sampled dataset to compute the universal perturbation
        :param validation: validation dataset to assess the universal perturbation
        :param device:
        :return: the estimated universal adversarial perturbation
        """
        print('\n\nstarting to compute the universal adversarial perturbation with the training dataset ......\n')
        iteration, ratio = 0, 0.0
        uni_pert = torch.zeros(size=dataset[0:1].shape)
        true_labels = torch.from_numpy(true_labels)
        while ratio < self.fooling_rate and iteration < self.max_iter_universal:
            print('iteration: {}'.format(iteration + 1))

            self.model.eval()
            for index in range(dataset.shape[0]):
                print("Processing {}/{}...".format(index, dataset.shape[0]), end='\r')
                image = torch.from_numpy(dataset[index:index + 1])
                original = torch.max(self.model(image.to(device)), 1)[1]  # prediction of the nature image
                perturbed_image = torch.clamp(image + uni_pert, 0.0, 1.0)  # predication of the perturbed image
                current = torch.max(self.model(perturbed_image.to(device)), 1)[1]
                if original == current:
                    # compute the minimal perturbation using the DeepFool
                    deepfool = DeepFoolAttack(model=self.model, overshoot=self.overshoot_deepfool,
                                              max_iters=self.max_iter_deepfool, num_classes=self.num_classes)
                    _, delta, iter_num = deepfool.perturbation_single(sample=perturbed_image.numpy(), device=device)
                    # update the universal perturbation
                    if iter_num < self.max_iter_deepfool - 1:
                        uni_pert += torch.from_numpy(delta)
                        uni_pert = self.projection_linf(v=uni_pert, eps=self.epsilon)

            iteration += 1

            print('\tcomputing the fooling rate w.r.t current the universal adversarial perturbation ......')

            perturbed_sample = torch.from_numpy(dataset.copy())
            perturbed_sample += uni_pert
            perturbed_sample = torch.clamp(perturbed_sample, 0.0, 1.0)
            pred = torch.max(predict(self.model, perturbed_sample, device), 1)[1].cpu()
            success = torch.sum(pred != true_labels).item()
            total = dataset.shape[0]
            ratio = success / total
            print('\tcurrent fooling rate is {}/{}={}\n'.format(success, total, ratio))
        return uni_pert

    def perturbation(self, xs, uni_pert, device):
        """

        :param xs:
        :param uni_pert: the computed universal adversarial perturbation
        :param device:
        :return:
        """
        xs += uni_pert
        xs = np.clip(xs, 0.0, 1.0)
        return xs
