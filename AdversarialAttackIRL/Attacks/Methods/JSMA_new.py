import numpy as np
import torch
from torch.autograd.gradcheck import zero_gradients

import sys
import os
sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))

from AttackUtils import tensor2variable
from attacks import Attack


class JSMAAttack(Attack):
    def __init__(self, model=None, theta=1.0, gamma=0.1):
        """

        :param model:
        :param theta:
        :param gamma:
        """
        super(JSMAAttack, self).__init__(model=model)
        self.model = model.eval()

        self.theta = theta
        self.gamma = gamma

    def compute_jacobian(self, input, device):
        """
        computing the derivative of model with respect to the input features (jacobian matrix)
        :param input: input with 1 X C X H X W size
        :param device: specified device
        :return: jacobian matrix (10 X [H*W])
        """
        self.model.eval()
        output = self.model(input)[:, :10]

        num_features = int(np.prod(input.shape[1:]))
        jacobian = torch.zeros([output.size()[1], num_features])
        mask = torch.zeros(output.size()).to(device)  # chooses the derivative to be calculated
        for i in range(output.size()[1]):
            mask[:, i] = 1
            zero_gradients(input)
            # compute one logit's gradient
            output.backward(mask, retain_graph=True)
            # copy the derivative to the target place
            jacobian[i] = input._grad.squeeze().view(-1, num_features).clone()
    
            mask[:, i] = 0  # reset

        return jacobian.to(device)

    def saliency_map(self, jacobian, target_index, increasing, search_space, nb_features, device):
        """

        :param jacobian: the forward derivative (jacobian matrix)
        :param target_index: target class
        :param increasing: to increase or decrease pixel intensities
        :param search_space: the features indicate the perturbation search space
        :param nb_features: total number of feature
        :param device: specified device
        :return: a pair of pixel
        """
        print(jacobian.shape)

        try:
            domain = torch.eq(search_space, 1).float()  # The search domain
            # domain is one-dimensional expanded tensor.
            # the sum of all features' derivative with respect to each class
            all_sum = torch.sum(jacobian, dim=0, keepdim=True)
            target_grad = jacobian[target_index]  # The forward derivative of the target class
            others_grad = all_sum - target_grad  # The sum of forward derivative of other classes
            
            # this list blanks out those that are not in the search domain
            if increasing:
                increase_coef = 2 * (torch.eq(domain, 0)).float().to(device)
            else:
                increase_coef = -1 * 2 * (torch.eq(domain, 0)).float().to(device)
            increase_coef = increase_coef.view(-1, nb_features)

            # calculate sum of target forward derivative of any 2 features.
            target_tmp = target_grad.clone()
            target_tmp -= increase_coef * torch.max(torch.abs(target_grad))
            alpha = target_tmp.view(-1, 1, nb_features) + target_tmp.view(-1, nb_features, 1)  # PyTorch will automatically extend the dimensions
            # calculate sum of other forward derivative of any 2 features.
            others_tmp = others_grad.clone()
            others_tmp += increase_coef * torch.max(torch.abs(others_grad))
            beta = others_tmp.view(-1, 1, nb_features) + others_tmp.view(-1, nb_features, 1)

            # zero out the situation where a feature sums with itself
            tmp = np.ones((nb_features, nb_features), int)
            np.fill_diagonal(tmp, 0)
            zero_diagonal = torch.from_numpy(tmp).byte().to(device)

            # According to the definition of saliency map in the paper (formulas 8 and 9),
            # those elements in the saliency map that doesn't satisfy the requirement will be blanked out.
            # Here (p,q) loop is replaced by a matrix in which entry (p,q) stands for the same
            if increasing:
                mask1 = torch.gt(alpha, 0.0)
                mask2 = torch.lt(beta, 0.0)
            else:
                mask1 = torch.lt(alpha, 0.0)
                mask2 = torch.gt(beta, 0.0)
            # apply the mask to the saliency map
            mask = torch.mul(torch.mul(mask1, mask2), zero_diagonal.view_as(mask1))
            # do the multiplication according to formula 10 in the paper
            saliency_map = torch.mul(torch.mul(alpha, torch.abs(beta)), mask.float())
            # get the most significant two pixels
            max_value, max_idx = torch.max(saliency_map.view(-1, nb_features * nb_features), dim=1)
            p = max_idx // nb_features
            q = max_idx % nb_features
            return p, q
        except:
            # GPU out of memory
            raise NotImplementedError()

    def perturbation_single(self, sample, ys_target, device):
        """

        :param sample:
        :param ys_target:
        :param device:
        :return:
        """
        copy_sample = np.copy(sample)
        var_sample = tensor2variable(torch.from_numpy(copy_sample), device=device, requires_grad=True)
        var_target = tensor2variable(torch.LongTensor(ys_target), device=device)

        if self.theta > 0:
            increasing = True
        else:
            increasing = False

        num_features = int(np.prod(copy_sample.shape[1:]))
        shape = var_sample.size()

        # perturb two pixels in one iteration, thus max_iters is divided by 2.0
        max_iters = int(np.ceil(num_features * self.gamma / 2.0))

        # masked search domain, if the pixel has already reached the top or bottom, we don't bother to modify it.
        if increasing:
            search_domain = torch.lt(var_sample, 1).to(device)
        else:
            search_domain = torch.gt(var_sample, 0).to(device)
        search_domain = search_domain.view(num_features)

        self.model.eval().to(device)
        output = self.model(var_sample)
        current = torch.max(output.data, 1)[1].cpu().numpy()

        iter = 0
        while (iter < max_iters) and (current[0] != ys_target[0]) and (search_domain.sum() != 0):
            # calculate Jacobian matrix of forward derivative
            jacobian = self.compute_jacobian(input=var_sample, device=device)
            # get the saliency map and calculate the two pixels that have the greatest influence
            p1, p2 = self.saliency_map(jacobian, var_target, increasing, search_domain, num_features, device)
            # apply modifications
            var_sample_flatten = var_sample.view(-1, num_features)
            var_sample_flatten[0, p1] += self.theta
            var_sample_flatten[0, p2] += self.theta

            new_sample = torch.clamp(var_sample_flatten, min=0.0, max=1.0)
            new_sample = new_sample.view(shape)
            search_domain[p1] = 0
            search_domain[p2] = 0
            var_sample = tensor2variable(new_sample.clone().detach(), device=device, requires_grad=True)

            output = self.model(var_sample)
            current = torch.max(output.data, 1)[1].cpu().numpy()
            iter += 1

        adv_samples = var_sample.data.cpu().numpy()
        return adv_samples

    def perturbation(self, xs, ys_target, device):
        """

        :param xs:
        :param ys_target:
        :param device:
        :return:
        """
        assert len(xs) == len(ys_target), "The lengths of samples and its ys should be equal"
        print('The JSMA attack perturbs the samples one by one ...... ')

        adv_samples = []
        for i in range(len(xs)):
            print("Harassing sample {}/{}...".format(i+1, len(xs)), end='\r')
            adv_image = self.perturbation_single(sample=xs[i: i + 1], ys_target=ys_target[i: i + 1], device=device)
            adv_samples.extend(adv_image)
        return np.array(adv_samples)
