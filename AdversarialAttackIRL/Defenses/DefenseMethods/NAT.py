import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from Defenses.DefenseMethods.defenses import Defense
from Defenses.DefenseMethods.defenses import validation_evaluation
from SurrogateModel.train import evaluate
from utils import write_log_file


class NATDefense:

    def __init__(self, model: torch.nn.Module, dataset, num_epochs, batch_size, lr, weight_decay, adv_ratio, min, max,
                 mu, sigma, device, trainable_params):
        """

        :param model:
        :param model_name:
        :param dataset:
        :param training_parameters:
        :param device:
        :param kwargs:
        """
        self.model = model
        self.dataset = dataset
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        # prepare the optimizers
        self.optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)

        self.adv_ratio = adv_ratio
        self.eps_mu = mu
        self.eps_sigma = sigma
        self.clip_eps_min = min
        self.clip_eps_max = max
        self.device = device

    def random_llc_generation(self, var_natural_images=None):
        """

        :param var_natural_images:
        :return:
        """
        self.model.eval()
        clone_var_natural_images = var_natural_images.clone()

        # get the random epsilon for the Random LLC generation
        random_eps = np.random.normal(loc=self.eps_mu, scale=self.eps_sigma, size=[var_natural_images.size(0)]) / 255.0
        random_eps = np.clip(np.abs(random_eps), self.clip_eps_min, self.clip_eps_max)

        clone_var_natural_images.requires_grad = True

        # prepare the least likely class labels (avoid label leaking effect)
        logits = self.model(clone_var_natural_images)
        llc_labels = torch.min(logits, dim=1)[1]
        # get the loss and gradients
        loss_llc = F.cross_entropy(logits, llc_labels)
        gradients_llc = torch.autograd.grad(loss_llc, clone_var_natural_images)[0]

        clone_var_natural_images.requires_grad = False

        gradients_sign = torch.sign(gradients_llc)
        var_random_eps = torch.from_numpy(random_eps).float().to(self.device)

        # generation of adversarial examples
        with torch.no_grad():
            list_var_adv_images = []
            for i in range(var_natural_images.size(0)):
                var_adv_image = var_natural_images[i] - var_random_eps[i] * gradients_sign[i]
                var_adv_image = torch.clamp(var_adv_image, min=0.0, max=1.0)
                list_var_adv_images.append(var_adv_image)
            ret_adv_images = torch.stack(list_var_adv_images)
        ret_adv_images = torch.clamp(ret_adv_images, min=0.0, max=1.0)

        return ret_adv_images

    def train_one_epoch_with_adv_and_nat(self, train_loader, epoch):
        """

        :param train_loader:
        :param epoch:
        :return:
        """

        for index, (images, labels) in enumerate(train_loader):
            nat_images = images.to(self.device)
            nat_labels = labels.to(self.device)

            # set the model in the eval mode and generate the adversarial examples using the LLC (Least Likely Class) attack
            self.model.eval()
            adv_images = self.random_llc_generation(var_natural_images=nat_images)

            # set the model in the train mode
            self.model.train()

            logits_nat = self.model(nat_images)
            loss_nat = F.cross_entropy(logits_nat, nat_labels)  # loss on natural images

            logits_adv = self.model(adv_images)
            loss_adv = F.cross_entropy(logits_adv, nat_labels)  # loss on the generated adversarial images

            # add two parts of loss
            loss = (loss_nat + self.adv_ratio * loss_adv) / (1.0 + self.adv_ratio)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def defense(self, train_loader, validation_loader, test_loader, model_save_path, log_path):
        print(f'log_path:{log_path}')
        best_valid_accu = 0
        for i in range(self.num_epochs):
            print(f'Epoch:{i}/{self.num_epochs}')
            self.train_one_epoch_with_adv_and_nat(train_loader, i)
            loss, accu = evaluate(model=self.model, device=self.device, validation_loader=validation_loader)
            if accu > best_valid_accu:
                write_log_file(log_path,
                               'Epoch {}: validation accuracy {} > {}, saving'.format(i, accu, best_valid_accu))
                best_valid_accu = accu
                torch.save(self.model.state_dict(), model_save_path)
            else:
                write_log_file(log_path,
                               'Epoch {}: validation accuracy {}'.format(i, accu, best_valid_accu))

        self.model.load_state_dict(torch.load(model_save_path))
        test_loss, test_accu = evaluate(self.model, self.device, test_loader)
        write_log_file(log_path, 'Testing Phase:\n\t loss = {}, accuracy = {}.'.format(test_loss, test_accu))
