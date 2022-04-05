import os
import sys
import time
from abc import ABCMeta, abstractmethod

import numpy as np
import torch

from AttackUtils import predict

sys.path.append('..')
from utils import concat_dir, model_map, write_log_file

ae_candidate_dir = 'AECandidates'
model_file_name = 'model.pth'
log_file_name = 'log.txt'


class Generation(object):
    __metaclass__ = ABCMeta

    def __init__(self,
                 dataset, model, model_depth, dataset_type, pretrained,
                 attack_name, clean_data_root, ae_save_root, model_root,
                 device):

        config_dir = concat_dir(dataset=dataset, model=model, data_type=dataset_type, depth=model_depth,
                                pretrained=pretrained)

        self.clean_data_dir = os.path.join(clean_data_root, dataset, ae_candidate_dir)
        self.raw_model_path = os.path.join(model_root, config_dir, 'model.pth')
        self.ae_save_dir = os.path.join(ae_save_root, config_dir, attack_name)
        os.makedirs(self.ae_save_dir, exist_ok=True)
        # if os.path.isfile(self.ae_save_dir + '.0/AE.npy') or os.path.isfile(self.ae_save_dir + '/AE.npy'):
        #     print(f'AE already generated at: {self.ae_save_dir}')
        #     exit(0)
        self.log_path = os.path.join(self.ae_save_dir, log_file_name)

        self.dataset = dataset
        self.attack_name = attack_name

        if dataset == 'ImageNet':
            self.num_classes = 10
        elif dataset == 'NudeNet' or dataset == 'AdienceGenderG':
            self.num_classes = 2
        else:
            raise NotImplementedError('unsupported dataset')

        try:
            model_generator = model_map['{}{}'.format(model, model_depth)]
        except KeyError:
            raise NotImplementedError('unsupported model type')
        if model != 'inception':
            self.model = model_generator(num_classes=self.num_classes)
            self.model.load_state_dict(torch.load(self.raw_model_path, map_location=device))
            self.model = self.model.to(device)
        if model == 'inception':
            self.model = model_generator(pretrained=False)
            self.model.fc = torch.nn.Linear(in_features=self.model.fc.in_features, out_features=self.num_classes)
            self.model.load_state_dict(torch.load(self.raw_model_path, map_location=device))
            self.model = self.model.to(device)
            self.model.aux_logits = False

        write_log_file(self.log_path,
                       'Loading the prepared clean samples (nature inputs and corresponding labels) that will be attacked ...... ')
        self.raw_samples = np.load('{}/raw_samples.npy'.format(self.clean_data_dir))
        self.true_labels = np.load('{}/true_labels.npy'.format(self.clean_data_dir))
        self.target_labels = np.load('{}/target_labels.npy'.format(self.clean_data_dir))

        self.device = device

    def generate(self):
        t1 = time.time()
        self._generate()
        t2 = time.time()
        write_log_file(self.log_path, "Generation time usage: {:.2f}s".format(t2 - t1))

    @abstractmethod
    def _generate(self):
        write_log_file(self.log_path, "abstract method of Generation is not implemented")
        raise NotImplementedError

    def evaluate(self, adv_samples):
        adv_labels = predict(model=self.model, samples=adv_samples, device='cpu')#self.device
        adv_labels = torch.max(adv_labels, 1)[1]
        adv_labels = adv_labels.cpu().numpy()
        np.save(os.path.join(self.ae_save_dir, 'AE.npy'), adv_samples)
        np.save(os.path.join(self.ae_save_dir, 'AdvLabels.npy'), adv_labels)
        np.save(os.path.join(self.ae_save_dir, 'TrueLabels.npy'), self.true_labels)
        true_labels = self.true_labels
        mis = np.sum(adv_labels != true_labels)
        write_log_file(self.log_path,
                       'For {} on {}, {}/{}={:.1f}% samples are misclassified\n'.format(self.attack_name, self.dataset,
                                                                                        mis,
                                                                                        len(adv_samples),
                                                                                        mis / len(adv_samples) * 100.0))
