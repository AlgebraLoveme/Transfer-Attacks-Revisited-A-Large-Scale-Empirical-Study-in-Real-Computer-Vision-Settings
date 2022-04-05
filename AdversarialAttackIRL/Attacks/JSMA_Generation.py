import argparse
import os
import random

import numpy as np
import torch

from AttackUtils import predict, save_args
from Methods.JSMA import JSMAAttack
from Generation import Generation


class JSMAGeneration(Generation):

    def __init__(self, dataset, attack_name, targeted, raw_model_location, clean_data_location, adv_save_dir, device, theta, gamma, restricted, **kwards):
        super(JSMAGeneration, self).__init__(dataset, attack_name, raw_model_location, clean_data_location, adv_save_dir, device, restricted, targeted)

        self.theta = theta
        self.gamma = gamma

    def _generate(self):
        attacker = JSMAAttack(model=self.model, theta=self.theta, gamma=self.gamma)

        # get the targeted labels
        targets = self.target_labels
        # generating
        adv_samples = attacker.perturbation(xs=self.raw_samples, ys_target=targets, device=self.device)

        adv_labels = predict(model=self.model, samples=adv_samples, device=self.device)
        adv_labels = torch.max(adv_labels, 1)[1]
        adv_labels = adv_labels.cpu().numpy()

        np.save('{}{}_AdvExamples.npy'.format(self.adv_saver, self.attack_name), adv_samples)
        np.save('{}{}_AdvLabels.npy'.format(self.adv_saver, self.attack_name), adv_labels)
        np.save('{}{}_TrueLabels.npy'.format(self.adv_saver, self.attack_name), self.true_labels)

        true_labels = self.true_labels
        mis = np.sum(adv_labels!=true_labels)
        print('\nFor **{}** on **{}**: misclassification ratio is {}/{}={:.1f}%\n'.format(self.attack_name, self.dataset, mis, len(adv_samples),
                                                                                          mis / len(adv_labels) * 100))


def main(args):
    # Device configuration
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set the random seed manually for reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    name = 'JSMA'
    targeted = True
    state = vars(args).copy()
    state['attack_name'] = name
    state['targeted'] = targeted
    state['device'] = device
    if 'limit' in state['raw_model_location']:
        state['restricted'] = True
    else:
        state['restricted'] = False

    jsma = JSMAGeneration(**state)
    args_dir = '{}{}/{}/'.format(args.adv_saver, name, args.dataset)
    if os.path.isdir(args_dir):
        pass
    else:
        os.mkdir(args_dir)

    save_args(args, args_dir+'init.txt')
    
    print("Conducting {} attack".format(name))
    jsma.generate()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The JSMA Attack Generation')

    # common arguments
    parser.add_argument('--dataset', type=str, default='selected_imagenet')
    parser.add_argument('--raw_model_location', type=str, default='../selected_imagenet/models/nonaugmented/resnets_raw/resnet18_unpretrained.ckpt', help='the directory for the model')
    parser.add_argument('--clean_data_location', type=str, default='../selected_imagenet/target_npy/test/', help='the directory for the dataset that will be attacked')
    parser.add_argument('--adv_saver', type=str, default='../AdversarialExampleDatasets/nonaug-raw-18-unpre/test/',
                        help='the directory used to save the generated adversarial examples')
    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")

    # arguments for the particular attack
    parser.add_argument('--theta', type=float, default=1.0, help='theta') # how much a pixel in each epoch changes
    parser.add_argument('--gamma', type=float, default=0.1, help="gamma") 

    arguments = parser.parse_args()
    main(arguments)
