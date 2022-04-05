import argparse

import torch

from AttackUtils import init
from Generation import Generation
from Methods.CW2 import CW2Attack


class CW2Generation(Generation):
    def __init__(self,
                 dataset, model, model_depth, dataset_type, pretrained,
                 clean_data_root, ae_save_root, model_root, device, name,
                 norm, attack_batch_size, kappa, init_const, lr, binary_search_steps, max_iterations,
                 lower_bound, upper_bound, **kwargs):
        super(CW2Generation, self).__init__(dataset, model, model_depth, dataset_type, pretrained,
                                            name, clean_data_root, ae_save_root, model_root, device)
        self.norm = norm
        self.attack_batch_size = attack_batch_size
        self.kappa = kappa
        self.init_const = init_const
        self.lr = lr
        self.binary_search_steps = binary_search_steps
        self.max_iter = max_iterations
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def _generate(self):
        attacker = CW2Attack(model=self.model, norm=self.norm, kappa=self.kappa, init_const=self.init_const,
                             lr=self.lr, binary_search_steps=self.binary_search_steps, max_iters=self.max_iter,
                             lower_bound=self.lower_bound, upper_bound=self.upper_bound, num_classes=self.num_classes)
        # get the targeted labels
        targets = self.target_labels
        # generating
        adv_samples = attacker.batch_perturbation(xs=self.raw_samples, ys_target=targets,
                                                  batch_size=self.attack_batch_size,
                                                  device=self.device)
        self.evaluate(adv_samples)


def main(args):
    device = init(args)
    print(device)
    cw2 = CW2Generation(**vars(args), device=device)
    cw2.generate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The CW2 Attack Generation')
    # File arguments
    parser.add_argument('--dataset', type=str, default='AdienceGenderG')
    parser.add_argument('--clean_data_root', type=str, default='../SurrogateDataset')
    parser.add_argument('--ae_save_root', type=str, default='../AdversarialExample/npys')
    parser.add_argument('--model', type=str, default='resnet')
    parser.add_argument('--model_root', type=str, default='../SurrogateModel/SavedModels')
    parser.add_argument('--dataset_type', type=str, default='raw')
    parser.add_argument('--model_depth', type=str, default='18')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--untargeted', action='store_true')
    parser.add_argument('--name', type=str, default='CW2')

    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")

    # arguments for the particular attack
    parser.add_argument('--kappa', type=float, default=0., help='the confidence of adversarial examples')
    parser.add_argument('--init_const', type=float, default=0.001,
                        help="the initial value of const c in the binary search.")
    parser.add_argument('--norm', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.02, help="the learning rate of gradient descent.")
    parser.add_argument('--max_iterations', type=int, default=100, help='maximum iteration')

    parser.add_argument('--lower_bound', type=float, default=0.0,
                        help='the minimum pixel value for examples (default=0.0).')
    parser.add_argument('--upper_bound', type=float, default=1.0,
                        help='the maximum pixel value for examples (default=1.0).')
    parser.add_argument('--binary_search_steps', type=int, default=10,
                        help="the binary search steps to find the optimal const.")
    parser.add_argument('--attack_batch_size', type=int, default=10,
                        help='the default batch size for adversarial example generation')

    arguments = parser.parse_args()

    # to test the transferability of CW2 attack under different settings of kappa,
    # we use the attack_name to record setting of kappa
    # specially, 'CW2' means that kappa is set to 0
    if arguments.kappa == 0.:
        arguments.name = 'CW2'
    else:
        arguments.name = f'CW2-kappa-{arguments.kappa}'
    print(f'arguments.kappa:{arguments.kappa}')

    main(arguments)
