import argparse

import torch

from AttackUtils import init
from Generation import Generation
from Methods.BLB import BLBAttack


class BLBGeneration(Generation):
    def __init__(self,
                 dataset, model, model_depth, dataset_type, pretrained,
                 clean_data_root, ae_save_root, model_root, device, name,
                 init_const, binary_search_steps, max_iterations, attack_batch_size, **kwargs):
        super(BLBGeneration, self).__init__(dataset, model, model_depth, dataset_type, pretrained,
                                            name, clean_data_root, ae_save_root, model_root, device)
        self.attack_batch_size = attack_batch_size
        self.init_const = init_const
        self.binary_search_steps = binary_search_steps
        self.max_iter = max_iterations

    def _generate(self):
        attacker = BLBAttack(model=self.model, init_const=self.init_const, binary_search_steps=self.binary_search_steps,
                             max_iterations=self.max_iter, num_classes=self.num_classes)
        # get the targeted labels
        targets = self.target_labels
        # generating
        adv_samples = attacker.batch_perturbation(xs=self.raw_samples, ys_target=targets,
                                                  batch_size=self.attack_batch_size,
                                                  device=self.device)
        self.evaluate(adv_samples)


def main(args):
    device = init(args)
    blb = BLBGeneration(**vars(args), device=device)
    blb.generate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BLB Attack Generation')

    # common arguments
    parser.add_argument('--dataset', type=str, default='AdienceGenderG')
    parser.add_argument('--clean_data_root', type=str, default='../SurrogateDataset')
    parser.add_argument('--ae_save_root', type=str, default='../AdversarialExample/npys')
    parser.add_argument('--model', type=str, default='vgg')
    parser.add_argument('--model_root', type=str, default='../SurrogateModel/SavedModels')
    parser.add_argument('--dataset_type', type=str, default='raw')
    parser.add_argument('--model_depth', type=str, default='16')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--untargeted', action='store_true')
    parser.add_argument('--name', type=str, default='BL-BFGS')

    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    parser.add_argument('--gpu_index', type=str, default='1', help="gpu index to use")

    # arguments for the particular attack
    parser.add_argument('--binary_search_steps', type=int, default=10,
                        help="The binary search steps to find the optimal const.")
    parser.add_argument('--max_iterations', type=int, default=100, help='maximum iteration')
    parser.add_argument('--init_const', type=float, default=0.01, help="The initial value of c in the binary search.")
    parser.add_argument('--attack_batch_size', type=int, default=2,
                        help='the default batch size for adversarial example generation')

    arguments = parser.parse_args()
    main(arguments)
