import argparse
import numpy as np
import os

import Methods.UAP_own as UAP_own
from AttackUtils import init
from Generation import Generation


class UAPGeneration(Generation):
    def __init__(self,
                 dataset, model, model_depth, dataset_type, pretrained, clean_data_root, ae_save_root, model_root,
                 device, name,
                 max_iter_uni, frate, eps, overshoot, max_iter_deepfool, extra_data,
                 **kwargs):
        super(UAPGeneration, self).__init__(dataset, model, model_depth, dataset_type, pretrained,
                                            name, clean_data_root, ae_save_root, model_root, device)

        self.max_iter_uni = max_iter_uni
        self.fooling_rate = frate
        self.epsilon = eps
        self.extra_data = extra_data

        self.overshoot = overshoot
        self.max_iter_deepfool = max_iter_deepfool

    def _generate(self):
        attacker = UAP_own.UniversalAttack(model=self.model, fooling_rate=self.fooling_rate,
                                           max_iter_universal=self.max_iter_uni,
                                           epsilon=self.epsilon, overshoot=self.overshoot,
                                           max_iter_deepfool=self.max_iter_deepfool, num_classes=self.num_classes)
        universal_perturbation = attacker.universal_perturbation(dataset=self.raw_samples, device=self.device,
                                                                 true_labels=self.true_labels)
        universal_perturbation = universal_perturbation.cpu().numpy()
        np.save(os.path.join(self.ae_save_dir, 'uap.npy'), universal_perturbation)
        adv_samples = attacker.perturbation(xs=self.raw_samples, uni_pert=universal_perturbation, device=self.device)
        self.evaluate(adv_samples)


def main(args):
    device = init(args)
    df = UAPGeneration(**vars(args), device=device)
    df.generate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The UAP Attack Generation')
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
    parser.add_argument('--name', type=str, default='UAP')

    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    parser.add_argument('--gpu_index', type=str, default='3', help="gpu index to use")

    # arguments for the particular attack
    parser.add_argument('--frate', type=float, default=1.0, help="the fooling rate")
    parser.add_argument('--eps', type=float, default=0.05, help='controls the magnitude of the perturbation')
    parser.add_argument('--max_iter_uni', type=int, default=10, help="the maximum iterations for UAP")
    parser.add_argument('--extra_data', type=bool, default=False, help='whether use extra data')

    parser.add_argument('--overshoot', type=float, default=0.02, help='the overshoot parameter for DeepFool')
    parser.add_argument('--max_iter_deepfool', type=int, default=100, help='the maximum iterations for DeepFool')

    arguments = parser.parse_args()
    main(arguments)
