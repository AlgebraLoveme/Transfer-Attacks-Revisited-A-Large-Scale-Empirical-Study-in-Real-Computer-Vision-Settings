import argparse

from AttackUtils import init
from Generation import Generation
from Methods.FGSM import FGSMAttack


class FGSMGeneration(Generation):

    def __init__(self,
                 dataset, model, model_depth, dataset_type, pretrained,
                 clean_data_root, ae_save_root, model_root, device, name,
                 eps, attack_batch_size, **kwargs):
        super(FGSMGeneration, self).__init__(dataset, model, model_depth, dataset_type, pretrained,
                                             name, clean_data_root, ae_save_root, model_root, device)
        self.attack_batch_size = attack_batch_size
        self.epsilon = eps

    def _generate(self):
        attacker = FGSMAttack(model=self.model, epsilon=self.epsilon, num_classes=self.num_classes)
        adv_samples = attacker.batch_perturbation(xs=self.raw_samples, ys=self.true_labels,
                                                  batch_size=self.attack_batch_size,
                                                  device=self.device)
        self.evaluate(adv_samples)


def main(args):
    device = init(args)
    fgsm = FGSMGeneration(**vars(args), device=device)
    fgsm.generate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The FGSM Attack Generation')
    # File arguments
    parser.add_argument('--dataset', type=str, default='ImageNet')
    parser.add_argument('--clean_data_root', type=str, default='../SurrogateDataset')
    parser.add_argument('--ae_save_root', type=str, default='../AdversarialExample/npys')
    parser.add_argument('--model', type=str, default='resnet')
    parser.add_argument('--model_root', type=str, default='../SurrogateModel/SavedModels')
    parser.add_argument('--dataset_type', type=str, default='raw')
    parser.add_argument('--model_depth', type=str, default='18')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--untargeted', action='store_true')
    parser.add_argument('--name', type=str, default='FGSM')

    # common arguments
    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")

    # arguments for the particular attack
    parser.add_argument('--eps', type=float, default=0.05, help='the epsilon value of FGSM')
    parser.add_argument('--attack_batch_size', type=int, default=20,
                        help='the default batch size for adversarial example generation')

    arguments = parser.parse_args()
    main(arguments)
