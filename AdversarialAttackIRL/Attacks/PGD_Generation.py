import argparse

from AttackUtils import init
from Generation import Generation
from Methods.PGD import PGDAttack


class PGDGeneration(Generation):

    def __init__(self,
                 dataset, model, model_depth, dataset_type, pretrained,
                 clean_data_root, ae_save_root, model_root, device, name,
                 attack_batch_size, eps, eps_iter, num_steps, **kwargs):
        super(PGDGeneration, self).__init__(dataset, model, model_depth, dataset_type, pretrained,
                                            name, clean_data_root, ae_save_root, model_root, device)
        self.attack_batch_size = attack_batch_size

        self.epsilon = eps
        self.epsilon_iter = eps_iter
        self.num_steps = num_steps

    def _generate(self):
        attacker = PGDAttack(model=self.model, epsilon=self.epsilon, eps_iter=self.epsilon_iter,
                             num_steps=self.num_steps, num_classes=self.num_classes)
        adv_samples = attacker.batch_perturbation(xs=self.raw_samples, ys=self.true_labels,
                                                  batch_size=self.attack_batch_size, device=self.device)
        self.evaluate(adv_samples)


def main(args):
    device = init(args)
    pgd = PGDGeneration(**vars(args), device=device)
    pgd.generate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The PGD Attack Generation')

    # common arguments
    parser.add_argument('--dataset', type=str, default='ImageNet')
    parser.add_argument('--clean_data_root', type=str, default='../SurrogateDataset')
    parser.add_argument('--ae_save_root', type=str, default='../AdversarialExample/npys')
    parser.add_argument('--model', type=str, default='resnet')
    parser.add_argument('--model_root', type=str, default='../SurrogateModel/SavedModels')
    parser.add_argument('--dataset_type', type=str, default='raw')
    parser.add_argument('--model_depth', type=str, default='18')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--untargeted', action='store_true')
    parser.add_argument('--name', type=str, default='PGD')

    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")

    # arguments for the particular attack
    parser.add_argument('--eps', type=float, default=0.05, help='the max epsilon value that is allowed to be perturbed')
    parser.add_argument('--eps_iter', type=float, default=0.01, help='the one iterative eps of PGD')
    parser.add_argument('--num_steps', type=int, default=10, help='the number of perturbation steps')
    parser.add_argument('--attack_batch_size', type=int, default=10,
                        help='the default batch size for adversarial example generation')

    arguments = parser.parse_args()
    main(arguments)
