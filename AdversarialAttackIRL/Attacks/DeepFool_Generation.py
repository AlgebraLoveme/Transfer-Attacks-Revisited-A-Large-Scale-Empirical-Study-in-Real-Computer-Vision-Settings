import argparse

from AttackUtils import init
from Generation import Generation
from Methods.DEEPFOOL import DeepFoolAttack


class DeepFoolGeneration(Generation):
    def __init__(self,
                 dataset, model, model_depth, dataset_type, pretrained,
                 clean_data_root, ae_save_root, model_root, device, name,
                 overshoot, max_iters, **kwargs):
        super(DeepFoolGeneration, self).__init__(dataset, model, model_depth, dataset_type, pretrained,
                                                 name, clean_data_root, ae_save_root, model_root, device)
        self.overshoot = overshoot  # i.e. eta in the original paper
        self.max_iters = max_iters

    def _generate(self):
        attacker = DeepFoolAttack(model=self.model, overshoot=self.overshoot, max_iters=self.max_iters,
                                  num_classes=self.num_classes)
        adv_samples = attacker.perturbation(xs=self.raw_samples, device=self.device)
        self.evaluate(adv_samples)


def main(args):
    device = init(args)
    df = DeepFoolGeneration(**vars(args), device=device)
    df.generate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The DeepFool Attack Generation')
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
    parser.add_argument('--name', type=str, default='DeepFool')

    # common arguments
    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    parser.add_argument('--gpu_index', type=str, default='3', help="gpu index to use")

    # arguments for the particular attack
    parser.add_argument('--max_iters', type=int, default=100, help="the max iterations")
    parser.add_argument('--overshoot', type=float, default=0.02, help='the overshoot')

    arguments = parser.parse_args()
    main(arguments)
