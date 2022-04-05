import argparse
import os
import random
import sys

parser = argparse.ArgumentParser(description='The NAT Defenses')
# common arguments
parser.add_argument('--model', type=str, default='resnet')
parser.add_argument('--depth', type=str, default='18')
parser.add_argument('--dataset', type=str, default='ImageNet')
parser.add_argument('--data_root', type=str, default='../SurrogateDataset')
parser.add_argument('--augment', action='store_true')
parser.add_argument('--save_root', type=str, default='../SurrogateModel/SavedModels')
parser.add_argument('--pretrained_model_dir', type=str, default='../SurrogateModel/PretrainedModels')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--name', type=str, default='NAT')
parser.add_argument('--pretrained', action='store_true')

parser.add_argument('--gpu_index', type=str, default='2', help="gpu index to use")
parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')

# parameters for the NAT Defense
parser.add_argument('--adv_ratio', type=float, default=0.5,
                    help='the weight of adversarial example when adversarial training')
parser.add_argument('--clip_min', type=float, default=0.0, help='the min of epsilon allowed')
parser.add_argument('--clip_max', type=float, default=0.3, help='the max of epsilon allowed')
parser.add_argument('--eps_mu', type=int, default=0, help='the \mu value of normal distribution for epsilon')
parser.add_argument('--eps_sigma', type=int, default=50, help='the \sigma value of normal distribution for epsilon')
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-6)

arguments = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = f'{arguments.gpu_index}'

import torch
import numpy as np
sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))
from utils import image_loader, concat_dir, model_map
from Defenses.DefenseMethods.NAT import NATDefense


def main(args):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Set the random seed manually for reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config_dir = concat_dir(dataset=args.dataset, model=args.model, data_type='adversarial', depth=args.depth,
                            pretrained=args.pretrained)

    data_root_dir = os.path.join(args.data_root, args.dataset)
    train_dir = os.path.join(data_root_dir, 'train')
    valid_dir = os.path.join(data_root_dir, 'validation')
    test_dir = os.path.join(data_root_dir, 'test')
    train_loader, valid_loader, test_loader = image_loader(train_root=train_dir, validation_root=valid_dir,
                                                           test_root=test_dir, batch_size=args.batch_size)

    model_name = args.model + args.depth
    model_generator = model_map[model_name]
    if args.dataset == 'ImageNet':
        num_classes = 10
    else:
        num_classes = 2
    model = model_generator(pretrained=False).to(device)
    if args.pretrained:
        pretrained_model_path = os.path.join(args.pretrained_model_dir, model_name + '.pth')
        model.load_state_dict(torch.load(pretrained_model_path))
    model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=num_classes)

    if args.pretrained and args.dataset == 'ImageNet':
        trainable_parameters = []
        train_layers = ['fc']
        for name, p in model.named_parameters():
            for item in train_layers:
                if item in name:
                    trainable_parameters.append(p)
    else:
        trainable_parameters = model.parameters()

    model = model.to(device)

    nat = NATDefense(model=model, dataset=args.dataset, num_epochs=args.num_epochs, batch_size=args.batch_size,
                     lr=args.lr, weight_decay=args.weight_decay, adv_ratio=args.adv_ratio, min=args.clip_min,
                     max=args.clip_max, mu=args.eps_mu, sigma=args.eps_sigma, device=device,
                     trainable_params=trainable_parameters)
    # make sure the save path exist
    save_path = os.path.join(args.save_root, config_dir)
    print(f'model save pth:{save_path}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    nat.defense(train_loader=train_loader, validation_loader=valid_loader, test_loader=test_loader,
                model_save_path=os.path.join(args.save_root, config_dir, 'model.pth'),
                log_path='{}{}_{}.txt'.format(args.model, args.depth, args.dataset))


if __name__ == '__main__':
    main(arguments)
