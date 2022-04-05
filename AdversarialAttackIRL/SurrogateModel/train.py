import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='vgg')
parser.add_argument('--depth', type=str, default='16')
parser.add_argument('--dataset', type=str, default='AdienceGenderG')
parser.add_argument('--data_root', type=str, default='../SurrogateDataset')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--augment', action='store_true')
parser.add_argument('--gpu_index', type=str, default='0')
parser.add_argument('--save_root', type=str, default='SavedModels')
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--pretrained_model_dir', type=str, default='PretrainedModels')
parser.add_argument('--epochs', type=int, default=50)
args = parser.parse_args()
# args.pretrained = True

os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu_index}'

sys.path.append('..')

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from utils import write_log_file, image_loader, model_map


def train(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, label)
        loss.backward()
        optimizer.step()


def evaluate(model, device, validation_loader):
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, label in validation_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss += F.cross_entropy(output, label, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()
    loss /= len(validation_loader.dataset)
    accuracy = 100 * correct / len(validation_loader.dataset)
    return loss, accuracy


def main(args):
    # device = 'cuda:{}'.format(args.gpu_index) if torch.cuda.is_available() else 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice:{device}\n')

    dataset_root = os.path.join(args.data_root, args.dataset)
    train_root = os.path.join(dataset_root, 'train')
    validation_root = os.path.join(dataset_root, 'validation')
    test_root = os.path.join(dataset_root, 'test')
    batch_size = args.batch_size
    augment = args.augment
    log_path = os.path.join('train_log', '{}+{}+{}+{}+{}.txt'.format(args.model, args.depth, args.dataset,
                                                                     'augmented' if augment else 'raw',
                                                                     'pretrained' if args.pretrained else 'unpretrained'))
    for k, v in vars(args).items():
        write_log_file(log_path, '{}: {}'.format(k, v))

    train_loader, validation_loader, test_loader = image_loader(train_root, validation_root, test_root, batch_size,
                                                                augment)
    write_log_file(log_path, 'Number of train samples: {}'.format(len(train_loader.dataset)))
    write_log_file(log_path, 'Number of validation samples: {}'.format(len(validation_loader.dataset)))
    num_classes = len(train_loader.dataset.classes)

    print('\nData loaders has been set!\n')

    model_name = args.model + args.depth
    try:
        model_generator = model_map[model_name]
    except KeyError:
        model_generator = None
        Exception('Invalid model {}'.format(model_name))

    if args.pretrained:
        model = model_generator(pretrained=False)
        pretrained_model_path = os.path.join(args.pretrained_model_dir, model_name + '.pth')
        model.load_state_dict(torch.load(pretrained_model_path))
        if args.model == 'resnet' or args.model == 'inception':
            model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=num_classes)
            if args.dataset != 'ImageNet':
                trainable_parameters = model.parameters()
            else:
                trainable_parameters = []
                train_layers = ['fc']
                for name, p in model.named_parameters():
                    for item in train_layers:
                        if item in name:
                            trainable_parameters.append(p)
        elif args.model == 'vgg':
            model.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )
            if args.dataset != 'ImageNet':
                trainable_parameters = model.parameters()
            else:
                trainable_parameters = []
                train_layers = ['classifier']
                for name, p in model.named_parameters():
                    for item in train_layers:
                        if item in name:
                            trainable_parameters.append(p)
        else:
            raise NotImplementedError
    else:
        model = model_generator(num_classes=num_classes)
        trainable_parameters = model.parameters()
    if args.model == 'inception':
        model.aux_logits = False
    model = model.to(device)

    print('\nModel generated!\n')

    if args.model == 'vgg':
        optimizer = optim.Adam(trainable_parameters, lr=1e-4, weight_decay=1e-6)
    else:
        optimizer = optim.Adam(trainable_parameters, lr=1e-3, weight_decay=1e-6)

    if args.augment:
        dataset_type = 'augmented'
    else:
        dataset_type = 'raw'

    model_save_dir = os.path.join(args.save_root, args.dataset, args.model, dataset_type, args.depth,
                                  'pretrained' if args.pretrained else 'unpretrained')
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, 'model.pth')

    # model.load_state_dict(torch.load(model_save_path))
    # test_loss, test_accu = evaluate(model, device, test_loader)
    # print('model_save_path:', model_save_path)
    # print('test_acc:', test_accu)
    # exit()

    best_valid_accu = 0

    print('\nStart training!\n')

    for epoch in range(args.epochs):
        print(f'Epoch:{epoch}/{args.epochs}')
        # reload dataset
        train_loader, validation_loader, test_loader = image_loader(train_root, validation_root, test_root, batch_size,
                                                                    augment)
        train(model, device, train_loader, optimizer)
        valid_loss, valid_accu = evaluate(model, device, validation_loader)
        if valid_accu > best_valid_accu:
            write_log_file(log_path, 'Epoch {}: Validation accuracy = {} > {}, saving.'.format(epoch + 1, valid_accu,
                                                                                               best_valid_accu))
            best_valid_accu = valid_accu
            torch.save(model.state_dict(), model_save_path)
        else:
            write_log_file(log_path, 'Epoch {}: Validation accuracy = {}.'.format(epoch, valid_accu))

    model.load_state_dict(torch.load(model_save_path))
    test_loss, test_accu = evaluate(model, device, test_loader)
    write_log_file(log_path, 'Testing Phase:\n\t loss = {}, accuracy = {}.'.format(test_loss, test_accu))


if __name__ == '__main__':
    main(args)
