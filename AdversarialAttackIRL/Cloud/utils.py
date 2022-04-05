import os

import torch
import torchvision
from torchvision import transforms, models as models


def image_loader(train_root, validation_root, test_root, batch_size=32, augment=False):
    train_transform = validation_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    if augment:
        trans = [
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomAffine(30, translate=(0.1,0.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomPerspective(),
            transforms.RandomRotation(30),
            transforms.RandomVerticalFlip(),
        ]
        train_transform = transforms.Compose([
                *[transforms.RandomApply([T], p=0.5) for T in trans],
                transforms.ToTensor(),
            ])
    train_data = torchvision.datasets.ImageFolder(train_root, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_data = torchvision.datasets.ImageFolder(validation_root, transform=validation_transform)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=True)
    test_data = torchvision.datasets.ImageFolder(test_root, transform=validation_transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, validation_loader, test_loader


def concat_dir(dataset: str, model: str, data_type: str, depth: str, pretrained: bool):
    return os.path.join(dataset, model, data_type, '{}'.format(depth), 'pretrained' if pretrained else 'unpretrained')


def write_log_file(file_name_path, log_str, print_flag=True):
    if print_flag:
        print(log_str)
    if log_str is None:
        log_str = 'None'
    if os.path.isfile(file_name_path):
        with open(file_name_path, 'a+') as log_file:
            log_file.write(log_str + '\n')
    else:
        with open(file_name_path, 'w+') as log_file:
            log_file.write(log_str + '\n')


def listdir_without_hidden(path):
    orig = os.listdir(path)
    return [x for x in orig if x[0] != '.']


def parse_ae_name(name):
    # Parse photo file name
    photo_name_split = name.split('_')
    true_label_prompt_pos = photo_name_split.index('TrueLabel')
    adv_label_prompt_pos = photo_name_split.index('LocalAdvLabel')
    id_pos = photo_name_split.index('id')
    true_label_list = photo_name_split[true_label_prompt_pos + 1:adv_label_prompt_pos]
    adv_label_list = photo_name_split[adv_label_prompt_pos + 1:id_pos]
    true_label = ' '.join(true_label_list)
    adv_label = ' '.join(adv_label_list)
    id = int(photo_name_split[id_pos + 1:][0].split('.')[0])
    return {
        'true_label': true_label,
        'adv_label': adv_label,
        'id': id
    }


model_map = {
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'vgg11': models.vgg11,
    'vgg13': models.vgg13,
    'vgg16': models.vgg16,
    'vgg19': models.vgg19,
    'inceptionV3': models.inception_v3
}
