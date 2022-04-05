# This util file is used to transfer Pytorch whole model to Pytorch model state_dict files

import torch
import os

src = './ImageNet/resnet/augmented/'

def traverse_dir(src):
    if os.path.isfile(src):
        model_to_dict(src)
    else:
        for dir in os.listdir(src):
            next_src = f'{src}/{dir}'
            traverse_dir(next_src)

def model_to_dict(filename):
    model = torch.load(filename)
    try:
        state_dict = model.state_dict()
        torch.save(state_dict, filename)
        print(filename, 'transfered')
    except:
        pass

traverse_dir(src)