import random

import numpy as np
import torch
from torch.autograd import Variable
import json
import argparse
import math


def tensor2variable(x=None, device=None, requires_grad=False):
    """

    :param x:
    :param device:
    :param requires_grad:
    :return:
    """
    x = x.to(device)
    return Variable(x, requires_grad=requires_grad)


def predict(model, samples, device='cpu'):
    """

    :param model:
    :param samples:
    :param device:
    :return:
    """
    model.eval()
    model = model.to(device)
    copy_samples = np.copy(samples)
    var_samples = tensor2variable(torch.from_numpy(copy_samples), device=device, requires_grad=False)
    with torch.no_grad():
        try:
            prediction = model(var_samples.float())
        except:
            # cuda out of memory, so we use mini-batch prediction
            prediction = []
            total_size = var_samples.shape[0]
            batch_size = int(total_size / 20)
            number_batch = int(math.ceil(total_size / batch_size))
            for index in range(number_batch):
                start = index * batch_size
                end = min((index + 1) * batch_size, total_size)
                batch_samples = var_samples[start:end]
                pred = model(batch_samples.float())
                prediction.append(pred)
            prediction = torch.cat(prediction)

    return prediction


def save_args(args, filename):
    with open(filename, 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def load_args(filename):
    with open(filename, 'r') as f:
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        args.__dict__ = json.load(f)
    return args


def init(args):
    # Device configuration
    device = torch.device('cuda:{}'.format(args.gpu_index) if torch.cuda.is_available() else 'cpu')
    # Set the random seed manually for reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    return device
