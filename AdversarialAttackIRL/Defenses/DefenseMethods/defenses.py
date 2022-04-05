#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************
# @Time    : 2018/9/9 15:52
# @Author  : Xiang Ling
# @Lab     : nesa.zju.edu.cn
# @File    : defenses.py
# **************************************

import os
from abc import ABCMeta
from abc import abstractmethod
import torch


class Defense(object):
    __metaclass__ = ABCMeta

    def __init__(self, model=None, defense_name='NAT'):
        self.model = model
        self.defense_name = defense_name


# evaluate the model using validation dataset
def validation_evaluation(model, validation_loader, device):
    """

    :param model:
    :param validation_loader:
    :param device:
    :return:
    """
    model = model.to(device)
    model.eval()

    total = 0.0
    correct = 0.0
    with torch.no_grad():
        for index, (inputs, labels) in enumerate(validation_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total = total + labels.size(0)
            correct = correct + (predicted == labels).sum().item()
        ratio = correct / total
    print('validation dataset accuracy is {:.2f}%'.format(ratio*100))
    return ratio

