import re

import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import math
import os
from bisect import bisect_left

from tqdm import tqdm

from Cloud.test_ae import tester_map
from Cloud.Tester.tester_utils import listdir_without_hidden, parse_ae_name


def load_model(dataset, model_path, device):
    if dataset == 'AdienceGenderG':
        num_classes = 2
    else:
        raise NotImplementedError('unsupported dataset')

    if 'resnet' in model_path.lower() and '18' in model_path.lower():
        model = torchvision.models.resnet18(num_classes=num_classes)
    elif 'resnet' in model_path.lower() and '34' in model_path.lower():
        model = torchvision.models.resnet34(num_classes=num_classes)
    elif 'resnet' in model_path.lower() and '50' in model_path.lower():
        model = torchvision.models.resnet50(num_classes=num_classes)
    elif 'vgg' in model_path.lower():
        model = torchvision.models.vgg16(num_classes=num_classes)
    elif 'inception' in model_path.lower():
        model = torchvision.models.inception_v3(pretrained=False)
        model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=num_classes)
    else:
        raise NotImplementedError('unsupported model type')

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    if 'inception' in model_path.lower():
        model.aux_logits = False

    model = model.to(device)
    return model


def model_predict(model, samples, truncation, batches=20, device='cpu'):
    model.eval()
    model = model.to(device)
    samples = samples.to(device)
    # with torch.no_grad():
    # cuda out of memory, so we use minibatch prediction
    prediction = []
    total_size = samples.shape[0]
    batch_size = int(total_size / batches)
    number_batch = int(math.ceil(total_size / batch_size))
    for index in range(number_batch):
        start = index * batch_size
        end = min((index + 1) * batch_size, total_size)
        batch_samples = samples[start:end]
        pred = model(batch_samples.float())[:, :truncation]
        prediction.append(pred)
        # print(f'index: {index}')
    prediction = torch.cat(prediction)
    return prediction


def get_kappa(model, adv_data, true_label, adv_label, truncation, device):
    adv_data = torch.from_numpy(adv_data)
    preds = model_predict(model, adv_data, truncation, 20, device).cpu().numpy()
    lins = np.linspace(0, adv_data.shape[0] - 1, adv_data.shape[0]).astype('int')
    print('Local attack success', np.sum(np.argmax(preds, 1) != true_label))
    unmatch = np.sum(np.argmax(preds, 1) != adv_label)
    if unmatch:
        print(f"warning!!!!!! {unmatch} samples unmatch!!!")
    correct_pred = preds[lins, true_label].copy()
    preds[lins, true_label] = -1e10
    others_max = np.max(preds, axis=1)
    kappa = others_max - correct_pred
    print(f'mean(kappa) on local model: {np.mean(kappa)}')
    return kappa


def get_transformed_kappa(model, adv_data, true_label, adv_label, truncation, device):
    adv_data = torch.from_numpy(adv_data)
    preds = F.softmax(model_predict(model, adv_data, truncation, 20, device), dim=1).cpu().numpy()
    lins = np.linspace(0, adv_data.shape[0] - 1, adv_data.shape[0]).astype('int')
    print('Local attack success', np.sum(np.argmax(preds, 1) != true_label))
    unmatch = np.sum(np.argmax(preds, 1) != adv_label)
    if unmatch:
        print(f"warning!!!!!! {unmatch} samples unmatch!!!")
    correct_pred = preds[lins, true_label].copy()
    preds[lins, true_label] = -1e10
    others_max = np.max(preds, axis=1)
    kappa = others_max - correct_pred
    print(np.mean(kappa))
    return kappa


def _get_input_gradient_size(input_data, model, device):
    input_data = torch.from_numpy(input_data).float().to(device)
    input_data.requires_grad = True
    logits = model(input_data)
    preds = logits.argmax(dim=1)
    loss = F.cross_entropy(logits, preds)
    loss.backward()
    grad = input_data.grad.cpu().numpy()
    L2 = np.linalg.norm(grad.reshape(grad.shape[0], -1), axis=1, ord=2)
    return L2


def get_input_gradient_size(samples, model, device, batches=20):
    model.eval()
    model = model.to(device)
    # cuda out of memory, so we use minibatch prediction
    L2 = []
    total_size = samples.shape[0]
    batch_size = int(total_size / batches)
    number_batch = int(math.ceil(total_size / batch_size))
    for index in range(number_batch):
        start = index * batch_size
        end = min((index + 1) * batch_size, total_size)
        batch_samples = samples[start:end]
        L2.append(_get_input_gradient_size(batch_samples, model, device))
    L2 = np.concatenate(L2)
    print(f'mean of the L2 norm of input grad size on the targeted model: {np.mean(L2)}')
    return L2


def get_atk_results(adv_data, true_label, model, truncation, device):
    adv_data = torch.from_numpy(adv_data)
    preds = F.softmax(model_predict(model, adv_data, truncation, 20, device), dim=1).cpu().numpy()
    results_array = np.argmax(preds, 1) != true_label
    print('Attack success number on local model: ', np.sum(results_array))
    return results_array


def histogram(values, dividers):
    count = [0] * (1 + len(dividers))
    for element in values:
        i = bisect_left(dividers, element)
        count[i] += 1
    return count


def count_conditional_success(kappas, cloud_result, platform, dataset, true_label, img_ae_dir):
    cond_success, cond_failure = [], []
    processed_result = preprocess_result(dataset, platform, cloud_result, img_ae_dir)
    # True False NotAE
    for i in tqdm(range(200)):
        # within 200 samples
        try:
            cloud_transfer_succeed = processed_result[i]
        except:
            # indicates this sample does not pass local surrogate
            continue
        if cloud_transfer_succeed == 'NotAE':
            print('NotAE sample detected, skipping...')
            continue
        if not cloud_transfer_succeed:
            cond_failure.append(kappas[i])
        else:
            cond_success.append(kappas[i])
    return cond_success, cond_failure


def calculate_ratios(cond_success, cond_failure, dividers):
    cond_total = cond_success + cond_failure
    # remove less than 0 and greater than maximum value
    count_success = np.array(histogram(cond_success, dividers)[1:-1])
    count_total = np.array(histogram(cond_total, dividers)[1:-1])
    print(f'count_bins_success:{count_success}')
    print(f'count_bins_total:{count_total}')
    ratios = count_success / count_total
    print(f'transfer_success_ratios:{ratios}')
    return ratios


def preprocess_result(dataset, platform_name, cloud_result, img_ae_dir):
    if dataset == 'ImageNet':
        task = 'classify'
    elif dataset == 'NudeNet':
        task = 'monitor'
    elif dataset == 'AdienceGenderG':
        task = 'gender'
    else:
        raise NotImplementedError('Only support classification/monitor/gender tasks.')
    tester = tester_map[platform_name](task)

    all_imgs = listdir_without_hidden(img_ae_dir)
    if 'finish' in all_imgs:
        all_imgs.pop(all_imgs.index('finish'))

    all_imgs = sorted(all_imgs, key=lambda i: int(re.findall(r'\d+', i)[0]))

    result = []

    if dataset == 'AdienceGenderG':
        for img_path in all_imgs:
            img_id = parse_ae_name(img_path)['id']
            # print(f'img_id: {img_id}')
            res = tester.analyze_one_image_gender(img_path, cloud_result)
            if res is None:
                result.append('NotAE')
            else:
                _, is_success = res
                result.append(is_success)
    else:
        raise NotImplementedError('Not expected dataset:', dataset)

    return result
    # result: attack successfully or not
