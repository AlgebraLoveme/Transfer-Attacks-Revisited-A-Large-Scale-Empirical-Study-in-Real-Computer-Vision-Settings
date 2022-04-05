import os
import numpy as np
import time
import csv

dataset = "ImageNet"
origin_dir = f'/home/wsz/newdisk_wsz/AdvEmpirical/TransferAttack/SurrogateDataset/{dataset}/AECandidates'
AE_dir = f'/home/wsz/newdisk_wsz/AdvEmpirical/TransferAttack/AdversarialExample/npys/{dataset}/resnet'
conditions = os.listdir(AE_dir)


def load_adv_data(root):
    adv_data = np.load(root + '/AE.npy')
    adv_label = np.load(root + '/AdvLabels.npy')
    return adv_data, adv_label


def cal_effective_distortion(clean_data, adv_data, true_label, adv_label):
    '''
    calculate mean L1, L2, L_inf norms of perturbation of successful adversaries
    '''
    # get successful adv
    mask = adv_label != true_label
    size = np.sum(mask)
    # reshape the image to 1 dimensional array
    clean_data = clean_data[mask].reshape(size, -1)
    adv_data = adv_data[mask].reshape(size, -1)

    perturbation = np.abs(adv_data - clean_data)
    print(perturbation.shape[0])
    # L1 =  np.mean(np.linalg.norm(perturbation, ord=1, axis=1))
    L2 = np.mean(np.linalg.norm(perturbation, ord=2, axis=1))
    Linf = np.mean(np.max(perturbation, 1))
    return np.round(np.array([L2, Linf]), 4)


clean_data = np.load(origin_dir + '/raw_samples.npy')
true_label = np.load(origin_dir + '/true_labels.npy')

w = csv.writer(open("norm_info.csv", 'w'))
for cond in conditions:
    p1 = AE_dir + f'/{cond}'
    for depth in os.listdir(p1):
        p2 = p1 + f'/{depth}'
        for pre in os.listdir(p2):
            p3 = p2 + f'/{pre}'
            for attack in os.listdir(p3):
                p4 = p3 + f'/{attack}'
                adv_data, adv_label = load_adv_data(p4)
                L2, Linf = cal_effective_distortion(clean_data, adv_data, true_label, adv_label)
                w.writerow([cond, depth, pre, attack, L2, Linf])
