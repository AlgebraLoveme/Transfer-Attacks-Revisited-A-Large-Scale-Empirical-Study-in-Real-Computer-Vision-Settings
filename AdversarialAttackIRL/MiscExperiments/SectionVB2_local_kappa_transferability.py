# _sample_kappa_transferability
import sys

sys.path.append('..')
import argparse
import torch
import utils
import os
import pickle as pkl
import MiscExperiments.utils_misc as utils_misc
import numpy as np
import matplotlib.pyplot as plt
from Cloud.Tester.tester_utils import listdir_without_hidden

parser = argparse.ArgumentParser()
parser.add_argument('--kappa_mode', type=str, default='SIK', choices=['SSK', 'SIK'])
parser.add_argument('--gpu_index', type=str, default='3')
parser.add_argument('--use_existing_pkl_record', action='store_true')
args = parser.parse_args()
# args.use_existing_pkl_record = True

# Device configuration
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

dataset = 'AdienceGenderG'
ae_img_root = '../AdversarialExample/images'
ae_npy_root = '../AdversarialExample/npys'
cloud_result_root = '../Cloud/Results'
surrogate_model_root = '../SurrogateModel/SavedModels'
kappa_mode = args.kappa_mode  # SSK SIK-softmax-transformed

if dataset == 'AdienceGenderG':
    truncation = 2
else:
    raise NotImplementedError('unsupported dataset')

attacks = ['BL-BFGS', 'CW2', 'DeepFool', 'UAP', 'FGSM', 'RFGSM', 'PGD', 'LLC', 'STEP_LLC']
platforms = ['aliyun', 'aws', 'baidu']

kappa_record_dict = {}
success_kappas_record_dict = {}
failure_kappas_record_dict = {}
for platform in platforms:
    kappa_record_dict[platform] = []
    success_kappas_record_dict[platform] = []
    failure_kappas_record_dict[platform] = []

if not args.use_existing_pkl_record:
    for model_arch in ['inception', 'resnet', 'vgg']:
        if model_arch == 'resnet':
            data_types = ['raw', 'augmented', 'adversarial']
            depths = ['18', '34', '50']
            pretrain = [True, False]
        elif model_arch == 'vgg':
            data_types = ['raw']
            depths = ['16']
            pretrain = [True]
        else:
            data_types = ['raw']
            depths = ['V3']
            pretrain = [True]

        for data_type in data_types:
            for depth in depths:
                for if_pretrain in pretrain:
                    config_dir = utils.concat_dir(dataset=dataset, model=model_arch, data_type=data_type,
                                                  depth=depth,
                                                  pretrained=if_pretrain)

                    surrogate_model_path = os.path.join(surrogate_model_root, config_dir, 'model.pth')
                    # print(surrogate_model_path)
                    surrogate_model = utils_misc.load_model(dataset, surrogate_model_path, device)

                    for attack in attacks:
                        print(f'\nconfig: {config_dir} -- {attack}')
                        ae_img_data_dir = os.path.join(ae_img_root, config_dir, attack)
                        ae_npy_data_dir = os.path.join(ae_npy_root, config_dir, attack)
                        ae_data = np.load(os.path.join(ae_npy_data_dir, 'AE.npy'))
                        ae_true_label = np.load(os.path.join(ae_npy_data_dir, 'TrueLabels.npy'))
                        ae_adv_label = np.load(os.path.join(ae_npy_data_dir, 'AdvLabels.npy'))

                        # print(ae_img_data_dir)
                        # all_imgs = listdir_without_hidden(ae_img_data_dir)
                        # if 'finish' in all_imgs:
                        #     all_imgs.pop(all_imgs.index('finish'))
                        # print('img num: ', len(all_imgs))

                        # get kappa
                        with torch.no_grad():
                            if kappa_mode == 'SSK':
                                kappas = utils_misc.get_kappa(surrogate_model, ae_data, ae_true_label, ae_adv_label,
                                                              truncation, device)
                            elif kappa_mode == 'SIK':
                                kappas = utils_misc.get_transformed_kappa(surrogate_model, ae_data, ae_true_label,
                                                                          ae_adv_label, truncation, device)
                            else:
                                raise ValueError('Unexpected kappa computation mode.')
                        # print(f'200 kappas ndarray: {kappas}')

                        for platform in platforms:
                            print(f'platform: {platform}')
                            kappa_record_dict[platform] += kappas.tolist()

                            cloud_result_path = os.path.join(cloud_result_root, platform, config_dir, attack,
                                                             'original.pkl')
                            # print(cloud_result_path)
                            with open(cloud_result_path, 'rb') as f:
                                cloud_result = pkl.load(f)

                            # get transfer success status
                            cond_success, cond_failure = utils_misc.count_conditional_success(kappas, cloud_result,
                                                                                              platform, dataset,
                                                                                              ae_true_label,
                                                                                              ae_img_data_dir)
                            success_kappas_record_dict[platform] += cond_success
                            failure_kappas_record_dict[platform] += cond_failure
                            print(f'transfer success num:{len(cond_success)}')
                            print(f'transfer failure num:{len(cond_failure)}\n')
        #             break
        #         break
        #     break
        # break
    file1 = open(f'{kappa_mode}-success_kappas_record_dict.pkl', 'wb')
    pkl.dump(success_kappas_record_dict, file1)
    file2 = open(f'{kappa_mode}-failure_kappas_record_dict.pkl', 'wb')
    pkl.dump(failure_kappas_record_dict, file2)

file1 = open(f'{kappa_mode}-success_kappas_record_dict.pkl', 'rb')
success_kappas_record_dict = pkl.load(file1)
file2 = open(f'{kappa_mode}-failure_kappas_record_dict.pkl', 'rb')
failure_kappas_record_dict = pkl.load(file2)

# plot for each platform
if kappa_mode == 'SSK':
    bar_length = 10
    low_bar = -5
    high_bar = 55
elif kappa_mode == 'SIK':
    bar_length = 0.2
    low_bar = -0.1
    high_bar = 1.1
else:
    raise ValueError('Unexpected kappa computation mode.')
dividers = np.arange(low_bar, high_bar, bar_length)

for platform in platforms:
    print(f'\nplotting for platform: {platform}')
    success = success_kappas_record_dict[platform]
    failure = failure_kappas_record_dict[platform]
    # print(f'cond_suc: {success}')
    # print(f'cond_fail: {failure}')
    ratios = utils_misc.calculate_ratios(success, failure, dividers)
    plt.plot(ratios, label=platform)

ticks = [0, 1, 2, 3, 4]
plt.xticks(ticks, ["[{:.1f}, {:.1f}]".format(bar_length * i, bar_length * (i + 1)) for i in ticks], size=12)
plt.xlabel(f"{kappa_mode}")
# plt.ylabel("P($\kappa$|S)/P($\kappa$)")
legend = plt.legend(loc='best', bbox_to_anchor=(1, 1))
# export_legend(legend)
plt.savefig(f"./{kappa_mode}-local_sample_kappa_transferability.pdf", bbox_inches='tight')
plt.close()
