# the input grad size (on the targeted model) of the sample &   transferability (on the targeted model)
import re
import sys

from tqdm import tqdm
import matplotlib.image as mpimg

sys.path.append('..')
import argparse
import torch
import utils
import os
import pickle as pkl
import MiscExperiments.utils_misc as utils_misc
import numpy as np
import matplotlib.pyplot as plt
from Cloud.Tester.tester_utils import listdir_without_hidden, parse_ae_name

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_index', type=str, default='2')
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
targeted_model_path = f'../SurrogateModel/SavedModels/{dataset}/vgg/raw/16/pretrained/model.pth'

if dataset == 'AdienceGenderG':
    truncation = 2
else:
    raise NotImplementedError('unsupported dataset')

attacks = ['BL-BFGS', 'CW2', 'DeepFool', 'UAP', 'FGSM', 'RFGSM', 'PGD', 'LLC', 'STEP_LLC']

input_grad_size_record = []
success_input_grad_size_record = []
failure_input_grad_size_record = []

targeted_model = utils_misc.load_model(dataset, targeted_model_path, device)


def count_conditional_success_for_input_grad_size(input_grad_sizes, atk_results):
    cond_success, cond_failure = [], []
    for i in tqdm(range(200)):
        # within 200 samples
        try:
            transfer_succeed = atk_results[i]
        except:
            # indicates this sample does not pass local surrogate
            continue
        if transfer_succeed == 'NotAE':
            print('NotAE sample detected, skipping...')
            continue
        if not transfer_succeed:
            cond_failure.append(input_grad_sizes[i])
        else:
            cond_success.append(input_grad_sizes[i])
    return cond_success, cond_failure


if not args.use_existing_pkl_record:
    for model_arch in ['inception', 'vgg', 'resnet']:
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

                        # get input grad size on the targeted model
                        input_grad_size_200 = utils_misc.get_input_gradient_size(ae_data, targeted_model, device)
                        # print(f'200 input_grad_size ndarray: {input_grad_size_200}')
                        input_grad_size_record += input_grad_size_200.tolist()

                        # get atk_results_raw on the targeted model
                        with torch.no_grad():
                            atk_results_raw = utils_misc.get_atk_results(ae_data, ae_true_label, targeted_model, truncation,
                                                                         device)

                        # get transfer success status
                        atk_results = []  # True False NotAE
                        all_imgs = listdir_without_hidden(ae_img_data_dir)
                        if 'finish' in all_imgs:
                            all_imgs.pop(all_imgs.index('finish'))
                        all_imgs = sorted(all_imgs, key=lambda i: int(re.findall(r'\d+', i)[0]))
                        if dataset == 'AdienceGenderG':
                            for img_path in all_imgs:
                                img_info = parse_ae_name(img_path)
                                if img_info['true_label'] == img_info['adv_label']:
                                    atk_results.append('NotAE')
                                else:
                                    try:
                                        if mpimg.imread(os.path.join(ae_img_data_dir, img_path)).max() == 0.:
                                            atk_results.append('NotAE')
                                            continue
                                    except:
                                        atk_results.append('NotAE')
                                        continue
                                    is_success = atk_results_raw[img_info['id']]
                                    atk_results.append(is_success)
                        else:
                            raise NotImplementedError('Unexpected dataset.')

                        cond_success, cond_failure = count_conditional_success_for_input_grad_size(input_grad_size_200,
                                                                                                   atk_results)
                        success_input_grad_size_record += cond_success
                        failure_input_grad_size_record += cond_failure
                        print(f'transfer success num:{len(cond_success)}')
                        print(f'transfer failure num:{len(cond_failure)}\n')
        #             break
        #         break
        #     break
        # break
    file1 = open(f'success_input_grad_size_record.pkl', 'wb')
    pkl.dump(success_input_grad_size_record, file1)
    file2 = open(f'failure_input_grad_size_record.pkl', 'wb')
    pkl.dump(failure_input_grad_size_record, file2)

file1 = open(f'success_input_grad_size_record.pkl', 'rb')
success_input_grad_size_record = pkl.load(file1)
file2 = open(f'failure_input_grad_size_record.pkl', 'rb')
failure_input_grad_size_record = pkl.load(file2)

# plot
bar_length = 0.2
low_bar = -0.1
high_bar = 1.1

dividers = np.arange(low_bar, high_bar, bar_length)

success = success_input_grad_size_record
failure = failure_input_grad_size_record
ratios = utils_misc.calculate_ratios(success, failure, dividers)
plt.plot(ratios)

ticks = [0, 1, 2, 3, 4, 5]
plt.xticks(ticks, ["[{:.1f}, {:.1f}]".format(bar_length * i, bar_length * (i + 1)) for i in ticks], size=12)
plt.xlabel(f"Input gradient size")
legend = plt.legend(loc='best', bbox_to_anchor=(1, 1))
# export_legend(legend)
plt.savefig(f"./sample_input_grad_size_transferability_local_targeted_model.pdf", bbox_inches='tight')
plt.close()
