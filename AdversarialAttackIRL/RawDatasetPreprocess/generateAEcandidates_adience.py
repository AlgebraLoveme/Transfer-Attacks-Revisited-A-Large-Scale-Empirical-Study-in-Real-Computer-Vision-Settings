# male half, female half (100 per gender)
# target_labels is versus to true_labels
# shapes: [200,3,384,384] [200,] [200,]
import copy
import random

import numpy as np
from PIL import Image
import os


def get_all_img_file_path_list(path):
    file_list = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            # only store img files
            if '.jpg' in filename:
                file_list.append(os.path.join(home, filename))
    return file_list


dataset_name = 'AdienceGenderG'
class_num = 2
dict_class2label = {0: 'female', 1: 'male'}

adience_root = '/home/fuchong/TransferAttackIRL/SurrogateDataset/AdienceGenderG/'

test_img_paths = adience_root + 'test/'

ae_candidate_path = adience_root + 'AECandidates/'
if not os.path.exists(ae_candidate_path):
    os.makedirs(ae_candidate_path)

# get path list
female_img_paths_list = get_all_img_file_path_list(test_img_paths + 'female/')
male_img_paths_list = get_all_img_file_path_list(test_img_paths + 'male/')
img_paths_lists = [female_img_paths_list, male_img_paths_list]

# randomly select 100 imgs per gender as AE candidates
random_select_img_ids_one_gender = random.sample(range(len(female_img_paths_list)), 100)

# for each gender, read 100 imgs and store these 100 imgs to .npy files
img_array_all = None
for _class_gender in [0, 1]:
    str_gender = dict_class2label[_class_gender]
    img_paths_list = img_paths_lists[_class_gender]
    for img_id in random_select_img_ids_one_gender:
        img_path = img_paths_list[img_id]
        img = Image.open(img_path)
        img_array = np.array(img)
        img_array_transposed = img_array.transpose().transpose((0, 2, 1))
        # print(img_array_transposed.shape)
        # print(img_array.shape)
        # exit()
        img_array_tr_exp = np.expand_dims(img_array_transposed, 0)
        if img_array_all is None:
            img_array_all = copy.deepcopy(img_array_tr_exp)
        else:
            img_array_all = np.concatenate((img_array_all, img_array_tr_exp), 0)
        print(f'gender({str_gender}), id({img_id})')
print(f'raw_samples.npy shape:{img_array_all.shape}')
# the img array should be within [0. 1.]
img_array_all = img_array_all / 255.
# the data type is float32
img_array_all = img_array_all.astype(np.float32)
np.save(ae_candidate_path + 'raw_samples.npy', img_array_all)

ones_100 = np.ones(shape=100, dtype=int)
zeros_100 = np.zeros(shape=100, dtype=int)

true_labels_array = np.concatenate((zeros_100, ones_100))
target_labels_array = np.concatenate((ones_100, zeros_100))

print(f'true_labels.npy shape:{true_labels_array.shape}')
np.save(ae_candidate_path + 'true_labels.npy', true_labels_array)
print(f'target_labels.npy shape:{target_labels_array.shape}')
np.save(ae_candidate_path + 'target_labels.npy', target_labels_array)

print("save .npy done")
