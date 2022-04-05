import numpy as np
import matplotlib.pyplot as plt

dataset_name = 'AdienceGenderG'
# AdienceGenderG NudeNet

ae_candidates_dir = f'Your path here, e.g., ./TransferAttackIRL/SurrogateDataset/{dataset_name}/AECandidates/'
file_names = ['raw_samples.npy', 'target_labels.npy', 'true_labels.npy']

for file_name in file_names:
    _array = np.load(ae_candidates_dir + file_name)
    print(f'\nfile_name:{file_name}')
    if 'raw' in file_name:
        # check the 197st img
        # img_1 = _array[197].reshape((384, 384, 3))
        img_1 = _array[188].transpose((1, 2, 0))
        plt.imshow(img_1)
        plt.show()
        print(_array)
        print(f'max:{np.max(_array)}')
    else:
        print(_array)
    print(_array.shape)
