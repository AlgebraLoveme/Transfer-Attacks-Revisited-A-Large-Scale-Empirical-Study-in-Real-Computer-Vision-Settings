import os
from Tester.baidu_tester import BaiduCloudTester
from utils import listdir_without_hidden
from tqdm import tqdm

dir = '../Original/ImageNet_AECandidates'
all_images = listdir_without_hidden(dir)
tester = BaiduCloudTester('classify')

results = {}
for image in tqdm(all_images):
    iid = int(image.split('.')[0])
    results[iid] = tester.test_one_image(os.path.join(dir, image))

with open('../Original/baidu_ImageNet.pkl', 'wb') as f:
    import pickle
    pickle.dump(results, f)