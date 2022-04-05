from datetime import datetime
import os
import pickle
from Tester.aliyun_tester import AliyunCloudTester
from utils import listdir_without_hidden
import stat

root_dir = 'Results/aliyun/ImageNet/resnet'
data_types = ['augmented', 'raw']
depths = ['18', '34', '50']
pretrained = 'pretrained'
tester = AliyunCloudTester(task='classify')
for dtype in data_types:
    for depth in depths:
        surrogate_dir = os.path.join(root_dir, dtype, depth, pretrained)
        attacks = listdir_without_hidden(surrogate_dir)
        for attck in attacks:
            if attck == 'UAP_OWN':
                continue
            if attck == 'DEEPFOOL':
                continue
            file_path = os.path.join(surrogate_dir, attck, 'original.pkl')
            fstat = os.stat(file_path)
            mtime = datetime.fromtimestamp(fstat.st_mtime)
            if mtime.month == 5 and mtime.day == 2:
                print('type: {}'.format(dtype))
                print('Depth: {}'.format(depth))
                print('Attack: {}'.format(attck))
                data_dir = os.path.join('../AdversarialExample/images/ImageNet/resnet', dtype, depth, pretrained, attck)
                result = tester.test_one_dir(data_dir)
                with open(file_path, 'wb') as f:
                    pickle.dump(result, f)
