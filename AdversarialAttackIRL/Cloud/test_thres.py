import json
import pathlib
import os
import matplotlib.pyplot as plt
import pickle
import seaborn

import numpy as np

from Cloud.Tester.aliyun_tester import AliyunCloudTester
from Cloud.Tester.aws_tester import AWSCloudTester
from Cloud.Tester.baidu_tester import BaiduCloudTester
from Cloud.Tester.gcv_tester import GCVCloudTester

# threshold = float(input('Threshold: '))
aliyun_tester = AliyunCloudTester('classify')
baidu_tester = BaiduCloudTester('classify')
aws_tester = AWSCloudTester('classify')
gcv_tester = GCVCloudTester('classify')

tester_dict = {
    'aliyun': aliyun_tester,
    'baidu': baidu_tester,
    'aws': aws_tester,
    'gcv': gcv_tester
}

original_preds = {}

with open('Original/ImageNet/aliyun.pkl', 'rb') as f:
    original_preds['aliyun'] = pickle.load(f)
with open('Original/ImageNet/baidu.pkl', 'rb') as f:
    original_preds['baidu'] = pickle.load(f)
with open('Original/ImageNet/aws.pkl', 'rb') as f:
    original_preds['aws'] = pickle.load(f)
with open('Original/ImageNet/gcv.pkl', 'rb') as f:
    original_preds['gcv'] = pickle.load(f)

true_labels = np.load('../SurrogateDataset/ImageNet/AECandidates/true_labels.npy')
label_map_dict = {}

# aws_dict = gcv_tester.build_label_dict(orig_preds=original_preds['gcv'], true_labels=true_labels, threshold=90)
#
# current_dir = pathlib.Path(__file__).resolve().parent
# aws_dict_json_path = current_dir / 'EQ_Dict/json/gcv_dict.json'
# with aws_dict_json_path.open(mode='w') as f:
#     json.dump(aws_dict, f)

thresholds = []
accuracies = {'aliyun': [], 'baidu': [], 'aws': [], 'gcv': []}
for t in range(0, 100, 5):
    threshold = float(t)
    for k in tester_dict.keys():
        # label_dict = tester_dict[k].build_label_dict(orig_preds=original_preds[k], true_labels=true_labels,
        #                                              threshold=threshold)
        accuracy = tester_dict[k].test_accuracy(label_dict=tester_dict[k].label_dict, original_preds=original_preds[k],
                                                true_labels=true_labels, threshold=threshold)
        accuracies[k].append(accuracy)
    thresholds.append(threshold)

# equivalence_dict_map = {}
# threshold_platform = {'aliyun': 0, 'baidu': 0, 'aws': 95, 'gcv': 95}
# for k in tester_dict.keys():
#     equivalence_dict_map[k] = tester_dict[k].build_label_dict(orig_preds=original_preds[k], true_labels=true_labels,
#                                                               threshold=threshold_platform[k])

# eq_dict_dir = pathlib.Path('EQ_Dict/json')
# if not os.path.isdir(eq_dict_dir):
#     os.mkdir(eq_dict_dir)
# gcv_label_dict = gcv_tester.build_label_dict(orig_preds=original_preds['gcv'], true_labels=true_labels, threshold=90)
# with (eq_dict_dir / 'gcv_dict.json').open(mode='w') as f:
#     json.dump(gcv_label_dict, f)

legend_name = {
    'aliyun': 'Alibaba',
    'baidu': 'Baidu',
    'aws': 'AWS',
    'gcv': 'Google'
}

plt.style.use('ggplot')

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (8, 6),
         'axes.labelsize': 20,
         'axes.titlesize':20,
         'axes.labelcolor':"#000000",
         'xtick.labelsize':16,
         'ytick.labelsize':16,
         'font.weight':"normal",
         'xtick.color':"#000000",
         'ytick.color':"#000000",
         'axes.labelweight':'normal'}
pylab.rcParams.update(params)
plt.tight_layout()
fig, ax = plt.subplots()
for k in tester_dict.keys():
    # with open(eq_dict_dir + '/{}_dict.json'.format(k), 'w') as f:
    #     json.dump(equivalence_dict_map[k], f, ensure_ascii=False, indent=2)
    ax.plot(thresholds, accuracies[k], label=legend_name[k])
ax.set_xlabel('Threshold(%)')
ax.set_ylabel('Accuracy')
ax.legend()
plt.savefig('thresholds.eps', format='eps')
plt.show()
