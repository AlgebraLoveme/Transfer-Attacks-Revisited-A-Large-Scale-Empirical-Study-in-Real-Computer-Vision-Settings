import numpy as np
import pickle

platform = input('Platform: ')
dataset = input('Dataset: ')
true_labels = np.load('../SurrogateDataset/{}/AECandidates/true_labels.npy'.format(dataset))

with open('Original/Dictionary/{}.pkl'.format(platform), 'rb') as f:
    class_map = pickle.load(f)

succ = 0
with open('Original/{}/{}.pkl'.format(dataset, platform), 'rb') as f:
    result = pickle.load(f)

for index in result.keys():
    pred = result[index]
    # Do some platform-specific matching logic
    if platform == 'aliyun':
        item = pred[0]['value']
        if index < 100:
            succ += (item != '正常')
        else:
            succ += (item == '正常')

    elif platform == 'baidu':
        item = pred['conclusion']
        if index < 100:
            succ += (item == '不合规')
        else:
            succ += (item == '合规')
            # if coarse_grained_match(tag=item['keyword'],
            #                         candidates=class_map[idx2class[dataset][true_labels[index]]]):
            #     succ += 1
            #     break

    elif platform == 'aws':
        if index < 100:
            succ += (len(pred['ModerationLabels']) > 0)
        else:
            succ += (len(pred['ModerationLabels']) == 0)
        # for item in pred['Labels']:
        #     name = item['Name']
        #     confidence = item['Confidence']
        #     if name in class_map[idx2class[dataset][true_labels[index]]] and confidence > 50:
        #         succ += 1
        #         break

    elif platform == 'gcv':
        item = pred.safe_search_annotation.adult
        if index < 100:
            succ += (item >= 3)
        else:
            succ += (item <= 3)
        # for item in pred.label_annotations:
        #     name = item.description
        #     confidence = item.score
        #     if name in class_map[idx2class[dataset][true_labels[index]]] and confidence > 0.5:
        #         succ += 1
        #         break

print('{}@{}: {}'.format(dataset, platform, succ / len(result)))

# if platform == 'aliyun':
#     with open('Original/ImageNet/aliyun.pkl', 'rb') as f:
#         result = pickle.load(f)
#     for index in result.keys():
#         pred = result[index]
#         for item in pred:
#             if fine_grained_match(tag=item['value'],
#                                   candidates=imagenet_tag_map[idx2class[imagenet_true_labels[index]]]):
#                 succ += 1
#                 break
#     print('{}: {}'.format(platform, succ / len(result)))
#
# elif platform == 'baidu':
#     with open('Original/ImageNet/baidu.pkl', 'rb') as f:
#         result = pickle.load(f)
#     for index in result.keys():
#         pred = result[index]['result']
#         for item in pred:
#             if fine_grained_match(tag=item['keyword'], candidates=imagenet_tag_map[idx2class[imagenet_true_labels[index]]]):
#                 succ += 1
#                 break
#     print('{}: {}'.format(platform, succ / len(result)))
