label_map = {
    0: 'baseball',
    1: 'computer_keyboard',
    2: 'container',
    3: 'engine',
    4: 'helicopter',
    5: 'microphone',
    6: 'musical_instrument',
    7: 'remote_control',
    8: 'toilet',
    9: 'weapon'
}
import numpy as np
import pickle
# threshold = 50
threshold = 0.5
true_labels = np.load('../SurrogateDataset/ImageNet/AECandidates/true_labels.npy')
with open('baidu_orig_dict.pkl', 'rb') as f:
    results = pickle.load(f)

aws_dict = {}
for i in range(200):
    pred = results[i]['result']
    true_label = label_map[true_labels[i]]
    for item in pred:
        if item['score'] < threshold:
            continue
        if true_label in aws_dict and item['keyword'] not in aws_dict[true_label]:
            aws_dict[true_label].append(item['keyword'])
        else:
            aws_dict[true_label] = [item['keyword']]

with open('baidu_dict.pkl', 'wb') as f:
    pickle.dump(aws_dict, f)