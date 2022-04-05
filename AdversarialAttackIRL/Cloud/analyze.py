import argparse
import os
import pickle
import sys

sys.path.append('..')

import utils
from test_ae import tester_map

metrics_map = {
    'ImageNet': ['misclassification_rate', 'matching_rate', 'total', 'mis_ratio', 'match_ratio'],
    'NudeNet': ['misclassification_rate', 'porn2safe_rate', 'safe2porn_rate', 'total', 'nr_porn', 'nr_safe',
                'mis_ratio', 'porn2safe_rate_100', 'safe2porn_rate_100'],
    'AdienceGenderG': ['misclassification_rate', 'female2male_rate', 'male2female_rate', 'total', 'nr_female',
                       'nr_male',
                       'mis_ratio', 'female2male_rate_100', 'male2female_rate_100']
}


def main(args):
    task = ''
    if args.dataset == 'ImageNet':
        task = 'classify'
    elif args.dataset == 'AdienceGenderG':
        task = 'gender'
    else:
        raise NotImplementedError('Only support classification/gender tasks.')
    tester = tester_map[args.platform](task)
    if args.analyze_kappa:
        f = open(os.path.join(args.analyze_root, '{}_{}_cw2_kappa.csv'.format(args.platform, args.dataset)), 'w')
    else:
        f = open(os.path.join(args.analyze_root, '{}_{}.csv'.format(args.platform, args.dataset)), 'w')
    f.write('Architecture,Data_type,Depth,Pretrained,Attack,')
    # Print sheet header
    metrics = metrics_map[args.dataset]
    for metric in metrics:
        f.write('{},'.format(metric))
    f.write('\n')

    for model_arch in ['resnet', 'inception', 'vgg']:
        data_types = ['raw', 'augmented', 'adversarial']
        if model_arch == 'resnet':
            depths = ['18', '34', '50']
            pretrain = [True, False]
        elif model_arch == 'vgg':
            depths = ['16']
            pretrain = [True]
        else:
            depths = ['V3']
            pretrain = [True]

        os.makedirs(args.analyze_root, exist_ok=True)

        if args.analyze_kappa:
            selected_attacks = ['CW2'] * 15
            kappa_values = list(range(6, 21))
            for kappa_value_id in range(15):
                kappa_value = kappa_values[kappa_value_id] * 10.
                selected_attacks[kappa_value_id] += (f'-kappa-' + str(kappa_value))
        else:
            selected_attacks = ['BL-BFGS', 'CW2', 'DeepFool', 'UAP', 'FGSM', 'RFGSM', 'PGD', 'LLC', 'STEP_LLC']

        for data_type in data_types:
            for depth in depths:
                for if_pretrain in pretrain:
                    config_dir = utils.concat_dir(dataset=args.dataset, model=model_arch, data_type=data_type,
                                                  depth=depth,
                                                  pretrained=if_pretrain)
                    surrogate_model_setting_dir = os.path.join(args.ae_root, config_dir)
                    if not os.path.isdir(surrogate_model_setting_dir):
                        continue
                    attacks = os.listdir(os.path.join(args.ae_root, config_dir))
                    for attack in attacks:
                        if attack not in selected_attacks:
                            continue
                        data_dir = os.path.join(args.ae_root, config_dir, attack)
                        result_dir = os.path.join(args.result_root, args.platform, config_dir, attack)
                        result_path = os.path.join(result_dir, 'original.pkl')
                        if not os.path.exists(result_path):
                            continue

                        settings = {
                            'Architecture': model_arch,
                            'Data_type': data_type,
                            'Depth': depth,
                            'Pretrained': if_pretrain,
                            'Attack': attack
                        }
                        print('--one dir--')
                        result = tester.analyze(data_dir=data_dir, result_path=result_path)

                        for v in settings.values():
                            f.write('{},'.format(v))
                        for metric in metrics:
                            f.write('{},'.format(result[metric]))
                        f.write('\n')

    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--platform', type=str, default='aliyun')
    parser.add_argument('--dataset', type=str, default='ImageNet')
    parser.add_argument('--ae_root', type=str, default='../AdversarialExample/images')
    # images_cw2_kappa images
    parser.add_argument('--result_root', type=str, default='Results')
    parser.add_argument('--analyze_root', type=str, default='Analysis')
    parser.add_argument('--analyze_kappa', action='store_true')
    args = parser.parse_args()
    # args.analyze_kappa = True
    main(args)
