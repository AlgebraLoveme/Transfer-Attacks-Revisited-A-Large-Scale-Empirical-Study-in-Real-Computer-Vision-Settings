import argparse
import os
import pickle
import sys

# os.environ["http_proxy"] = "http://127.0.0.1:1080"
# os.environ["https_proxy"] = "http://127.0.0.1:1081"

sys.path.append('..')
import utils

from Tester.aliyun_tester import AliyunCloudTester
from Tester.baidu_tester import BaiduCloudTester
from Tester.aws_tester import AWSCloudTester
from Tester.gcv_tester import GCVCloudTester

tester_map = {
    'aliyun': AliyunCloudTester,
    'baidu': BaiduCloudTester,
    'aws': AWSCloudTester,
    'google': GCVCloudTester
}


def main(args):
    if args.dataset == 'ImageNet':
        task = 'classify'
    elif args.dataset == 'AdienceGenderG':
        task = 'gender'
    else:
        raise NotImplementedError('Unsupported dataset.')

    try:
        tester = tester_map[args.platform](task)
    except KeyError:
        raise NotImplementedError('Unsupported cloud platform: {}'.format(args.platform))

    for model_arch in ['resnet', 'inception', 'vgg']:
        data_types = ['raw', 'augmented', 'adversarial']
        if model_arch == 'resnet':
            depths = ['18', '34', '50']
        elif model_arch == 'vgg':
            depths = ['16']
        else:
            depths = ['V3']

        for data_type in data_types:
            for depth in depths:
                config_dir = utils.concat_dir(dataset=args.dataset, model=model_arch, data_type=data_type, depth=depth,
                                              pretrained=args.pretrained)

                surrogate_model_setting_dir = os.path.join(args.ae_root, config_dir)
                if not os.path.isdir(surrogate_model_setting_dir):
                    continue
                attacks = utils.listdir_without_hidden(surrogate_model_setting_dir)
                # attacks = ['UAP']
                for attack in attacks:
                    data_dir = os.path.join(args.ae_root, config_dir, attack)

                    result_dir = os.path.join(args.result_root, args.platform, config_dir, attack)
                    result_path = os.path.join(result_dir, 'original.pkl')
                    if os.path.exists(result_path):
                        print(f'Cloud result file already exist at: {result_path}')
                        continue

                    settings = {
                        'Architecture': model_arch,
                        'Data type': data_type,
                        'Depth': depth,
                        'Pretrained': args.pretrained,
                        'Attack': attack
                    }
                    for k, v in settings.items():
                        print('{}: {}'.format(k, v))

                    # print(f'\n\ndata_dir:{data_dir}')
                    # exit()
                    result = tester.test_one_dir(data_dir=data_dir)
                    os.makedirs(result_dir, exist_ok=True)
                    with open(result_path, 'wb') as f:
                        pickle.dump(result, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--platform', type=str, default='aliyun')
    parser.add_argument('--ae_root', type=str, default='../AdversarialExample/images')
    # images_cw2_kappa images
    parser.add_argument('--result_root', type=str, default='Results')
    parser.add_argument('--dataset', type=str, default='ImageNet')
    parser.add_argument('--pretrained', action='store_true')
    args = parser.parse_args()
    if args.platform == 'google':
        os.environ["http_proxy"] = "http://127.0.0.1:1080"
    main(args)
