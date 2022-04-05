import base64
import datetime
import hashlib
import hmac
import json
import pathlib
import time

import numpy as np
import requests
import urllib3
import matplotlib.image as mpimg

from .tester import CloudTester
from .tester_utils import parse_ae_name, image_idx2class

from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.acs_exception.exceptions import ClientException
from aliyunsdkcore.acs_exception.exceptions import ServerException
from aliyunsdkcore.auth.credentials import AccessKeyCredential
from aliyunsdkcore.auth.credentials import StsTokenCredential
from aliyunsdkimm.request.v20170906.DetectImageFacesRequest import DetectImageFacesRequest
from aliyunsdkimagerecog.request.v20190930.TaggingImageRequest import TaggingImageRequest

AK_ID = 'Your access key ID here'
AK_SECRET = 'Your secret access key here'

CLASSIFY_URL = 'https://dtplus-cn-shanghai.data.aliyuncs.com/image/tag'  # Might be deprecated. Please check the latest yourself.
MONITOR_URL = 'https://dtplus-cn-shanghai.data.aliyuncs.com/image/porn'  # Might be deprecated. Please check the latest yourself.
# GENDER_URL = 'https://dtplus-cn-shanghai.data.aliyuncs.com/face/attribute' # Might be deprecated. Please check the latest yourself.

# some aliyun APIs can only use files on OSS, so you need to set the address of the OSS bucket and name of the project
OSS_BUCKET_GENDER = 'Your bucket name'
OSS_BUCKET_CLASSIFY = 'Your bucket name'
PROJECT_NAME = 'Your project name'


def to_md5_base64(body):
    hash = hashlib.md5()
    hash.update(bytes(body, encoding='ascii'))
    digest = hash.digest()
    return base64.b64encode(digest).strip()


def to_sha1_base64(stringToSign, secret):
    secret = bytes(secret, encoding='ascii')
    stringToSign = bytes(stringToSign, encoding='ascii')
    hmacsha1 = hmac.new(secret, stringToSign, hashlib.sha1)
    return base64.b64encode(hmacsha1.digest())


def generate_signature(body, header, url, ak_secret):
    body_md5 = to_md5_base64(body).decode('ascii')
    url_path = urllib3.util.parse_url(url)
    url_path = url_path.path
    stringToSign = 'POST' + '\n' + header['accept'] + '\n' + body_md5 + '\n' + header['content-type'] + '\n' + header[
        'date'] + '\n' + url_path
    signature = to_sha1_base64(stringToSign, ak_secret)
    return signature


class AliyunCloudTester(CloudTester):

    def __init__(self, task: str):
        super(AliyunCloudTester, self).__init__(task)

        # Prepare request URLs
        if self.task == 'classify':
            self.host_url = CLASSIFY_URL
        elif self.task == 'gender':
            pass
            # self.host_url = GENDER_URL
        else:
            raise NotImplementedError('Only support classification and gender tasks.')

        # Prepare for "Label dictionary" style of matching
        current_dir = pathlib.Path(__file__).resolve().parent
        label_dict_path = current_dir / '../EQ_Dict/json/aliyun_dict.json'
        with label_dict_path.open(mode='r', encoding='UTF-8') as f:
            self.label_dict = json.loads(f.read())

    def build_label_dict(self, orig_preds: dict, true_labels: np.ndarray, threshold: float) -> dict:

        """
        Builds an equivalence dictionary between local and cloud.
        :param orig_preds: dict {id: prediction}, original predictions by MLaaS platforms
        :param true_labels: np.ndarray, represents local class index at corresponding position
        :param threshold: confidence threshold to build the dictionary
        :return: an equivalence dictionary: {local class: [cloud labels]}
        """

        image_ids = list(orig_preds.keys())
        label_dict = {}

        for image_id in image_ids:
            local_class_index = true_labels[image_id]
            local_class_name = image_idx2class[local_class_index]

            if local_class_name not in label_dict:
                label_dict[local_class_name] = set()

            cloud_preds = orig_preds[image_id]
            for item in cloud_preds:  # item = {'value': name, 'confidence': (0, 100)}
                if item['confidence'] >= threshold:
                    label_dict[local_class_name].add(item['value'])

        for k in label_dict.keys():
            label_dict[k] = list(label_dict[k])

        return label_dict

    def test_accuracy(self, label_dict: dict, original_preds: dict, true_labels: np.ndarray, threshold: float) -> float:

        """
        Test the accuracy of the cloud platform, given local true label and label map.
        :param label_dict: The equivalence dictionary between cloud and local.
        :param original_preds: Original predictions of cloud platform.
        :param true_labels: Local true label class index.
        :param threshold: Confidence Threshold
        :return: Predicition accuracy.
        """

        image_ids = list(original_preds.keys())
        num_images = len(image_ids)
        success = 0

        for image_id in image_ids:
            cloud_prediction = original_preds[image_id]
            local_class_index = true_labels[image_id]
            local_class_name = image_idx2class[local_class_index]
            cloud_class_names = label_dict[local_class_name]
            for item in cloud_prediction:
                if item['confidence'] >= threshold:
                    if item['value'] in cloud_class_names:
                        success += 1
                        break

        accuracy = success / num_images
        return accuracy

    def test_one_image(self, image_path):
        # check whether the image is totally made up with 0.0.0 pixels
        # if so, it means that this AE is a failure of CW2 attack,
        # and should be regarded as a failed AE, which will be remarked with 'fail' in the response.
        try:
            if mpimg.imread(image_path).max() == 0.:
                print('Bad CW2 AE detected. Recording this abnormal AE...')
                return {"BadCW2AE": "BadCW2AE"}
        except:
            print('Bad image format detected. Not an AE...')
            return {"BadCW2AE": "BadCW2AE"}

        credentials = AccessKeyCredential(AK_ID, AK_SECRET)

        if self.task == 'gender':
            _pos1 = image_path.find('AdienceGenderG') + len('AdienceGenderG')
            image_path = image_path[_pos1:].replace('\\', '/')
            image_path = 'oss://' + OSS_BUCKET_GENDER + image_path
            client = AcsClient(region_id='cn-hangzhou', credential=credentials)
            request = DetectImageFacesRequest()
            request.set_accept_format('json')
            request.set_ImageUri(image_path)
            request.set_Project(PROJECT_NAME)
        elif self.task == 'classify':
            _pos1 = image_path.find('ImageNet')
            image_path = image_path[_pos1:].replace('\\', '/')
            image_path = f'https://{OSS_BUCKET_CLASSIFY}.oss-cn-shanghai.aliyuncs.com/{image_path}'
            # replace space with '%20'...nails like this drive me crazy
            image_path = image_path.replace(' ', '%20')
            client = AcsClient(region_id='cn-shanghai', credential=credentials)
            request = TaggingImageRequest()
            request.set_accept_format('json')
            request.set_ImageURL(image_path)
        else:
            raise ValueError('Unexpected task!')

        retry_count = 0
        while retry_count < 10:
            try:
                r = client.do_action_with_exception(request)
                # print('r:', r.decode('utf-8'))
                break
            except:
                time.sleep(0.5)
                # print(f'Retrying({retry_count}/10)...',)
                retry_count += 1
        if retry_count == 10:
            print('last retry')
            try:
                r = client.do_action_with_exception(request)
                print('r:', r.decode('utf-8'))
            except:
                print(f'image path: {image_path}')
                print('Bad image. This is not an AE.')
                return {"BadCW2AE": "BadCW2AE"}
        return r

    def analyze_one_image_classification(self, image, results):
        # Parse image file name to get image id, true label and local adversarial label
        img_info = parse_ae_name(image)

        # If true label and local adversarial label are the same, it means the attack fails locally and
        # there's no point in talking about its transferability. So we can just skip it.
        if img_info['true_label'] == img_info['adv_label']:
            return None

        # If we didn't meet true label in cloud prediction, it means we succeeded in untargeted attack.
        untargeted_succ = True
        # If we didn't meet local adversarial label in cloud prediction, it means we succeeded in targeted attack.
        targeted_succ = False

        predictions = results[img_info['id']]

        try:
            if 'BadCW2AE' in predictions.keys():
                print("Empty image generated by CW2 detected. Return NONE (this is not an AE).")
                return None
        except:
            pass

        # # "tag-map" style matching
        # for item in predictions:
        #     # if item['confidence'] <= 50:
        #     #     continue
        #     if fine_grained_match(tag=item['value'], candidates=imagenet_tag_map[img_info['true_label']]):
        #         untargeted_succ = False
        #     if coarse_grained_match(tag=item['value'], candidates=imagenet_tag_map[img_info['adv_label']]):
        #         targeted_succ = True

        # "Label-dictionary" style matching
        true_label = img_info['true_label'].replace(' ', '_')
        adv_label = img_info['adv_label'].replace(' ', '_')
        predictions = json.loads(predictions.decode('utf8'))['Data']['Tags']
        for item in predictions:
            if item['Confidence'] <= 10:
                continue
            if item['Value'] in self.label_dict[true_label]:
                untargeted_succ = False
            if item['Value'] in self.label_dict[adv_label]:
                targeted_succ = True

        if not untargeted_succ and targeted_succ:
            print('untargeted fails, targeted succeed.')

        return untargeted_succ, targeted_succ

    def analyze_one_image_gender(self, image, results):
        img_info = parse_ae_name(image)
        if img_info['true_label'] == img_info['adv_label']:
            return None
        predictions = results[img_info['id']]

        true_label = img_info['true_label']

        try:
            if 'BadCW2AE' in predictions.keys():
                print("Empty image generated by CW2 detected. Return NONE (this is not an AE).")
                return None
        except:
            pass

        try:
            label = json.loads(predictions.decode('utf8'))['Faces'][0]['Gender']
        except:
            print('No face detected. Return NONE (this is not an AE).')
            return None
        label = label.lower()

        if true_label == 'female':
            is_female = True
            is_success = (label == 'male')
        else:
            is_female = False
            is_success = (label == 'female')
        return is_female, is_success


if __name__ == '__main__':
    aliyun_tester = AliyunCloudTester('classify')
    img_pth_test = 'test path here'
    print(img_pth_test)
    res = aliyun_tester.test_one_image(img_pth_test)
    print(res)
