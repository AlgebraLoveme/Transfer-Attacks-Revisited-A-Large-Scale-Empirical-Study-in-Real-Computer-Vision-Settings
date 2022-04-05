import abc
import os
import sys

sys.path.append('{}/../..'.format(os.path.dirname(os.path.abspath(__file__))))
import Cloud.utils as utils
from Cloud.Tester.tester_utils import listdir_without_hidden
from tqdm import tqdm


class CloudTester(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, task):
        self.task = task

    def test_one_dir(self, data_dir) -> dict:
        """
        Send images in a directory to cloud platform for testing and return the results.
        :param data_dir: Directory containing images to be tested
        :return: dictionary of {image id: response}
        """
        results = {}

        image_list = utils.listdir_without_hidden(data_dir)
        # Remove finish file
        if 'finish' in image_list:
            image_list.pop(image_list.index('finish'))

        for image_name in tqdm(image_list):
            image_info = utils.parse_ae_name(image_name)
            image_path = os.path.join(data_dir, image_name)
            cloud_response = self.test_one_image(image_path)
            results[image_info['id']] = cloud_response

            # observe once every dir to make sure the cloud test process is going normally
            if '_id_0' in image_name:
                print(f'Response 0: {cloud_response}')

        return results

    @abc.abstractmethod
    def test_one_image(self, image_path):
        raise NotImplementedError

    def analyze(self, data_dir: str, result_path: str) -> dict:
        import pickle
        with open(result_path, 'rb') as f:
            results = pickle.load(f)
        print('data_dir: ', data_dir)
        print('results: ', results)
        all_imgs = listdir_without_hidden(data_dir)
        if 'finish' in all_imgs:
            all_imgs.pop(all_imgs.index('finish'))
        if self.task == 'classify':
            return self.analyze_classification(all_imgs, results)
        elif self.task == 'gender':
            return self.analyze_gender(all_imgs, results)

    def analyze_classification(self, original_imgs, results):
        nr_samples = 0
        nr_untargeted_succ = 0
        nr_targeted_succ = 0

        for img in original_imgs:
            res = self.analyze_one_image_classification(img, results)
            if res is None:
                continue
            untargeted_succ, targeted_succ = res
            nr_samples += 1
            nr_untargeted_succ += int(untargeted_succ)
            nr_targeted_succ += int(targeted_succ)

        return {
            'misclassification_rate': nr_untargeted_succ / nr_samples if nr_samples != 0 else 0.,
            'matching_rate': nr_targeted_succ / nr_samples if nr_samples != 0 else 0.,
            'mis_ratio': nr_untargeted_succ / 200,
            'match_ratio': nr_targeted_succ / 200,
            'total': nr_samples
        }

    @abc.abstractmethod
    def analyze_one_image_classification(self, image, results):
        raise NotImplementedError

    def analyze_gender(self, original_imgs, results):
        nr_success = 0
        nr_samples = 0
        nr_female = 0
        nr_female2male = 0
        nr_male = 0
        nr_male2female = 0

        for img in original_imgs:
            res = self.analyze_one_image_gender(img, results)
            if res is None:
                continue
            nr_samples += 1
            is_female, is_success = res
            nr_success += int(is_success)
            if is_female:
                nr_female += 1
                if is_success:
                    nr_female2male += 1
            else:
                nr_male += 1
                if is_success:
                    nr_male2female += 1

        # Prevent division by 0
        if nr_female == 0: nr_female = 1
        if nr_male == 0: nr_male = 1

        return {
            'misclassification_rate': nr_success / nr_samples if nr_samples != 0 else 0.,
            'female2male_rate': nr_female2male / nr_female if nr_female != 0 else 0.,
            'male2female_rate': nr_male2female / nr_male if nr_male != 0 else 0.,
            'mis_ratio': nr_success / 200,
            'female2male_rate_100': nr_female2male / 100,
            'male2female_rate_100': nr_male2female / 100,
            'total': nr_samples,
            'nr_female': nr_female,
            'nr_male': nr_male
        }

    @abc.abstractmethod
    def analyze_one_image_gender(self, image, results):
        raise NotImplementedError
