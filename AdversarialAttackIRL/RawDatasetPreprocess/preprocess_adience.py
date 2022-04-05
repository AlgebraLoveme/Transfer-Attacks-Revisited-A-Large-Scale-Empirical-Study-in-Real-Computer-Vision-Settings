from PIL import Image, ImageStat, ImageOps
import os
import random
import pandas as pd
from PIL import Image, ImageStat, ImageOps
from shutil import copyfile


def generate_des_img_path(dataset_path, str_gender, img_file_name):
    img_path_des_root = f"{dataset_path}AdienceGenderG/{dataset_divide_type}/{str_gender}/"
    if not os.path.isdir(img_path_des_root):
        os.makedirs(img_path_des_root)
    img_path_des = img_path_des_root + img_file_name
    return img_path_des


def transform(img, width, height):
    """
    crop img into shape of (width*height)
    missing part will be seen as blank
    crop by the center

    parameters:
    img: a PIL image
    width: width of the cropped shape
    height: width of the cropped shape
    """
    x, y = img.size  # original width & height
    dx, dy = width - x, height - y
    padding = (dx // 2, dy // 2, dx - dx // 2, dy - dy // 2)
    new_img = ImageOps.expand(img, padding)
    return new_img


def copy_one_img(img_id_in_df, dataset_path, dataset_divide_type, df_img_paths):
    if dataset_divide_type not in ['validation', 'test', 'train']:
        raise ValueError('Wrong dataset divide type.')
    src_path = df_img_paths.iloc[img_id_in_df]['path']
    str_gender = df_img_paths.iloc[img_id_in_df]['gender']
    dict_gender2Gender = {'f': 'female', 'm': 'male'}
    str_gender = dict_gender2Gender[str_gender]
    _pos = src_path.find('coarse')
    img_file_name = src_path[_pos:]

    img_path_des = generate_des_img_path(dataset_path, str_gender, img_file_name)
    if os.path.isfile(src_path):
        copyfile(src_path, img_path_des)
        # transform the img after copying, size = 384
        transform_one_img(img_path_des, img_size=384)
    else:
        print('src_path not exist!')
        print(src_path)


def transform_one_img(img_path, img_size):
    img = Image.open(img_path)
    img = transform(img, img_size, img_size)
    img.save(img_path)


dataset_path = 'Your path here, e.g., ./Datasets/AdienceGenderAge/AdienceGenderClassification/'
NUM_FOLD = 5

# get df of img info (face_id2gender)
data_frames = []
for table_id in range(NUM_FOLD):
    df_img_info = pd.read_table(dataset_path + f'fold_{table_id}_data.txt', sep='\t')
    data_frames.append(df_img_info)
df_img_info_all = pd.concat(data_frames)
print(df_img_info_all)


# generate df of img paths and gender


def get_all_img_file_path_list(path):
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            # only store img files
            if '.jpg' in filename:
                Filelist.append(os.path.join(home, filename))
    return Filelist


img_file_path_list = get_all_img_file_path_list(dataset_path + 'faces/')
print(f'Number of imgs: {len(img_file_path_list)}')

# read and store gender attribute of all imgs
img_gender_list = []
for id_t in range(len(img_file_path_list)):
    # read face_id of every img
    file_path = img_file_path_list[id_t]
    start_pos_face_id_str = file_path.find('face.') + len('face.')
    str_face_id = file_path[start_pos_face_id_str:]
    end_pos = str_face_id.find('.')
    str_face_id = str_face_id[:end_pos]
    # read the gender attribute of this img
    str_gender = df_img_info_all[df_img_info_all['face_id'] == int(str_face_id)].iloc[0]['gender']
    img_gender_list.append(str_gender)
# generate df of img paths and gender
_data = {'path': pd.Series(img_file_path_list),
         'gender': pd.Series(img_gender_list)}
df_img_path_gender = pd.DataFrame(_data)

# df split into male and female
female_df = df_img_path_gender[df_img_path_gender['gender'] == 'f']
male_df = df_img_path_gender[df_img_path_gender['gender'] == 'm']
dict_gender2df = {'male': male_df, 'female': female_df}

print(f'Number of female pics:{len(female_df)}')
print(f'Number of male pics:{len(male_df)}')
minor_df_length = min([len(female_df), len(male_df)])
NUM_TOTAL_IMG_PER_GENDER = 5000
print(f'Number of selected pics per gender:{NUM_TOTAL_IMG_PER_GENDER}')
random_select_img_ids_one_gender = random.sample(range(minor_df_length), NUM_TOTAL_IMG_PER_GENDER)

TEST_RATIO = 0.1
VAL_RATIO = 0.1
dict_divide2ids = {
    'validation': random_select_img_ids_one_gender[:int(VAL_RATIO * NUM_TOTAL_IMG_PER_GENDER)],
    'test': random_select_img_ids_one_gender[-int(TEST_RATIO * NUM_TOTAL_IMG_PER_GENDER):],
    'train': random_select_img_ids_one_gender[int(VAL_RATIO * NUM_TOTAL_IMG_PER_GENDER):
                                              -int(TEST_RATIO * NUM_TOTAL_IMG_PER_GENDER):]
}

for gender_type in ['male', 'female']:
    df_img_paths = dict_gender2df[gender_type]
    for dataset_divide_type in ['validation', 'test', 'train']:
        random_select_img_ids_divide = dict_divide2ids[dataset_divide_type]
        for img_id_in_df in random_select_img_ids_divide:
            copy_one_img(img_id_in_df, dataset_path, dataset_divide_type, df_img_paths)
            # break
