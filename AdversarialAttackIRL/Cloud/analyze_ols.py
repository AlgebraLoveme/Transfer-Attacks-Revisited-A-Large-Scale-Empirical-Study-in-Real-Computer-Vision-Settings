import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
import seaborn as sb
# import pingouin as pg
import random
import os
from statsmodels.formula.api import ols
from scipy import stats

plt.style.use('ggplot')

import matplotlib.pylab as pylab

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (6, 6),
          'axes.labelsize': 20,
          'axes.titlesize': 16,
          'axes.labelcolor': "#000000",
          'xtick.labelsize': 20,
          'ytick.labelsize': 16,
          'font.weight': "normal",
          'xtick.color': "#000000",
          'ytick.color': "#000000",
          'axes.labelweight': 'normal'}
pylab.rcParams.update(params)
# %matplotlib inline

data_src = './Analysis/'
save_dir = f'./Analysis/OLS/'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)


def Variable2Vitual(df, attribute, values=None):
    # convert a multi-value attribute column to multiple virtual variables
    if values is None:
        values = set(df[attribute])
    for v in values:
        df[f'is_{v}'] = (df[attribute] == v).astype(int)
    return df


def read_cloud(csv_path, cloud_name):
    # preprocess the csv data
    data = pd.read_csv(csv_path)
    data['Attack'] = data['Attack'].str.upper()
    data['Attack'][data['Attack'] == 'BL-BFGS'] = 'BLB'
    data['is_Pretrained'] = data['Pretrained'].astype(int)
    data['local_success_rate'] = data['total'] / 200
    data['cloud'] = cloud_name
    return data


def export_ols(ols_result, filename):
    with open(filename, 'w') as f:
        f.write(ols_result.summary().as_csv())


dataset = 'AdienceGenderG'
df_list = []
for cloud in ['aliyun', 'baidu', 'aws']:
    csv_path = data_src + f'{cloud}_{dataset}.csv'
    df = read_cloud(csv_path, cloud)
    df_list.append(df)
data = pd.concat(df_list, axis=0)
print(data.head())

# reformalize data
for attr in ['Architecture', 'Data_type', 'Attack', 'cloud']:
    data = Variable2Vitual(data, attr)
data = Variable2Vitual(data, 'Depth', ['18', '34', '50'])
# remove unnecessary columns
df = data.drop(
    ['Architecture', 'Data_type', 'Attack', 'Depth', 'total', 'Pretrained', 'mis_ratio', 'cloud',
     'local_success_rate'], axis=1)
res_df = df[df['is_resnet'] == 1].copy(deep=True).reset_index(drop=True)
print(df.head())

print(df.columns)


def Test_Wald_R(ols_result, R_coef: list, R_var_name: list, return_result=False):
    # Test H0: R * coef = 0.
    # e.g. For R_coef = [1,-1], coef = [is_aliyun, is_baidu], this tests H0: is_aliyun - is_baidu = 0, which is equivalent to test H0: is_aliyun = is_baidu.
    # this is used to compare the coefficients of variables.
    assert len(R_coef) == len(R_var_name), 'One to one map should be given.'
    params = ols_result.params
    idx_map = dict([(list(params.index)[i], i) for i in range(len(list(params.index)))])
    R = np.zeros(len(params))
    for r, var in zip(R_coef, R_var_name):
        R[idx_map[var]] = r
    test = ols_result.wald_test(R, scalar=False)
    print(test.summary())
    if return_result:
        return test


print("\n\nRegression A")
exp_vs = ['is_aws', 'is_baidu', 'is_aliyun']
dp_variable = 'male2female_rate'
# dp_variable = 'matching_rate'
formula = '{} ~ 0+{}'.format(dp_variable, '+'.join(exp_vs))  # exclude the intercept
# formula = Generate_Compare_Formula(formula, 'is_FGSM', 'is_DEEPFOOL')
print(formula)
model = ols(formula, data=res_df)
results = model.fit()
export_ols(results, f'{save_dir}/ols_A.csv')
print(results.summary())

print("\n\nTest platform coefficients for misclassification rate")
# print(Test_Wald_R(results, [1, -1], ['is_aws', 'is_aliyun']))
# print(Test_Wald_R(results, [1, -1], ['is_aliyun', 'is_baidu']))

print("\n\nRegression B")
exp_vs = ['is_aws', 'is_baidu', 'is_aliyun', 'is_Pretrained', 'is_adversarial', 'is_augmented']
dp_variable = 'male2female_rate'
# dp_variable = 'matching_rate'
formula = '{} ~ 0+{}'.format(dp_variable, '+'.join(exp_vs))  # exclude the intercept
# formula = Generate_Compare_Formula(formula, 'is_FGSM', 'is_DEEPFOOL')
print(formula)
model = ols(formula, data=res_df)
results = model.fit()
export_ols(results, f'{save_dir}/ols_B.csv')
print(results.summary())

print("\n\nRegression C")
exp_vs = ['is_aws', 'is_baidu', 'is_aliyun', 'is_Pretrained', 'is_adversarial', 'is_augmented', 'is_PGD', 'is_FGSM',
          'is_BLB', 'is_CW2', 'is_DEEPFOOL', 'is_STEP_LLC', 'is_RFGSM', 'is_LLC']
dp_variable = 'male2female_rate'
# dp_variable = 'matching_rate'
formula = '{} ~ 0+{}'.format(dp_variable, '+'.join(exp_vs))  # exclude the intercept
# formula = Generate_Compare_Formula(formula, 'is_FGSM', 'is_DEEPFOOL')
print(formula)
model = ols(formula, data=res_df)
results = model.fit()
export_ols(results, f'{save_dir}/ols_C.csv')
print(results.summary())

print("\n\nTest adversarial algorithms")
# print(Test_Wald_R(results, [1, -1], ['is_DEEPFOOL', 'is_BLB']))
# print(Test_Wald_R(results, [1, -1], ['is_BLB', 'is_CW2']))
# # UAP here because the attacks above have negative coefficient and attacks below have positive coefficient. UAP is
# # the baseline and thus is viewed as zero coefficient.
# print(Test_Wald_R(results, [1, -1], ['is_LLC', 'is_RFGSM']))
# print(Test_Wald_R(results, [1, -1], ['is_RFGSM', 'is_PGD']))
# print(Test_Wald_R(results, [1, -1], ['is_PGD', 'is_STEP_LLC']))
# print(Test_Wald_R(results, [1, -1], ['is_STEP_LLC', 'is_FGSM']))

print("\n\nRegression D")
exp_vs = ['is_aws', 'is_baidu', 'is_aliyun', 'is_Pretrained', 'is_adversarial', 'is_augmented', 'is_PGD', 'is_FGSM',
          'is_BLB', 'is_CW2', 'is_DEEPFOOL', 'is_STEP_LLC', 'is_RFGSM', 'is_LLC', 'is_34', 'is_50']
dp_variable = 'male2female_rate'
# dp_variable = 'matching_rate'
formula = '{} ~ 0+{}'.format(dp_variable, '+'.join(exp_vs))  # exclude the intercept
# formula = Generate_Compare_Formula(formula, 'is_FGSM', 'is_DEEPFOOL')
print(formula)
model = ols(formula, data=res_df)
results = model.fit()
export_ols(results, f'{save_dir}/ols_D.csv')
print(results.summary())

print("\n\nRegression E")
exp_vs = ['is_aws', 'is_baidu', 'is_aliyun', 'is_Pretrained', 'is_adversarial', 'is_augmented', 'is_PGD',
          'is_FGSM', 'is_BLB', 'is_CW2', 'is_DEEPFOOL', 'is_STEP_LLC', 'is_RFGSM', 'is_LLC', 'is_34', 'is_50']
dp_variable = 'male2female_rate'
# dp_variable = 'matching_rate'
formula = '{} ~ 0+{}'.format(dp_variable, '+'.join(exp_vs))  # exclude the intercept
# formula = Generate_Compare_Formula(formula, 'is_FGSM', 'is_DEEPFOOL')
# add cross terms to formula
formula += '+' + '+'.join([f'I(is_Pretrained*{attr})' for attr in ['is_adversarial', 'is_augmented', ]])
formula += '+' + '+'.join(
    [f'I({attr1}*{attr2})' for attr1 in ['is_Pretrained', 'is_adversarial', 'is_augmented', ] for attr2 in
     ['is_34', 'is_50']])
# formula += '+' + '+'.join([f'I(is_Pretrained*{attr})' for attr in ['is_adversarial', 'is_PGD', 'is_FGSM', 'is_BLB', 'is_CW2', 'is_DEEPFOOL', 'is_STEP_LLC', 'is_RFGSM', 'is_LLC',]])
print(formula)
model = ols(formula, data=res_df)
results = model.fit()
export_ols(results, f'{save_dir}/ols_E.csv')
print(results.summary())

print("\n\nRegression F")
exp_vs = ['is_aws', 'is_baidu', 'is_aliyun']
dp_variable = 'female2male_rate'
# dp_variable = 'matching_rate'
formula = '{} ~ 0+{}'.format(dp_variable, '+'.join(exp_vs)) # exclude the intercept
# formula = Generate_Compare_Formula(formula, 'is_FGSM', 'is_DEEPFOOL')
# add cross terms to formulaprint(formula)
model = ols(formula, data=res_df)
results = model.fit()
export_ols(results, f'{save_dir}/ols_F.csv')
print(results.summary())

print("\n\nRegression G")
exp_vs = ['is_aws', 'is_baidu', 'is_aliyun', 'is_Pretrained', 'is_adversarial', 'is_augmented']
dp_variable = 'female2male_rate'
# dp_variable = 'matching_rate'
formula = '{} ~ 0+{}'.format(dp_variable, '+'.join(exp_vs)) # exclude the intercept
# formula = Generate_Compare_Formula(formula, 'is_FGSM', 'is_DEEPFOOL')
print(formula)
model = ols(formula, data=res_df)
results = model.fit()
export_ols(results, f'{save_dir}/ols_G.csv')
print(results.summary())

print("\n\nRegression H")
exp_vs = ['is_aws', 'is_baidu', 'is_aliyun', 'is_Pretrained', 'is_adversarial', 'is_augmented', 'is_PGD', 'is_FGSM', 'is_BLB', 'is_CW2', 'is_DEEPFOOL', 'is_STEP_LLC', 'is_RFGSM', 'is_LLC']
# dp_variable = 'misclassification_rate'
dp_variable = 'female2male_rate'
formula = '{} ~ 0+{}'.format(dp_variable, '+'.join(exp_vs)) # exclude the intercept
print(formula)
model = ols(formula, data=res_df)
results = model.fit()
export_ols(results, f'{save_dir}/ols_H.csv')
print(results.summary())

print("\n\nRegression I")
exp_vs = ['is_aws', 'is_baidu', 'is_aliyun', 'is_Pretrained', 'is_adversarial', 'is_augmented', 'is_PGD', 'is_FGSM', 'is_BLB', 'is_CW2', 'is_DEEPFOOL', 'is_STEP_LLC', 'is_RFGSM', 'is_LLC', 'is_34', 'is_50']
# dp_variable = 'misclassification_rate'
dp_variable = 'female2male_rate'
formula = '{} ~ 0+{}'.format(dp_variable, '+'.join(exp_vs)) # exclude the intercept
# formula = Generate_Compare_Formula(formula, 'is_FGSM', 'is_DEEPFOOL')
print(formula)
model = ols(formula, data=res_df)
results = model.fit()
export_ols(results, f'{save_dir}/ols_I.csv')
print(results.summary())

print("\n\nRegression J")
exp_vs = ['is_aws', 'is_baidu', 'is_aliyun', 'is_Pretrained', 'is_adversarial', 'is_augmented', 'is_PGD', 'is_FGSM',
          'is_BLB', 'is_CW2', 'is_DEEPFOOL', 'is_STEP_LLC', 'is_RFGSM', 'is_LLC', 'is_34', 'is_50']
# dp_variable = 'misclassification_rate'
dp_variable = 'female2male_rate'
formula = '{} ~ 0+{}'.format(dp_variable, '+'.join(exp_vs)) # exclude the intercept
# formula = Generate_Compare_Formula(formula, 'is_FGSM', 'is_DEEPFOOL')
# add cross terms to formula
formula += '+' + '+'.join([f'I(is_Pretrained*{attr})' for attr in ['is_adversarial', 'is_augmented',]])
formula += '+' + '+'.join([f'I({attr1}*{attr2})' for attr1 in ['is_Pretrained', 'is_adversarial', 'is_augmented', ] for attr2 in ['is_34', 'is_50']])
# formula += '+' + '+'.join([f'I(is_Pretrained*{attr})' for attr in ['is_adversarial', 'is_PGD', 'is_FGSM', 'is_BLB', 'is_CW2', 'is_DEEPFOOL', 'is_STEP_LLC', 'is_RFGSM', 'is_LLC',]])
print(formula)
model = ols(formula, data=res_df)
results = model.fit()
export_ols(results, f'{save_dir}/ols_J.csv')
print(results.summary())
