# -*- coding: utf-8 -*-

# ======================================================================================================================
#
#
#
#                                                     1. EDA
#
#
#
# ======================================================================================================================

# ----------------------------------------------------------------------------------------------------------------------
# 0. Set Environments
# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------->>> [Import modules]

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pickle

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import NMF

# exec(open('code/preprocessing/001.preprocessing_1.py').read())
exec(open('code/functions/001.feature_importance.py').read())


# --------------------------------------->>> [Set directory]


# ----- Set output path

filename = '001.EDA'

if filename not in os.listdir('out/EDA'):

    os.mkdir('{}/out/EDA/{}'.format(os.getcwd(), filename))

out_path = 'out/EDA/{}'.format(filename)

# --------------------------------------->>> [Set options]

# ----- Pandas max column showing options

pd.set_option('max.column', None)

# ----- Matplotlib axis offset 설정

mpl.rcParams['axes.formatter.useoffset'] = False

# ----- 한글 폰트 설정

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


# --------------------------------------->>> [Data loading]

train_df = pickle.load(open('data/train_df.sav', 'rb'))
test_df = pickle.load(open('data/test_df.sav', 'rb'))


# ----------------------------------------------------------------------------------------------------------------------
# 1. Data preprocessing
# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------->>> [이상값 처리]

# ----- "-" NAN 처리

raw_train_df = raw_train_df.replace({'-' : np.nan})
raw_train_df['임대보증금'] = raw_train_df['임대보증금'].astype(float)
raw_train_df['임대료'] = raw_train_df['임대료'].astype(float)

raw_test_df = raw_test_df.replace({'-' : np.nan})
raw_test_df['임대보증금'] = raw_test_df['임대보증금'].astype(float)
raw_test_df['임대료'] = raw_test_df['임대료'].astype(float)

# ----- 전용면적 15 미만 15로, 100초과 100으로

def tmp_fn(x):

    if x < 15:

        return 15

    if x >= 15 and x < 105:

        return x

    if x >= 105:

        return 100

raw_train_df['전용면적'] = raw_train_df['전용면적'].map(tmp_fn)
raw_test_df['전용면적'] = raw_test_df['전용면적'].map(tmp_fn)


# --------------------------------------->>> [요약 통계량 만들기, train set]


code_list_uq = raw_train_df['단지코드'].unique().tolist()

need_columns = ['단지코드', '총세대수', '임대건물구분', '지역', '공급유형'] +\
    [f'면적_{x}_세대수' for x in np.arange(5, 105, 5)] +\
    ['공가수', '자격유형', 'subway', 'bus', '단지내주차면수', '등록차량수']

base_frame = pd.DataFrame({},
                          columns = need_columns,
                          index = [0])

train_df_list = []

for code in code_list_uq:

    sub_df = raw_train_df.loc[raw_train_df['단지코드'] == code, :].copy()
    sub_df['전용면적'] = (sub_df['전용면적'] // 5) * 5

    sum_df = base_frame.copy()

    sum_df['단지코드'] = sub_df['단지코드'].unique()[0]
    sum_df['총세대수'] = sub_df['총세대수'].unique()[0]
    sum_df['임대건물구분'] = '+'.join(np.sort(sub_df['임대건물구분'].unique()).tolist())
    sum_df['지역'] = sub_df['지역'].unique()[0]
    sum_df['공급유형'] = '+'.join(np.sort(sub_df['공급유형'].unique()).tolist())
    sum_df['subway'] = sub_df['subway'].unique()[0]
    sum_df['bus'] = sub_df['bus'].unique()[0]
    sum_df['단지내주차면수'] = sub_df['단지내주차면수'].unique()[0]
    sum_df['등록차량수'] = sub_df['등록차량수'].unique()[0]
    sum_df['공가수'] = sub_df['공가수'].unique()[0]
    sum_df['자격유형'] = '+'.join(np.sort(sub_df['자격유형'].unique()).tolist())

    for ii in np.arange(15, 105, 5):

        if ii in sub_df['전용면적'].values:

            sum_df[f'면적_{ii}_세대수'] = sub_df.loc[sub_df['전용면적'] == ii, '전용면적별세대수'].sum()

        else:

            sum_df[f'면적_{ii}_세대수'] = 0

    sum_df['임대보증금_mean'] = sub_df['임대보증금'].mean()
    sum_df['임대보증금_min'] = sub_df['임대보증금'].min()
    sum_df['임대보증금_max'] = sub_df['임대보증금'].max()

    sum_df['임대료_mean'] = sub_df['임대료'].mean()
    sum_df['임대료_min'] = sub_df['임대료'].min()
    sum_df['임대료_max'] = sub_df['임대료'].max()

    train_df_list.append(sum_df)

train_df = pd.concat(train_df_list)

# --------------------------------------->>> [요약 통계량 만들기, test set]


code_list_uq = raw_test_df['단지코드'].unique().tolist()

need_columns = ['단지코드', '총세대수', '임대건물구분', '지역', '공급유형'] +\
    [f'면적_{x}_세대수' for x in np.arange(5, 105, 5)] +\
    ['공가수', '자격유형', 'subway', 'bus', '단지내주차면수']

base_frame = pd.DataFrame({},
                          columns = need_columns,
                          index = [0])

test_df_list = []

for code in code_list_uq:

    sub_df = raw_test_df.loc[raw_test_df['단지코드'] == code, :].copy()
    sub_df['전용면적'] = (sub_df['전용면적'] // 5) * 5

    sum_df = base_frame.copy()

    sum_df['단지코드'] = sub_df['단지코드'].unique()[0]
    sum_df['총세대수'] = sub_df['총세대수'].unique()[0]
    sum_df['임대건물구분'] = '+'.join(np.sort(sub_df['임대건물구분'].unique()).tolist())
    sum_df['지역'] = sub_df['지역'].unique()[0]
    sum_df['공급유형'] = '+'.join(np.sort(sub_df['공급유형'].unique()).tolist())
    sum_df['subway'] = sub_df['subway'].unique()[0]
    sum_df['bus'] = sub_df['bus'].unique()[0]
    sum_df['단지내주차면수'] = sub_df['단지내주차면수'].unique()[0]
    sum_df['공가수'] = sub_df['공가수'].unique()[0]
    sum_df['자격유형'] = '+'.join(np.sort(sub_df['자격유형'].dropna().unique()).tolist())

    for ii in np.arange(5, 105, 5):

        if ii in sub_df['전용면적'].values:

            sum_df[f'면적_{ii}_세대수'] = sub_df.loc[sub_df['전용면적'] == ii, '전용면적별세대수'].sum()

        else:

            sum_df[f'면적_{ii}_세대수'] = 0

    sum_df['임대보증금_mean'] = sub_df['임대보증금'].mean()
    sum_df['임대보증금_min'] = sub_df['임대보증금'].min()
    sum_df['임대보증금_max'] = sub_df['임대보증금'].max()

    sum_df['임대료_mean'] = sub_df['임대료'].mean()
    sum_df['임대료_min'] = sub_df['임대료'].min()
    sum_df['임대료_max'] = sub_df['임대료'].max()

    test_df_list.append(sum_df)

test_df = pd.concat(test_df_list)


# ----------------------------------------------------------------------------------------------------------------------
# 2. 상관성 보기
# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------->>> [총세대수]

fig, ax = plt.subplots(1, 1, figsize = (5, 5))

x_ = train_df['총세대수'].values
y_ = train_df['등록차량수'].values

x_apt = x_[train_df['임대건물구분'] == '아파트']
x_etc = x_[train_df['임대건물구분'] != '아파트']

y_apt = y_[train_df['임대건물구분'] == '아파트']
y_etc = y_[train_df['임대건물구분'] != '아파트']


# ax.scatter(x_, y_,
#            color = sns.color_palette()[0],
#            s = 5,
#            alpha = 0.3)

ax.scatter(x_apt, y_apt,
           color = sns.color_palette()[0],
           s = 5,
           alpha = 0.3)

ax.scatter(x_etc, y_etc,
           color = sns.color_palette()[1],
           s = 5,
           alpha = 0.3)

ax.set_xlabel('총세대수')
ax.set_ylabel('등록차량수')

fig.tight_layout()

plt.close(fig)

fig.show()

# --------------------------------------->>> [공가수]

fig, ax = plt.subplots(1, 1, figsize = (5, 5))

x_ = train_df['공가수'].values
y_ = train_df['등록차량수'].values

ax.scatter(x_, y_,
           color = sns.color_palette()[0],
           s = 5,
           alpha = 0.5)

plt.close(fig)

fig.show()

# ----------------------------------------------------------------------------------------------------------------------
# 3. Categorical 정보 확인해보기
# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------->>> [지역성]

# ----- 원본

fig, ax = plt.subplots(1, 1, figsize = (20, 5))

sns.boxplot(x = '지역', y = '등록차량수', data = train_df, ax = ax)

plt.close(fig)

fig.show()

# ----- Mean encoding 실시 후

fig, ax = plt.subplots(1, 1, figsize = (20, 5))

sns.boxplot(x = '지역', y = 'mean_enc_region', data = train_df, ax = ax)

plt.close(fig)

fig.show()

# ----- Mean encoding 실시 후 Test df

fig, ax = plt.subplots(1, 1, figsize = (20, 5))

sns.boxplot(x = '지역', y = 'mean_enc_region', data = test_df, ax = ax)

plt.close(fig)

fig.show()


# ----- 상관성

custom_pallete = sns.color_palette() + sns.color_palette('Pastel1')
# sns.palplot(custom_pallete)
# plt.show()

fig, ax = plt.subplots(1, 1, figsize = (5, 5))

for ii, name in enumerate(train_df['지역'].unique()):

    sub_df = train_df.loc[train_df['지역'] == name, :]

    x_ = sub_df['mean_enc_region'].values
    y_ = sub_df['등록차량수'].values

    ax.scatter(x_, y_,
               color=custom_pallete[ii],
               s=5,
               alpha=0.3)

plt.close(fig)

fig.show()


# --------------------------------------->>> [공급유형]

# ----- 원본

fig, ax = plt.subplots(1, 1, figsize = (20, 5))

sns.boxplot(x = '공급유형_merge', y = '등록차량수', data = train_df, ax = ax)

fig.tight_layout()

plt.close(fig)

fig.show()

# ----- Mean encoding 후

fig, ax = plt.subplots(1, 1, figsize = (20, 5))

sns.boxplot(x = '공급유형_merge', y = 'mean_enc_supply', data = train_df, ax = ax)

fig.tight_layout()

plt.close(fig)

fig.show()

# ----- Mean encoding 실시 후 Test df

fig, ax = plt.subplots(1, 1, figsize = (20, 5))

sns.boxplot(x = '공급유형_merge', y = 'mean_enc_supply', data = test_df, ax = ax)

plt.close(fig)

fig.show()

# ----------------------------------------------------------------------------------------------------------------------
# 4. 면적별 세대수 관련
# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------->>> [Heatmap 확인]

size_df.sort_values(by = '등록차량수', inplace = True)

fig, ax = plt.subplots(1, 1, figsize = (10, 10))

size_arr = size_df.iloc[:, :-1].values

ax = sns.heatmap(size_arr,
                 cmap = sns.color_palette('Greys'))

plt.close(fig)

fig.show()


scaler = MinMaxScaler()
scaler.fit(size_arr)

size_arr_scaled = scaler.transform(size_arr)

fig, ax = plt.subplots(1, 1, figsize = (10, 10))

size_arr = size_df.iloc[:, :-1].values

ax = sns.heatmap(size_arr_scaled,
                 cmap = sns.color_palette('Greys'))

plt.close(fig)

fig.show()


# --------------------------------------->>> [NMF로 latent feature 찾기]

X_train = train_df[[x for x in train_df.columns if 'size_' in x]].values
y_train = train_df['등록차량수'].values

imp_arr = rf_imp_fn(X = X_train,
                    y = y_train,
                    n_estimators = 10000)

imp_df = pd.DataFrame(imp_arr,
                      columns = [x for x in train_df.columns if 'size_' in x])


fig, ax = plt.subplots(1, 1, figsize = (40, 10))

sns.boxplot(data = imp_df, ax = ax)

plt.close(fig)

fig.show() # Size 3, Size 9이 중요해보임


fig, axes = plt.subplots(1, 2, figsize = (10, 5))

ax_0, ax_1 = axes[0], axes[1]

x_9 = train_df['size_9'].values
x_3 = train_df['size_3'].values
y_ = train_df['등록차량수'].values

ax_0.scatter(x_9, y_,
             color = sns.color_palette()[0],
             s = 5,
             alpha = 0.3)

ax_1.scatter(x_3, y_,
             color = sns.color_palette()[0],
             s = 5,
             alpha = 0.3)

plt.close(fig)

fig.show()

# --------------------------------------->>> [원본 면적에 대한 feature importance 확인]

X_train = train_df[[x for x in train_df.columns if '면적' in x]].values
y_train = train_df['등록차량수'].values

imp_arr = rf_imp_fn(X = X_train,
                    y = y_train,
                    n_estimators = 10000)

imp_df = pd.DataFrame(imp_arr,
                      columns = [x for x in train_df.columns if '면적' in x])


fig, ax = plt.subplots(1, 1, figsize = (30, 10))

sns.boxplot(data = imp_df, ax = ax)

plt.close(fig)

fig.show() # 원본 면적 중요 변수 : 35, 45, 50, 55, 70






















































