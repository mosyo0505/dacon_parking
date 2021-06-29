# -*- coding: utf-8 -*-

# ======================================================================================================================
#
#
#
#                                            1. Preprocessing 1
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

from sklearn.model_selection import KFold


# --------------------------------------->>> [Set directory]


# ----- Set output path

out_path = 'data'

# --------------------------------------->>> [Set options]

# ----- Pandas max column showing options

pd.set_option('max.column', None)

# ----- Matplotlib axis offset 설정

mpl.rcParams['axes.formatter.useoffset'] = False

# ----- 한글 폰트 설정

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False



# --------------------------------------->>> [Data loading]

raw_train_df = pd.read_csv('data/235745_parking_data/train.csv')
raw_test_df = pd.read_csv('data/235745_parking_data/test.csv')
raw_age_gender_info = pd.read_csv('data/235745_parking_data/age_gender_info.csv')
sample_submission = pd.read_csv('data/235745_parking_data/sample_submission.csv')

raw_train_df.rename({'도보 10분거리 내 지하철역 수(환승노선 수 반영)' : 'subway',
                     '도보 10분거리 내 버스정류장 수' : 'bus'}, axis = 1, inplace=  True)

raw_test_df.rename({'도보 10분거리 내 지하철역 수(환승노선 수 반영)' : 'subway',
                     '도보 10분거리 내 버스정류장 수' : 'bus'}, axis = 1, inplace = True)



# ----------------------------------------------------------------------------------------------------------------------
# 1. 단지별 요약 통계량 만들기
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
    [f'면적_{x}_세대수' for x in np.arange(15, 105, 5)] +\
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

train_df.reset_index(drop = True, inplace = True)

# --------------------------------------->>> [요약 통계량 만들기, test set]


code_list_uq = raw_test_df['단지코드'].unique().tolist()

need_columns = ['단지코드', '총세대수', '임대건물구분', '지역', '공급유형'] +\
    [f'면적_{x}_세대수' for x in np.arange(15, 105, 5)] +\
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

test_df.reset_index(drop = True, inplace = True)


# ----------------------------------------------------------------------------------------------------------------------
# 2. Categorical 변수 전처리
# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------->>> [지역 mean encoding]

# ----- Train set

mean_enc_list = []

for tr_idx, ts_idx in KFold(n_splits = 5, shuffle = False).split(train_df):

    tr_df = train_df.iloc[tr_idx, :].copy()
    ts_df = train_df.iloc[ts_idx, :].copy()

    ts_df['mean_enc_region'] = ts_df['지역'].map(tr_df.groupby('지역')['등록차량수'].mean())

    mean_enc_list.append(ts_df)

train_df = pd.concat(mean_enc_list)

global_mean = train_df['등록차량수'].mean()
train_df['mean_enc_region'].fillna(global_mean, inplace = True)

# ----- Test set

test_df['mean_enc_region'] = test_df['지역'].map(train_df.groupby('지역')['등록차량수'].mean())

pickle.dump(train_df, open('data/train_df.sav', 'wb'))
pickle.dump(test_df, open('data/test_df.sav', 'wb'))

























