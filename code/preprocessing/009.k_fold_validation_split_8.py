# -*- coding: utf-8 -*-

# ======================================================================================================================
#
#
#
#                                          9. K Fold validation split 8
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

filename = '007.k_fold_validation_split_6'

if filename not in os.listdir('out/preprocessing'):

    os.mkdir('{}/out/preprocessing/{}'.format(os.getcwd(), filename))

out_path = 'out/preprocessing/{}'.format(filename)

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
feat_names = pickle.load(open('data/feat_names_8.sav', 'rb'))

# feat_names_for_val = list((set(feat_names) - set(['mean_enc_supply', 'mean_enc_region'])))\
#     + ['공급유형_merge', '지역']

data_mart = train_df[feat_names + ['등록차량수']]

# ----------------------------------------------------------------------------------------------------------------------
# 1. Cross validation set 만들기
# ----------------------------------------------------------------------------------------------------------------------

train_val_list = []

# --------------------------------------->>> [random_state = 0]

kfold = KFold(n_splits = 5, shuffle = True, random_state = 0)

for tr_idx, ts_idx in kfold.split(data_mart):

    tr_df = data_mart.iloc[tr_idx, :].copy()
    ts_df = data_mart.iloc[ts_idx, :].copy()

    tr_df.reset_index(drop=True, inplace=True)
    ts_df.reset_index(drop=True, inplace=True)

    train_val_list.append((tr_df, ts_df))

# --------------------------------------->>> [random_state = 42]

kfold = KFold(n_splits = 5, shuffle = True, random_state = 42)

for tr_idx, ts_idx in kfold.split(data_mart):

    tr_df = data_mart.iloc[tr_idx, :].copy()
    ts_df = data_mart.iloc[ts_idx, :].copy()

    tr_df.reset_index(drop = True, inplace = True)
    ts_df.reset_index(drop = True, inplace = True)

    train_val_list.append((tr_df, ts_df))


# ----------------------------------------------------------------------------------------------------------------------
#  결과 저장
# ----------------------------------------------------------------------------------------------------------------------

pickle.dump(train_val_list,
            open('data/train_val_list_8.sav', 'wb'))