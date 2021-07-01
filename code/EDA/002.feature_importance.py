# -*- coding: utf-8 -*-

# ======================================================================================================================
#
#
#
#                                           2. Feature importance
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

exec(open('code/functions/001.feature_importance.py').read())


# --------------------------------------->>> [Set directory]


# ----- Set output path

filename = '002.feature_importance'

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
# 1. Part 1
# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------->>> [전체 변수 대상]

feat_names = ['총세대수',
              '면적_35_세대수',
              '면적_45_세대수',
              '면적_50_세대수',
              '면적_55_세대수',
              '면적_70_세대수',
              '공가수',
              'subway',
              'bus',
              '단지내주차면수',
              '세대수합',
              '임대보증금_mean',
              '임대보증금_min',
              '임대보증금_max',
              '임대료_mean',
              '임대료_min',
              '임대료_max',
              'mean_enc_region',
              'mean_enc_supply',
              'mean_enc_cond',
              'size_3',
              'size_9',
              '임대세대외',
              '실세대수',
              '임대세대비율',
              'subway_ratio',
              'bus_ratio',
              '단위주차면수']

feat_names_add = [x for x in train_df.columns if '대_' in x]

feat_names += feat_names_add


X_train = train_df[feat_names].values
y_train = train_df['등록차량수'].values


imp_arr = rf_imp_fn(X = X_train,
                    y = y_train,
                    n_estimators = 10000)

imp_df = pd.DataFrame(imp_arr,
                      columns = feat_names)

fig, ax = plt.subplots(1, 1, figsize = (40, 10))

sns.boxplot(data = imp_df, ax = ax)

plt.close(fig)

fig.show() # --->>> 단지내주차면수 중요

# --------------------------------------->>> [단지내주차면수 제외]

feat_names.remove('단지내주차면수')

X_train = train_df[feat_names].values
y_train = train_df['등록차량수'].values


imp_arr = rf_imp_fn(X = X_train,
                    y = y_train,
                    n_estimators = 10000)

imp_df = pd.DataFrame(imp_arr,
                      columns = feat_names)

fig, ax = plt.subplots(1, 1, figsize = (40, 10))

sns.boxplot(data = imp_df, ax = ax)

plt.close(fig)

fig.show() # --->>> 총세대수, 세대수합, 실세대수 중요

# --------------------------------------->>> [총세대수, 세대수합, 실세대수 제외]

feat_names.remove('총세대수')
feat_names.remove('세대수합')
feat_names.remove('실세대수')

X_train = train_df[feat_names].values
y_train = train_df['등록차량수'].values


imp_arr = rf_imp_fn(X = X_train,
                    y = y_train,
                    n_estimators = 10000)

imp_df = pd.DataFrame(imp_arr,
                      columns = feat_names)

fig, ax = plt.subplots(1, 1, figsize = (40, 10))

sns.boxplot(data = imp_df, ax = ax)

fig.tight_layout()

plt.close(fig)

fig.show()

# --------------------------------------->>> [중요도 분포 낮은 변수들 제외]

feat_names = ['총세대수',
              '면적_35_세대수',
              '면적_45_세대수',
              '면적_50_세대수',
              '면적_55_세대수',
              '면적_70_세대수',
              '공가수',
              'subway',
              'bus',
              '단지내주차면수',
              '세대수합',
              '임대보증금_mean',
              '임대보증금_min',
              '임대보증금_max',
              '임대료_mean',
              '임대료_min',
              '임대료_max',
              'mean_enc_region',
              'mean_enc_supply',
              'mean_enc_cond',
              'size_3',
              'size_9',
              '임대세대외',
              '실세대수',
              '임대세대비율',
              'subway_ratio',
              'bus_ratio',
              '단위주차면수']


delete_feat_names = ['면적_55_세대수',
                     'subway',
                     '임대세대외',
                     '임대세대비율',
                     'subway_ratio',
                     'mean_enc_region',
                     'mean_enc_cond']

for feat in delete_feat_names:

    feat_names.remove(feat)

# ----- 우선 대상 변수 저장

pickle.dump(feat_names,
            open('data/feat_names_1.sav', 'wb'))

















