# -*- coding: utf-8 -*-

# ======================================================================================================================
#
#
#
#                                  4. Baseline modeling : Random forest (on server)
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
import ray

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from skopt import gp_minimize
from skopt.space.space import Integer, Real



# --------------------------------------->>> [Set directory]


# ----- Set output path

filename = '007.baseline_2_randomforest'

if filename not in os.listdir('out/modeling'):

    os.mkdir('{}/out/modeling/{}'.format(os.getcwd(), filename))

out_path = 'out/modeling/{}'.format(filename)

# --------------------------------------->>> [Set options]

# ----- Pandas max column showing options

pd.set_option('max.column', None)

# ----- Matplotlib axis offset 설정

mpl.rcParams['axes.formatter.useoffset'] = False

# ----- 한글 폰트 설정

# plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False



# --------------------------------------->>> [Data loading]

train_df = pickle.load(open('data/train_df.sav', 'rb'))
test_df = pickle.load(open('data/test_df.sav', 'rb'))
feat_names = pickle.load(open('data/feat_names_7.sav', 'rb'))
train_val_list = pickle.load(open('data/train_val_list_7.sav', 'rb'))

sample_submission = pd.read_csv('data/235745_parking_data/sample_submission.csv')

# ----------------------------------------------------------------------------------------------------------------------
# 1. Cross validation using multiprocessing
# ----------------------------------------------------------------------------------------------------------------------


reg_rf = RandomForestRegressor(n_estimators = 5000,
                               random_state = 0,
                               n_jobs = 2,
                               criterion = 'mae')


# --------------------------------------->>> [병렬처리]

ray.init(num_cpus = 4)

@ray.remote
def xgboost_fit_fn(train_val_tupple):
    tr_df, val_df = train_val_tupple

    X_names = [x for x in tr_df.columns if x not in ['공급유형_merge', '등록차량수', '지역']]
    y_names = '등록차량수'

    X_tr, y_tr = tr_df[X_names].values, tr_df[y_names].values
    X_val, y_val = val_df[X_names].values, val_df[y_names].values

    reg_rf.fit(X_tr, y_tr)

    y_pred = reg_rf.predict(X_val)

    return mean_absolute_error(y_val, y_pred)


ray_list = [xgboost_fit_fn.remote(tuple_) for tuple_ in train_val_list]
mae_list = ray.get(ray_list)

ray.shutdown()

print(np.mean(mae_list)) # 126.37

# --------------------------------------->>> [비교 : MSE]

reg_rf = RandomForestRegressor(n_estimators = 5000,
                               random_state = 0,
                               n_jobs = 2)


# --------------------------------------->>> [병렬처리]

ray.init(num_cpus = 4)

@ray.remote
def xgboost_fit_fn(train_val_tupple):
    tr_df, val_df = train_val_tupple

    X_names = [x for x in tr_df.columns if x not in ['공급유형_merge', '등록차량수', '지역']]
    y_names = '등록차량수'

    X_tr, y_tr = tr_df[X_names].values, tr_df[y_names].values
    X_val, y_val = val_df[X_names].values, val_df[y_names].values

    reg_rf.fit(X_tr, y_tr)

    y_pred = reg_rf.predict(X_val)

    return mean_absolute_error(y_val, y_pred)


ray_list = [xgboost_fit_fn.remote(tuple_) for tuple_ in train_val_list]
mae_list = ray.get(ray_list)

ray.shutdown()

print(np.mean(mae_list)) # 126.87






























