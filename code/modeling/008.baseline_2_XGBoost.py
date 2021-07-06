# -*- coding: utf-8 -*-

# ======================================================================================================================
#
#
#
#                                       8. Baseline modeling : XGBoost (on server)
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

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from skopt import gp_minimize
from skopt.space.space import Integer, Real



# --------------------------------------->>> [Set directory]


# ----- Set output path

filename = '008.baseline_2_XGBoost'

if filename not in os.listdir('out/modeling'):

    os.mkdir('{}/out/modeling/{}'.format(os.getcwd(), filename))

out_path = 'out/modeling/{}'.format(filename)

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
feat_names = pickle.load(open('data/feat_names_7.sav', 'rb'))
train_val_list = pickle.load(open('data/train_val_list_7.sav', 'rb'))

sample_submission = pd.read_csv('data/235745_parking_data/sample_submission.csv')



# ----------------------------------------------------------------------------------------------------------------------
# 1. Cross validation using multiprocessing
# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------->>> [Bayesian optimization을 위한 목적 함수 정의]

def objective(params):

    # --------------------------------------->>> Unpack parameters

    n_estimators = params[0]
    max_depth = params[1]
    learning_rate = params[2]
    reg_alpha = params[3]
    reg_lambda = params[4]

    # --------------------------------------->>> Build XGBoost

    reg_xgboost = XGBRegressor(n_estimators = n_estimators,
                               random_state = 0,
                               max_depth = max_depth,
                               learning_rate = learning_rate,
                               reg_alpha = reg_alpha,
                               reg_lambda = reg_lambda,
                               n_jobs = 2)

    # ----- 병렬처리

    ray.init(num_cpus = 4)

    @ray.remote

    def xgboost_fit_fn(train_val_tupple):

        tr_df, val_df = train_val_tupple

        X_names = [x for x in tr_df.columns if x not in ['공급유형_merge', '등록차량수', '지역']]
        y_names = '등록차량수'

        X_tr, y_tr = tr_df[X_names].values, tr_df[y_names].values
        X_val, y_val = val_df[X_names].values, val_df[y_names].values

        reg_xgboost.fit(X_tr, y_tr)

        y_pred = reg_xgboost.predict(X_val)

        return mean_absolute_error(y_val, y_pred)

    ray_list = [xgboost_fit_fn.remote(tuple_) for tuple_ in train_val_list]
    mae_list = ray.get(ray_list)

    ray.shutdown()

    return np.mean(mae_list)

# --------------------------------------->>> [Hyper parameter space 범위 정하기]

space = [

    Integer(500, 5000),
    Integer(3, 100),
    Real(0.001, 0.1),
    Real(0.01, 10),
    Real(0.01, 10),

]

# --------------------------------------->>> [Optimization 실행]

opt_result  = gp_minimize(objective,
                          space,
                          n_calls = 200,
                          acq_func = 'EI',
                          random_state = 0,
                          verbose = True,
                          n_jobs = 6)


pickle.dump(opt_result,
            open(f'{out_path}/opt_result_1.sav', 'wb'))


# --------------------------------------->>> [Hyper parameter space 확인]

# opt_result = pickle.load(open(f'{out_path}/opt_result_1.sav', 'rb'))

search_result_df = pd.DataFrame(opt_result.x_iters,
                                columns = ['n_estimators',
                                           'max_depth',
                                           'learning_rate',
                                           'reg_alpha',
                                           'reg_lambda'])

search_result_df['MAE'] = opt_result.func_vals


fig, ax = plt.subplots(1, 1, figsize = (5, 5))

ax.scatter(search_result_df['reg_lambda'].values,
           search_result_df['MAE'].values,
           color = sns.color_palette()[0],
           s = 5,
           alpha = 0.5)

ax.set_ylim(120, 140)

plt.close(fig)

fig.show()

# --------------------------------------->>> [재탐색]

# --------------------------------------->>> [Bayesian optimization을 위한 목적 함수 정의]

def objective(params):

    # --------------------------------------->>> Unpack parameters

    n_estimators = params[0]
    learning_rate = params[1]
    reg_alpha = params[2]
    reg_lambda = params[3]

    # --------------------------------------->>> Build XGBoost

    reg_xgboost = XGBRegressor(n_estimators = n_estimators,
                               random_state = 0,
                               max_depth = 3,
                               learning_rate = learning_rate,
                               reg_alpha = reg_alpha,
                               reg_lambda = reg_lambda,
                               n_jobs = 2)

    # ----- 병렬처리

    ray.init(num_cpus = 4)

    @ray.remote

    def xgboost_fit_fn(train_val_tupple):

        tr_df, val_df = train_val_tupple

        X_names = [x for x in tr_df.columns if x not in ['공급유형_merge', '등록차량수', '지역']]
        y_names = '등록차량수'

        X_tr, y_tr = tr_df[X_names].values, tr_df[y_names].values
        X_val, y_val = val_df[X_names].values, val_df[y_names].values

        reg_xgboost.fit(X_tr, y_tr)

        y_pred = reg_xgboost.predict(X_val)

        return mean_absolute_error(y_val, y_pred)

    ray_list = [xgboost_fit_fn.remote(tuple_) for tuple_ in train_val_list]
    mae_list = ray.get(ray_list)

    ray.shutdown()

    return np.mean(mae_list)

# --------------------------------------->>> [Hyper parameter space 범위 정하기]

space = [

    Integer(500, 5000),
    Real(0.001, 0.1),
    Real(0.01, 10),
    Real(0.01, 10),

]

# --------------------------------------->>> [Optimization 실행]

opt_result  = gp_minimize(objective,
                          space,
                          n_calls = 100,
                          acq_func = 'EI',
                          random_state = 0,
                          verbose = True,
                          n_jobs = 6)

pickle.dump(opt_result,
            open(f'{out_path}/opt_result_2.sav', 'wb'))

search_result_df = pd.DataFrame(opt_result.x_iters,
                                columns = ['n_estimators',
                                           'learning_rate',
                                           'reg_alpha',
                                           'reg_lambda'])

search_result_df['MAE'] = opt_result.func_vals


fig, ax = plt.subplots(1, 1, figsize = (5, 5))

ax.scatter(search_result_df['reg_lambda'].values,
           search_result_df['MAE'].values,
           color = sns.color_palette()[0],
           s = 5,
           alpha = 0.5)

ax.set_ylim(120, 140)

plt.close(fig)

fig.show()

# --------------------------------------->>> [재탐색]

# --------------------------------------->>> [Hyper parameter space 범위 정하기]

space = [

    Integer(500, 5000),
    Real(0.001, 0.1),
    Real(0.01, 10),
    Real(10, 20),

]

# --------------------------------------->>> [Optimization 실행]

opt_result  = gp_minimize(objective,
                          space,
                          n_calls = 100,
                          acq_func = 'EI',
                          random_state = 0,
                          verbose = True,
                          n_jobs = 6)

search_result_df = pd.DataFrame(opt_result.x_iters,
                                columns = ['n_estimators',
                                           'learning_rate',
                                           'reg_alpha',
                                           'reg_lambda'])

search_result_df['MAE'] = opt_result.func_vals


fig, ax = plt.subplots(1, 1, figsize = (5, 5))

ax.scatter(search_result_df['reg_lambda'].values,
           search_result_df['MAE'].values,
           color = sns.color_palette()[0],
           s = 5,
           alpha = 0.5)

ax.set_ylim(120, 140)

plt.close(fig)

fig.show()


























