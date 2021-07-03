# -*- coding: utf-8 -*-

# ======================================================================================================================
#
#
#
#                               5. Baseline moedling : XGBoost + SVR + Randomforest (on server)
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

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from skopt import gp_minimize
from skopt.space.space import Integer, Real



# --------------------------------------->>> [Set directory]


# ----- Set output path

filename = '005.baseline_XGBosot_SVR_randomforest'

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
feat_names = pickle.load(open('data/feat_names_1.sav', 'rb'))
train_val_list = pickle.load(open('data/train_val_list_1.sav', 'rb'))

sample_submission = pd.read_csv('data/235745_parking_data/sample_submission.csv')

# ----------------------------------------------------------------------------------------------------------------------
# 1. XGBoost fitting
# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------->>> [Hyper parameter optimization 결과]

# --------------------------------------->>> [재탐색]

def objective(params):

    # --------------------------------------->>> Unpack parameters

    learning_rate = params[0]
    reg_alpha = params[1]
    reg_lambda = params[2]

    # --------------------------------------->>> Build XGBoost

    reg_xgboost = XGBRegressor(n_estimators = 3000,
                               random_state = 0,
                               max_depth = 5,
                               learning_rate = learning_rate,
                               reg_alpha = reg_alpha,
                               reg_lambda = reg_lambda,
                               n_jobs = 2)

    # ----- 병렬처리

    ray.init(num_cpus = 4)

    @ray.remote

    def xgboost_fit_fn(train_val_tupple):

        tr_df, val_df = train_val_tupple

        X_names = [x for x in tr_df.columns if x not in ['공급유형_merge', '등록차량수']]
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


opt_result_xgboost = pickle.load(open('out/modeling/001.baseline_XGBoost/opt_result_3.sav', 'rb'))

reg_xgboost = XGBRegressor(n_estimators = 3000,
                           random_state = 0,
                           max_depth = 5,
                           learning_rate = opt_result_xgboost.x[0],
                           reg_alpha = opt_result_xgboost.x[1],
                           reg_lambda = opt_result_xgboost.x[2],
                           n_jobs = 6)


X_names = [x for x in train_val_list[0][1].columns if x not in ['공급유형_merge', '등록차량수']]
y_names = '등록차량수'

X_train = train_df[X_names].values
y_train = train_df[y_names].values

X_test = test_df[X_names].values

reg_xgboost.fit(X_train, y_train)

y_pred_xgboost = reg_xgboost.predict(X_test)


# ----------------------------------------------------------------------------------------------------------------------
# 2. SVR fitting
# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------->>> [Bayesian optimization을 위한 목적 함수 정의]

def objective(params):

    # --------------------------------------->>> Build XGBoost

    reg_svr = SVR(kernel = 'rbf',
                  C = params[0],
                  gamma = params[1])

    # ----- 병렬처리

    ray.init(num_cpus = 4)

    @ray.remote

    def xgboost_fit_fn(train_val_tupple):

        tr_df, val_df = train_val_tupple

        X_names = [x for x in tr_df.columns if x not in ['공급유형_merge', '등록차량수']]
        y_names = '등록차량수'

        X_tr, y_tr = tr_df[X_names].values, tr_df[y_names].values
        X_val, y_val = val_df[X_names].values, val_df[y_names].values

        scaler = MinMaxScaler()
        scaler.fit(X_tr)

        X_tr_scaled = scaler.transform(X_tr)
        X_val_scaled = scaler.transform(X_val)

        reg_svr.fit(X_tr_scaled, y_tr)

        y_pred = reg_svr.predict(X_val_scaled)

        return mean_absolute_error(y_val, y_pred)

    ray_list = [xgboost_fit_fn.remote(tuple_) for tuple_ in train_val_list]
    mae_list = ray.get(ray_list)

    ray.shutdown()

    return np.mean(mae_list)

opt_result_svr = pickle.load(open('out/modeling/002.baseline_SVR/opt_result_4.sav', 'rb'))

reg_svr = SVR(kernel = 'rbf',
              C = opt_result_svr.x[0],
              gamma = opt_result_svr.x[1])

X_names = [x for x in train_val_list[0][1].columns if x not in ['공급유형_merge', '등록차량수']]
y_names = '등록차량수'

X_train = train_df[X_names].values
y_train = train_df[y_names].values

X_test = test_df[X_names].values

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

reg_svr.fit(X_train_scaled, y_train)

y_pred_svr = reg_svr.predict(X_test_scaled)

# ----------------------------------------------------------------------------------------------------------------------
# 3. Randomforest fitting
# ----------------------------------------------------------------------------------------------------------------------

ret_rf = RandomForestRegressor(n_estimators = 100000,
                               random_state = 0,
                               n_jobs = 4)

X_names = [x for x in train_val_list[0][1].columns if x not in ['공급유형_merge', '등록차량수']]
y_names = '등록차량수'

X_train = train_df[X_names].values
y_train = train_df[y_names].values

X_test = test_df[X_names].values

ret_rf.fit(X_train, y_train)

y_pred_rf = ret_rf.predict(X_test)

# ----------------------------------------------------------------------------------------------------------------------
# 4. Ensemble
# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------->>> [Ensemble prediction]

y_pred = (y_pred_xgboost + y_pred_svr + y_pred_rf) / 3


# --------------------------------------->>> [Prediction performance check]

fig, ax = plt.subplots(1, 1, figsize = (10, 5))

x_ = np.arange(1, len(y_pred_rf)+1)

ax.plot(x_, y_pred_xgboost,
        color = sns.color_palette()[0],
        ls = '--',
        label = 'XGBoost')

ax.plot(x_, y_pred_svr,
        color = sns.color_palette()[1],
        ls = '--',
        label = 'SVR')

ax.plot(x_, y_pred_rf,
        color = sns.color_palette()[2],
        ls = '--',
        label = 'RF')

# ax.plot(x_, y_pred,
#         color = 'black',
#         label = 'Ensemble')

plt.legend()

fig.tight_layout()

plt.close(fig)

fig.show()


pred_result = pd.DataFrame({'code' : test_df['단지코드'].values.tolist(),
                            'prediction' : y_pred})

pred_result_submit = pd.merge(sample_submission, pred_result,
                              on = 'code',
                              how = 'left')

pred_result_submit.drop(['num'], axis = 1, inplace = True)
pred_result_submit.rename({'prediction' : 'num'}, axis = 1, inplace = True)

pred_result_submit.to_csv(f'{out_path}/pred_result_submit.csv',
                          index = False)


fig, ax = plt.subplots(1, 1, figsize = (5, 5))

ax.scatter(y_pred_svr, y_pred_rf,
           color = sns.color_palette()[0],
           s = 5,
           alpha = 0.5)

plt.close(fig)

fig.show()


























