# -*- coding: utf-8 -*-

# ======================================================================================================================
#
#
#
#                                       6. Baseline modeling 2 : SVR (on server)
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

from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from skopt import gp_minimize
from skopt.space.space import Integer, Real



# --------------------------------------->>> [Set directory]


# ----- Set output path

filename = '006.baseline_2_SVR'

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
feat_names = pickle.load(open('data/feat_names_2.sav', 'rb'))
train_val_list = pickle.load(open('data/train_val_list_1.sav', 'rb'))

sample_submission = pd.read_csv('data/235745_parking_data/sample_submission.csv')

# ----------------------------------------------------------------------------------------------------------------------
# 1. Cross validation using multiprocessing
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


# --------------------------------------->>> [Hyper parameter space 범위 정하기]

space = [

    Real(10, 10000),
    Real(0.0001, 1)

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

opt_result = pickle.load(open(f'{out_path}/opt_result_1.sav', 'rb'))

search_result_df = pd.DataFrame(opt_result.x_iters,
                                columns = ['C',
                                           'gamma'])

search_result_df['MAE'] = opt_result.func_vals


fig, ax = plt.subplots(1, 1, figsize = (5, 5))

ax.scatter(search_result_df['gamma'].values,
           search_result_df['MAE'].values,
           color = sns.color_palette()[0],
           s = 5,
           alpha = 0.5)

ax.set_ylim(113, 120)

plt.close(fig)

fig.show()


# --------------------------------------->>> [재탐색]

# --------------------------------------->>> [Hyper parameter space 범위 정하기]

space = [

    Real(7500, 9500),
    Real(0.0001, 0.15)

]

# --------------------------------------->>> [Optimization 실행]

opt_result  = gp_minimize(objective,
                          space,
                          n_calls = 100,
                          acq_func = 'EI',
                          random_state = 0,
                          verbose = True,
                          n_jobs = 6)








