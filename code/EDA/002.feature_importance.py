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

from sklearn.linear_model import Lasso

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

# train_df = pickle.load(open('data/train_df.sav', 'rb'))
# test_df = pickle.load(open('data/test_df.sav', 'rb'))

train_df_2 = pickle.load(open('data/train_df_2.sav', 'rb'))
test_df_2 = pickle.load(open('data/test_df_2.sav', 'rb'))

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


# ----------------------------------------------------------------------------------------------------------------------
# 2. Part 2
# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------->>> [전체 변수 대상]

feat_names_1 = ['총세대수',
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
              '임대세대외',
              '실세대수',
              '임대세대비율',
              'subway_ratio',
              'bus_ratio',
              '단위주차면수']

feat_names_2 = [x for x in train_df.columns if '면적' in x]
feat_names_3 = [x for x in train_df.columns if 'size_' in x]


feat_names_total = feat_names_1 + feat_names_2 + feat_names_3

feat_names_total.remove('총세대수')
feat_names_total.remove('세대수합')
feat_names_total.remove('실세대수')
feat_names_total.remove('단지내주차면수')

X_train = train_df[feat_names_total].values
y_train = train_df['등록차량수'].values


imp_arr = rf_imp_fn(X = X_train,
                    y = y_train,
                    n_estimators = 10000)

imp_df = pd.DataFrame(imp_arr,
                      columns = feat_names_total)

fig, ax = plt.subplots(1, 1, figsize = (100, 10))

sns.boxplot(data = imp_df, ax = ax)

fig.tight_layout()

plt.close(fig)


fig.show()

fig.savefig(f'{out_path}/feat_imp.png')


imp_mean = imp_df.mean(axis = 0).reset_index(drop = False).rename({'index' : 'features',
                                                                   0 : 'imp'}, axis = 1).sort_values(by = 'imp',
                                                                                                     ascending = False)


fig, ax = plt.subplots(1, 1, figsize = (100, 10))

ax.plot(np.arange(imp_mean.shape[0]), imp_mean['imp'].values,
        color = sns.color_palette()[0],
        marker = 'o',
        ls = '--')

ax.set_xticks(np.arange(imp_mean.shape[0]))
ax.set_xticklabels(imp_mean['features'].values.tolist())

fig.tight_layout()

plt.close(fig)

fig.show()

fig.savefig(f'{out_path}/feat_imp_2.png')

feat_names_selected = ['size_9',
                       '임대료_max',
                       'bus_ratio',
                       '단위주차면수',
                       'size_3',
                       '면적_45_세대수',
                       '임대보증금_max',
                       'size_2',
                       '임대료_mean',
                       '임대보증금_mean',
                       'mean_enc_supply',
                       '면적_35_세대수']

feat_names_selected = ['총세대수', '실세대수', '세대수합', '단지내주차면수'] + feat_names_selected

pickle.dump(feat_names_selected,
            open(f'{out_path}/feat_names_selected.sav', 'wb'))


# ----------------------------------------------------------------------------------------------------------------------
# 3. Part 3
# ----------------------------------------------------------------------------------------------------------------------

delete_feat_names = ['단지코드',
                     '임대건물구분',
                     '공급유형',
                     '자격유형']

feat_names_total = list(set(train_df.columns) - set(delete_feat_names))

pickle.dump(feat_names_total,
            open('data/feat_names_3.sav', 'wb'))


# ----------------------------------------------------------------------------------------------------------------------
# 4. Part 4
# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------->>> [전체 변수 대상]

feat_names_1 = ['총세대수',
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
              '임대세대외',
              '실세대수',
              '임대세대비율',
              'subway_ratio',
              'bus_ratio',
              '단위주차면수']

feat_names_2 = [x for x in train_df.columns if '면적' in x]
feat_names_3 = [x for x in train_df.columns if 'size_' in x]


feat_names_total = feat_names_1 + feat_names_2 + feat_names_3

feat_names_total.remove('총세대수')
feat_names_total.remove('세대수합')
feat_names_total.remove('실세대수')
feat_names_total.remove('단지내주차면수')

X_train = train_df[feat_names_total].values
y_train = train_df['등록차량수'].values


imp_arr = rf_imp_fn(X = X_train,
                    y = y_train,
                    n_estimators = 10000)

imp_df = pd.DataFrame(imp_arr,
                      columns = feat_names_total)

imp_mean = imp_df.mean(axis = 0).reset_index(drop = False).rename({'index' : 'features',
                                                                   0 : 'imp'}, axis = 1).sort_values(by = 'imp',
                                                                                                     ascending = False)

imp_mean.reset_index(drop = True, inplace = True)

feat_names_selected = imp_mean.head(19).features.values.tolist()

feat_names_selected = ['총세대수', '실세대수', '세대수합', '단지내주차면수'] + feat_names_selected

pickle.dump(feat_names_selected,
            open('data/feat_names_4.sav', 'wb'))


# ----------------------------------------------------------------------------------------------------------------------
# 5. Part 5
# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------->>> [전체 변수 대상]

feat_names_1 = ['총세대수',
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
              '임대세대외',
              '실세대수',
              '임대세대비율',
              'subway_ratio',
              'bus_ratio',
              '단위주차면수']

feat_names_2 = [x for x in train_df.columns if '면적' in x]
feat_names_3 = [x for x in train_df.columns if 'size_' in x]


feat_names_total = feat_names_1 + feat_names_2 + feat_names_3

feat_names_total.remove('총세대수')
feat_names_total.remove('세대수합')
feat_names_total.remove('실세대수')
feat_names_total.remove('단지내주차면수')

X_train = train_df[feat_names_total].values
y_train = train_df['등록차량수'].values


imp_arr = rf_imp_fn(X = X_train,
                    y = y_train,
                    n_estimators = 10000)

imp_df = pd.DataFrame(imp_arr,
                      columns = feat_names_total)

fig, ax = plt.subplots(1, 1, figsize = (100, 10))

sns.boxplot(data = imp_df, ax = ax)

fig.tight_layout()

plt.close(fig)


fig.show()

feat_names_selected = ['임대료_max',
                       'bus_ratio',
                       '단위주차면수',
                       '임대보증금_max',
                       'mean_enc_supply',
                       '면적_35_세대수',
                       '면적_45_세대수',
                       '면적_50_세대수',
                       'size_2',
                       'size_3',
                       'size_9']

feat_names_selected = ['총세대수', '실세대수', '세대수합', '단지내주차면수'] + feat_names_selected

pickle.dump(feat_names_selected,
            open('data/feat_names_5.sav', 'wb'))


# ----------------------------------------------------------------------------------------------------------------------
# 6. Part 6
# ----------------------------------------------------------------------------------------------------------------------


# --------------------------------------->>> [전체 변수 대상]

feat_names_1 = ['총세대수',
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
              '임대세대외',
              '실세대수',
              '임대세대비율',
              'subway_ratio',
              'bus_ratio',
              '단위주차면수']

feat_names_2 = [x for x in train_df.columns if '면적' in x]
feat_names_3 = [x for x in train_df.columns if 'size_' in x]


feat_names_total = feat_names_1 + feat_names_2 + feat_names_3

feat_names_total.remove('총세대수')
feat_names_total.remove('세대수합')
feat_names_total.remove('실세대수')
feat_names_total.remove('단지내주차면수')

X_train = train_df[feat_names_total].values
y_train = train_df['등록차량수'].values


imp_arr = rf_imp_fn(X = X_train,
                    y = y_train,
                    n_estimators = 10000)

imp_df = pd.DataFrame(imp_arr,
                      columns = feat_names_total)

imp_median = imp_df.median(axis = 0).reset_index(drop = False).rename({'index' : 'features',
                                                                   0 : 'imp'}, axis = 1).sort_values(by = 'imp',
                                                                                                     ascending = False)

imp_median.reset_index(drop = True, inplace = True)

fig, ax = plt.subplots(1, 1, figsize = (100, 10))

ax.plot(np.arange(imp_median.shape[0]), imp_median['imp'].values,
        color = sns.color_palette()[0],
        marker = 'o',
        ls = '--')

ax.axhline(y = 0,
           color = sns.color_palette()[1])

ax.set_xticks(np.arange(imp_median.shape[0]))
ax.set_xticklabels(imp_median['features'].values.tolist())


fig.tight_layout()

plt.close(fig)

# fig.show()

fig.savefig(f'{out_path}/feat_imp_3.png')


feat_names_selected = imp_median.loc[imp_median.imp > 0 , :].features.values.tolist()

feat_names_selected = ['총세대수', '실세대수', '세대수합', '단지내주차면수'] + feat_names_selected

pickle.dump(feat_names_selected,
            open('data/feat_names_6.sav', 'wb'))


# ----------------------------------------------------------------------------------------------------------------------
# 7. Part 7
# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------->>> [전체 변수 대상]

feat_names_1 = ['총세대수',
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
              '임대세대외',
              '실세대수',
              '임대세대비율',
              'subway_ratio',
              'bus_ratio',
              '단위주차면수']

feat_names_2 = [x for x in train_df.columns if '면적' in x]
feat_names_3 = [x for x in train_df.columns if 'size_' in x]


feat_names_total = feat_names_1 + feat_names_2 + feat_names_3

feat_names_total.remove('총세대수')
feat_names_total.remove('세대수합')
feat_names_total.remove('실세대수')
feat_names_total.remove('단지내주차면수')

X_train = train_df[feat_names_total].values
y_train = train_df['등록차량수'].values


imp_arr = rf_imp_fn(X = X_train,
                    y = y_train,
                    n_estimators = 10000)

imp_df = pd.DataFrame(imp_arr,
                      columns = feat_names_total)

imp_median = imp_df.median(axis = 0).reset_index(drop = False).rename({'index' : 'features',
                                                                   0 : 'imp'}, axis = 1).sort_values(by = 'imp',
                                                                                                     ascending = False)

imp_median.reset_index(drop = True, inplace = True)


feat_names_selected = imp_median.loc[imp_median.imp > 0.005 , :].features.values.tolist()

feat_names_selected = ['총세대수', '실세대수', '세대수합', '단지내주차면수'] + feat_names_selected

pickle.dump(feat_names_selected,
            open('data/feat_names_7.sav', 'wb'))

# ----------------------------------------------------------------------------------------------------------------------
# 8. Part 8
# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------->>> [전체 변수 대상]

feat_names_1 = ['총세대수',
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
              '임대세대외',
              '실세대수',
              '임대세대비율',
              'subway_ratio',
              'bus_ratio',
              '단위주차면수']

feat_names_2 = [x for x in train_df.columns if '면적' in x]
feat_names_3 = [x for x in train_df.columns if 'size_' in x]


feat_names_total = feat_names_1 + feat_names_2 + feat_names_3

feat_names_total.remove('총세대수')
feat_names_total.remove('세대수합')
feat_names_total.remove('실세대수')
feat_names_total.remove('단지내주차면수')

X_train = train_df[feat_names_total].values
y_train = train_df['등록차량수'].values


imp_arr = rf_imp_fn(X = X_train,
                    y = y_train,
                    n_estimators = 10000)

imp_df = pd.DataFrame(imp_arr,
                      columns = feat_names_total)

imp_median = imp_df.median(axis = 0).reset_index(drop = False).rename({'index' : 'features',
                                                                   0 : 'imp'}, axis = 1).sort_values(by = 'imp',
                                                                                                     ascending = False)

imp_median.reset_index(drop = True, inplace = True)


feat_names_selected = imp_median.loc[imp_median.imp > 0.04 , :].features.values.tolist()

feat_names_selected = ['총세대수', '실세대수', '세대수합', '단지내주차면수'] + feat_names_selected

pickle.dump(feat_names_selected,
            open('data/feat_names_8.sav', 'wb'))


# ----------------------------------------------------------------------------------------------------------------------
# 9. Part 9
# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------->>> [Randomforest feature importance]

feat_names_1 = ['총세대수',
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
              '임대세대외',
              '실세대수',
              '임대세대비율',
              'subway_ratio',
              'bus_ratio',
              '단위주차면수']

feat_names_2 = [x for x in train_df_2.columns if '면적' in x]
feat_names_3 = [x for x in train_df_2.columns if 'size_' in x]
feat_names_4 = [x for x in train_df_2.columns if 'region_' in x]


feat_names_total = feat_names_1 + feat_names_2 + feat_names_3 + feat_names_4

feat_names_total.remove('총세대수')
feat_names_total.remove('세대수합')
feat_names_total.remove('실세대수')
feat_names_total.remove('단지내주차면수')

X_train = train_df_2[feat_names_total].values
y_train = train_df_2['등록차량수'].values


imp_arr = rf_imp_fn(X = X_train,
                    y = y_train,
                    n_estimators = 10000)

imp_df = pd.DataFrame(imp_arr,
                      columns = feat_names_total)

imp_median = imp_df.median(axis = 0).reset_index(drop = False).rename({'index' : 'features',
                                                                   0 : 'imp'}, axis = 1).sort_values(by = 'imp',
                                                                                                     ascending = False)

imp_median.reset_index(drop = True, inplace = True)

pickle.dump(imp_median,
            open('data/imp_median_1.sav', 'wb'))


# --------------------------------------->>> [LASSO coefficient]

reg_lasso = Lasso(alpha = 0.5,
                  normalize = True,
                  max_iter = 5000)

reg_lasso.fit(X_train, y_train)


lasso_coef = pd.DataFrame({'feature' : feat_names_total,
                           'coef' : abs(reg_lasso.coef_)}).sort_values('coef', ascending = False).\
    reset_index(drop = True)


# ----- Check

fig, ax = plt.subplots(1, 1, figsize = (5, 5))

ax.scatter(train_df_2['region_35'].values, train_df_2['등록차량수'].values,
           color = sns.color_palette()[0],
           s = 5,
           alpha = 0.5)

plt.close(fig)



# feat_names_selected = ['총세대수', '실세대수', '세대수합', '단지내주차면수'] + feat_names_selected






























