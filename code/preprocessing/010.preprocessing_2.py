# -*- coding: utf-8 -*-

# ======================================================================================================================
#
#
#
#                                            10. Preprocessing 2
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
from itertools import combinations

from sklearn.model_selection import KFold
from sklearn.decomposition import NMF

exec(open('code/functions/001.feature_importance.py').read())


# --------------------------------------->>> [Set directory]


# ----- Set output path

filename = '010.preprocessing_2'

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

# ----- 데이터 로딩

raw_train_df = pd.read_csv('data/235745_parking_data/train.csv')
raw_test_df = pd.read_csv('data/235745_parking_data/test.csv')
raw_age_gender_info = pd.read_csv('data/235745_parking_data/age_gender_info.csv')
sample_submission = pd.read_csv('data/235745_parking_data/sample_submission.csv')

raw_train_df.rename({'도보 10분거리 내 지하철역 수(환승노선 수 반영)' : 'subway',
                     '도보 10분거리 내 버스정류장 수' : 'bus'}, axis = 1, inplace=  True)

raw_test_df.rename({'도보 10분거리 내 지하철역 수(환승노선 수 반영)' : 'subway',
                     '도보 10분거리 내 버스정류장 수' : 'bus'}, axis = 1, inplace = True)

# ----- 데이터 사전 전처리

delete_code_train = ['C1095',
                     'C2051',
                     'C1218',
                     'C1894',
                     'C2483',
                     'C1502',
                     'C1988',
                     'C2085',
                     'C1397',
                     'C2431',
                     'C1649',
                     'C1036']

delete_code_test = ['C2335',
                    'C1327',
                    'C2675']

raw_train_df = raw_train_df.loc[~raw_train_df['단지코드'].isin(delete_code_train), :]
raw_test_df = raw_test_df.loc[~raw_test_df['단지코드'].isin(delete_code_test), :]

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
    sum_df['세대수합'] = sub_df['전용면적별세대수'].sum()
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
    sum_df['세대수합'] = sub_df['전용면적별세대수'].sum()
    sum_df['subway'] = sub_df['subway'].unique()[0]
    sum_df['bus'] = sub_df['bus'].unique()[0]
    sum_df['단지내주차면수'] = sub_df['단지내주차면수'].unique()[0]
    sum_df['공가수'] = sub_df['공가수'].unique()[0]
    sum_df['자격유형'] = '+'.join(np.sort(sub_df['자격유형'].dropna().unique()).tolist())

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

    test_df_list.append(sum_df)

test_df = pd.concat(test_df_list)

test_df.reset_index(drop = True, inplace = True)

# --------------------------------------->>> [공급유형 범주 합치기]

in_test_cate = test_df['공급유형'].unique().tolist()

train_df = train_df.loc[train_df['공급유형'].isin(in_test_cate), :]

train_df['공급유형_merge'] = train_df['공급유형'].map({'공공임대(50년)' : '기타',
                                                  '공공임대(10년)+공공임대(분납)' : '기타',
                                                  '국민임대+영구임대+행복주택' : '기타',
                                                  '영구임대' : '기타',
                                                  '국민임대' : '국민임대',
                                                  '공공임대(10년)' : '공공임대(10년)',
                                                  '영구임대+임대상가' : '영구임대+임대상가',
                                                  '행복주택' : '행복주택',
                                                  '국민임대+영구임대' : '국민임대+영구임대'})

test_df['공급유형_merge'] = test_df['공급유형'].map({'공공임대(50년)' : '기타',
                                              '공공임대(10년)+공공임대(분납)' : '기타',
                                              '국민임대+영구임대+행복주택' : '기타',
                                              '영구임대' : '기타',
                                              '국민임대' : '국민임대',
                                              '공공임대(10년)' : '공공임대(10년)',
                                              '영구임대+임대상가' : '영구임대+임대상가',
                                              '행복주택' : '행복주택',
                                              '국민임대+영구임대' : '국민임대+영구임대'})

# --------------------------------------->>> [자격유형 범주 합치기]

# ----- Train set

selected_items_list = ['A', 'C+D', 'H', 'J', 'A+E']

tf_result = np.array([False]*train_df.shape[0])

for item in selected_items_list:

    tf_list = train_df['자격유형'].values == item

    tf_result = tf_result | tf_list


train_df['자격유형_merge'] = [train_df['자격유형'].values[ii] if tf_result[ii] else '기타' for ii in range(train_df.shape[0])]

# ----- Test set

selected_items_list = ['A', 'C+D', 'H', 'J', 'A+E']

tf_result = np.array([False]*test_df.shape[0])

for item in selected_items_list:

    tf_list = test_df['자격유형'].values == item

    tf_result = tf_result | tf_list


test_df['자격유형_merge'] = [test_df['자격유형'].values[ii] if tf_result[ii] else '기타' for ii in range(test_df.shape[0])]


# ----------------------------------------------------------------------------------------------------------------------
# 2. Categorical 변수 전처리
# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------->>> [train / test set 지역 맞추기]

train_df = train_df.loc[~train_df['지역'].isin(['서울특별시']), :]
train_df.reset_index(drop = True, inplace = True)

# --------------------------------------->>> [지역 mean encoding]

# ----- Train set

mean_enc_list = []

for tr_idx, ts_idx in KFold(n_splits = 5, shuffle = True, random_state = 0).split(train_df):

    tr_df = train_df.iloc[tr_idx, :].copy()
    ts_df = train_df.iloc[ts_idx, :].copy()

    ts_df['mean_enc_region'] = ts_df['지역'].map(tr_df.groupby('지역')['등록차량수'].mean())

    mean_enc_list.append(ts_df)

train_df = pd.concat(mean_enc_list)

global_mean = train_df['등록차량수'].mean()
train_df['mean_enc_region'].fillna(global_mean, inplace = True)

# ----- Test set

test_df['mean_enc_region'] = test_df['지역'].map(train_df.groupby('지역')['등록차량수'].mean())


# --------------------------------------->>> [공급유형 mean encoding]

# ----- Train set

mean_enc_list = []

for tr_idx, ts_idx in KFold(n_splits = 5, shuffle = True, random_state = 0).split(train_df):

    tr_df = train_df.iloc[tr_idx, :].copy()
    ts_df = train_df.iloc[ts_idx, :].copy()

    ts_df['mean_enc_supply'] = ts_df['공급유형_merge'].map(tr_df.groupby('공급유형_merge')['등록차량수'].mean())

    mean_enc_list.append(ts_df)

train_df = pd.concat(mean_enc_list)

global_mean = train_df['등록차량수'].mean()
train_df['mean_enc_supply'].fillna(global_mean, inplace = True)

# ----- Test set

test_df['mean_enc_supply'] = test_df['공급유형_merge'].map(train_df.groupby('공급유형_merge')['등록차량수'].mean())

# --------------------------------------->>> [자격유형 mean encoding]

# ----- Train set

mean_enc_list = []

for tr_idx, ts_idx in KFold(n_splits = 5, shuffle = True, random_state = 0).split(train_df):

    tr_df = train_df.iloc[tr_idx, :].copy()
    ts_df = train_df.iloc[ts_idx, :].copy()

    ts_df['mean_enc_cond'] = ts_df['자격유형_merge'].map(tr_df.groupby('자격유형_merge')['등록차량수'].mean())

    mean_enc_list.append(ts_df)

train_df = pd.concat(mean_enc_list)

global_mean = train_df['등록차량수'].mean()
train_df['mean_enc_cond'].fillna(global_mean, inplace = True)

# ----- Test set

test_df['mean_enc_cond'] = test_df['자격유형_merge'].map(train_df.groupby('자격유형_merge')['등록차량수'].mean())




train_df.reset_index(drop = True, inplace = True)
test_df.reset_index(drop = True, inplace = True)


# ----------------------------------------------------------------------------------------------------------------------
# 3. 면적 관련 변수 전처리
# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------->>> [NMF로 latent feature extraction]

size_df_tr = train_df[[x for x in train_df.columns if '면적_' in x]]
size_df_ts = test_df[[x for x in test_df.columns if '면적_' in x]]
size_df_total = pd.concat([size_df_tr, size_df_ts])

size_arr_total = size_df_total.values

nmf = NMF(n_components = 50,
          init = 'random',
          random_state = 0,
          max_iter = 300)

W = nmf.fit_transform(size_arr_total)

W_tr = W[0:size_df_tr.shape[0], :]
W_ts = W[size_df_tr.shape[0]:, :]

size_df_train = pd.DataFrame(W_tr,
                             columns = [f'size_{x}' for x in range(W_tr.shape[1])])

size_df_test = pd.DataFrame(W_ts,
                            columns = [f'size_{x}' for x in range(W_ts.shape[1])])


train_df = pd.concat([train_df, size_df_train], axis = 1)
test_df = pd.concat([test_df, size_df_test], axis = 1)

train_df.reset_index(drop = True, inplace = True)
test_df.reset_index(drop = True, inplace = True)



# ----------------------------------------------------------------------------------------------------------------------
# 4. 결측치 처리
# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------->>> [Subway]

train_df['subway'] = train_df['subway'].fillna(0)
test_df['subway'] = test_df['subway'].fillna(0)

# --------------------------------------->>> [Bus]

global_median = np.nanmedian(np.r_[train_df['bus'].values, test_df['bus'].values])

train_df['bus'] = train_df['bus'].fillna(global_median)

# --------------------------------------->>> [임대보증금_mean]

global_mean = np.nanmean(np.r_[train_df['임대보증금_mean'].values, test_df['임대보증금_mean'].values])

train_df['임대보증금_mean'] = train_df['임대보증금_mean'].fillna(global_mean)
test_df['임대보증금_mean'] = test_df['임대보증금_mean'].fillna(global_mean)

# --------------------------------------->>> [임대보증금_min]

global_mean = np.nanmean(np.r_[train_df['임대보증금_min'].values, test_df['임대보증금_min'].values])

train_df['임대보증금_min'] = train_df['임대보증금_min'].fillna(global_mean)
test_df['임대보증금_min'] = test_df['임대보증금_min'].fillna(global_mean)

# --------------------------------------->>> [임대보증금_max]

global_mean = np.nanmean(np.r_[train_df['임대보증금_max'].values, test_df['임대보증금_max'].values])

train_df['임대보증금_max'] = train_df['임대보증금_max'].fillna(global_mean)
test_df['임대보증금_max'] = test_df['임대보증금_max'].fillna(global_mean)

# --------------------------------------->>> [임대료_mean]

global_mean = np.nanmean(np.r_[train_df['임대료_mean'].values, test_df['임대료_mean'].values])

train_df['임대료_mean'] = train_df['임대료_mean'].fillna(global_mean)
test_df['임대료_mean'] = test_df['임대료_mean'].fillna(global_mean)

# --------------------------------------->>> [임대료_min]

global_mean = np.nanmean(np.r_[train_df['임대료_min'].values, test_df['임대료_min'].values])

train_df['임대료_min'] = train_df['임대료_min'].fillna(global_mean)
test_df['임대료_min'] = test_df['임대료_min'].fillna(global_mean)

# --------------------------------------->>> [임대료_max]

global_mean = np.nanmean(np.r_[train_df['임대료_max'].values, test_df['임대료_max'].values])

train_df['임대료_max'] = train_df['임대료_max'].fillna(global_mean)
test_df['임대료_max'] = test_df['임대료_max'].fillna(global_mean)



# ----------------------------------------------------------------------------------------------------------------------
# 5. 해석 가능한 파생 변수 생성
# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------->>> [임대세대외]

train_df['임대세대외'] = train_df['총세대수'] - train_df['세대수합']
test_df['임대세대외'] = test_df['총세대수'] - test_df['세대수합']

# --------------------------------------->>> [실세대수]

train_df['실세대수'] = train_df['총세대수'] - train_df['공가수']
test_df['실세대수'] = test_df['총세대수'] - test_df['공가수']

# --------------------------------------->>> [임대세대 비율]

train_df['임대세대비율'] = train_df['세대수합'] / train_df['총세대수']
test_df['임대세대비율'] = test_df['세대수합'] / test_df['총세대수']

# --------------------------------------->>> [지하철 / 실세대수]

train_df['subway_ratio'] = train_df['subway'] / train_df['실세대수']
test_df['subway_ratio'] = test_df['subway'] / test_df['실세대수']

# --------------------------------------->>> [버스정류장 / 실세대수]

train_df['bus_ratio'] = train_df['bus'] / train_df['실세대수']
test_df['bus_ratio'] = test_df['bus'] / test_df['실세대수']

# --------------------------------------->>> [세대 당 주차면수]

train_df['단위주차면수'] = train_df['단지내주차면수'] / train_df['실세대수']
test_df['단위주차면수'] = test_df['단지내주차면수'] / test_df['실세대수']


# ----------------------------------------------------------------------------------------------------------------------
# 6. 인구정보 합치기
# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------->>> [컬럼명 변경]

raw_age_gender_info.columns = list(map(lambda x : x.replace('(', '_').replace(')', ''),
                                       raw_age_gender_info.columns))

train_df = pd.merge(train_df, raw_age_gender_info,
                    on = '지역',
                    how = 'left')

test_df = pd.merge(test_df, raw_age_gender_info,
                   on = '지역',
                   how = 'left')

train_df.reset_index(drop = True, inplace = True)
test_df.reset_index(drop = True, inplace = True)




# ----------------------------------------------------------------------------------------------------------------------
# 7. 현재까지 상황을 바탕으로 feature importance 확인하기
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

feat_names_selected = imp_median.loc[imp_median.imp > 0.01 , :].features.values.tolist()

feat_names_selected = ['총세대수', '실세대수', '세대수합', '단지내주차면수'] + feat_names_selected

train_df_selected = train_df[feat_names_selected].copy()
train_df_target = train_df['등록차량수'].values

test_df_selected = test_df[feat_names_selected].copy()


# ----------------------------------------------------------------------------------------------------------------------
# 8. Feature generation part 1
# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------->>> [Pairwise sum / abs diff / mutiply]

feat_names_pair = list(combinations(train_df_selected.columns.tolist(), 2))

for combi in feat_names_pair:

    # ----- Train

    feat_1 = train_df_selected[combi[0]].values
    feat_2 = train_df_selected[combi[1]].values

    sum_ = feat_1 + feat_2
    abs_diff_ = abs(feat_1 - feat_2)
    multip_ = feat_1 * feat_2

    train_df_selected[f'{combi[0]}_{combi[1]}_sum'] = sum_
    train_df_selected[f'{combi[0]}_{combi[1]}_abs_diff'] = abs_diff_
    train_df_selected[f'{combi[0]}_{combi[1]}_multip'] = multip_

    # ----- Test

    feat_1 = test_df_selected[combi[0]].values
    feat_2 = test_df_selected[combi[1]].values

    sum_ = feat_1 + feat_2
    abs_diff_ = abs(feat_1 - feat_2)
    multip_ = feat_1 * feat_2

    test_df_selected[f'{combi[0]}_{combi[1]}_sum'] = sum_
    test_df_selected[f'{combi[0]}_{combi[1]}_abs_diff'] = abs_diff_
    test_df_selected[f'{combi[0]}_{combi[1]}_multip'] = multip_


# --------------------------------------->>> [변수 중요도 확인]

feat_names = train_df_selected.columns.tolist()
# feat_names.remove('총세대수')
# feat_names.remove('세대수합')
# feat_names.remove('실세대수')
# feat_names.remove('단지내주차면수')


X_train = train_df_selected[feat_names].values
y_train = train_df_target

imp_arr = rf_imp_fn(X = X_train,
                    y = y_train,
                    n_estimators = 10000)

imp_df = pd.DataFrame(imp_arr,
                      columns = feat_names)

imp_median = imp_df.median(axis = 0).reset_index(drop = False).rename({'index' : 'features',
                                                                   0 : 'imp'}, axis = 1).sort_values(by = 'imp',
                                                                                                     ascending = False)

imp_median.reset_index(drop = True, inplace = True)



# ----------------------------------------------------------------------------------------------------------------------
# 9. Train / Test set 분포비교
# ----------------------------------------------------------------------------------------------------------------------

feat_total = train_df_selected.columns.tolist()
feat_order = imp_median.features.tolist()

for feat in feat_total:

    # feat = feat_total[0]

    fig, axes  = plt.subplots(1, 2, figsize = (20, 10))
    ax_0, ax_1 = axes[0], axes[1]

    tr_ = train_df_selected[feat].values
    ts_ = test_df_selected[feat].values

    # ----- histogram

    sns.histplot(x = tr_,
                 color = sns.color_palette()[0],
                 alpha = 0.5,
                 ax = ax_0,
                 label = 'Train',
                 stat = 'density')

    sns.histplot(x = ts_,
                 color = sns.color_palette()[1],
                 alpha = 0.5,
                 ax = ax_0,
                 label = 'Test',
                 stat = 'density')

    ax_0.legend()

    # ----- Boxplot

    plot_df = pd.DataFrame({'set' : ['Train']*len(tr_) + ['Test']*len(ts_),
                            'value' : np.r_[tr_, ts_]})

    sns.boxplot(data = plot_df,
                x = 'set',
                y = 'value',
                ax = ax_1)

    fig.suptitle(f'{feat} : {feat_order.index(feat)}/{len(feat_order)}',
                 fontsize = 20)

    plt.close(fig)

    fig.savefig(os.path.join(out_path, f'{feat}.png'))







pickle.dump(train_df, open('data/train_df_2.sav', 'wb'))
pickle.dump(test_df, open('data/test_df_2.sav', 'wb'))