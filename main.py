# %%
# Imports
import pandas as pd
import numpy as np

import lightgbm as lgm
import catboost as catb
import xgboost as xgb
import pymorphy2 as ph

import pickle

from deslib.des import DESKL

from itertools import combinations

from tqdm import tqdm

from sklearn.linear_model import *

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA, FastICA

from sklearn.feature_selection import RFECV

from sklearn.preprocessing import *

from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier

from sklearn.cluster import KMeans, OPTICS, AffinityPropagation, MiniBatchKMeans

from sklearn.neighbors import *

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, \
                            ExtraTreesClassifier, IsolationForest

from datetime import date, timedelta

import optuna

np.random.seed(42)

n_tfidf_comps = 100

# %%
# Data load
# загружаем, даункастим типы
# также я беру log1p от суммы транзакции, поскольку это позитивно влияет на распределения и cv в целом.

train_trans = pd.read_csv('transactions_train.csv')
train_target = pd.read_csv('train_target.csv')
descript = pd.read_csv('smg_hand.csv')
descript_spectral = pd.read_csv('small_group_descript_new.csv')

pred_good = pd.read_csv('Pseudo_data.csv')

train_trans.client_id = train_trans.client_id.astype(np.uint16)
train_trans.trans_date = train_trans.trans_date.astype(np.uint16)
train_trans.small_group = train_trans.small_group.astype(np.uint8)
train_trans.amount_rur = np.log1p(train_trans.amount_rur.astype(np.float32))

test = pd.read_csv('test.csv')
test_trans = pd.read_csv('transactions_test.csv')

test_trans.client_id = test_trans.client_id.astype(np.uint16)
test_trans.trans_date = test_trans.trans_date.astype(np.uint16)
test_trans.small_group = test_trans.small_group.astype(np.uint8)
test_trans.amount_rur = np.log1p(test_trans.amount_rur.astype(np.float32))

## Считаем, что нам дали 2017 и 2018 годы
## Хотя это, конечно, некорректно просто так считать

# %%
# FE

## Отклонение от exp среднего. Берем 0, 25, 50, 75 и 100 процентили.

# deviations_train = np.zeros((30000, 5))
# for n, client_id in tqdm(enumerate(train_target.client_id.values)):
#     asdfff = train_trans[train_trans.client_id==client_id].amount_rur.values[1:]-\
#         train_trans[train_trans.client_id==client_id].amount_rur.ewm(com=0.8).mean().values[:-1]
#     for a in range(5):
#         deviations_train[n, a] = np.percentile(asdfff, a*25)

# deviations_test = np.zeros((20000, 5))
# for n, client_id in tqdm(enumerate(test.client_id.values)):
#     asdfff = test_trans[test_trans.client_id==client_id].amount_rur.values[1:]-\
#         test_trans[test_trans.client_id==client_id].amount_rur.ewm(com=0.8).mean().values[:-1]
#     for a in range(5):
#         deviations_test[n, a] = np.percentile(asdfff, a*25)

# deviations_train = deviations_train.astype(np.float32)
# deviations_test = deviations_test.astype(np.float32)

# I should turn 194 and 201 into 203

"""
    мы знаем год, следовательно можем восстановить месяцы.
    по ним и будем в частности агрегировать.
"""

monthes_by_trans_date = np.zeros((730, 1))

for x in range(730):
    monthes_by_trans_date[x] = (date(2017, 1, 1) + timedelta(x)).month + ((date(2017, 1, 1) + timedelta(x)).year-2017)*12

monthes_by_trans_date = monthes_by_trans_date.astype(np.uint8)

train_trans['month'] = monthes_by_trans_date[train_trans.trans_date].astype(np.uint8)
test_trans['month'] = monthes_by_trans_date[test_trans.trans_date].astype(np.uint8)

"""
 код снизу отвечает за подготовку в train_trans для каждой транзакции её позиции внутри дня
 я заметил, что у некоторых клиентов транзакции внутри одного дня могут идти в определенном порядке, 
 например, клиент может посещать продуктовый последним магазином в дне
 Судя по всему, внутри дня транзакции никак не менялись. (их не перемешивали, не меняли структуры)
"""

client_n = train_trans.client_id.values
pos = 0
prev_client_id = 0
prev_trans_date = 0
nid = np.zeros((train_trans.shape[0], 1))
for n, a in enumerate(tqdm(train_trans.values)):
    if a[0] == prev_client_id and a[1] == prev_trans_date:
        pos+=1
        nid[n] = pos
    else:
        pos=0
    prev_client_id = a[0]
    prev_trans_date = a[1]
train_trans['number_inside_day'] = nid.astype(np.uint8)

client_n = test_trans.client_id.values
pos = 0
prev_client_id = 0
prev_trans_date = 0
nid = np.zeros((test_trans.shape[0], 1))
for n, a in enumerate(tqdm(test_trans.values)):
    if a[0] == prev_client_id and a[1] == prev_trans_date:
        pos+=1
        nid[n] = pos
    else:
        pos=0
    prev_client_id = a[0]
    prev_trans_date = a[1]
test_trans['number_inside_day'] = nid.astype(np.uint8)

"""

 Я брал группы, лемматизировал их названия, и кормил в GLOVE, правда с малым словарем, лень качать большой было.
 Затем, cosine similarity и SpectralClustering.
 Пробовал объединять по количеству появлений вместе, но результат мне не понравился.

"""

spectralgr = descript_spectral.sort_values(by='small_group_code').spectral_group.values

train_trans['spectral_group'] = spectralgr[train_trans.small_group]
test_trans['spectral_group'] = spectralgr[test_trans.small_group]


### aggregations over each client

group = train_trans.groupby('client_id').\
    amount_rur.agg(['sum', 'mean', 'std', 'max', 'count', 'median']).astype(np.float32)
group = group.rename(columns={'sum' : 'sum_cl', 'mean': 'mean_cl',
                              'std' : 'std_cl','max' : 'max_cl',
                              'median':'median_cl', 'count':'count_cl'})
train_target = train_target.merge(group, left_on = 'client_id', right_on = 'client_id')

group = test_trans.groupby('client_id').\
    amount_rur.agg(['sum', 'mean', 'std', 'max', 'count', 'median']).astype(np.float32)
group = group.rename(columns={'sum' : 'sum_cl','mean': 'mean_cl',
                              'std' : 'std_cl','max' : 'max_cl',
                              'median':'median_cl', 'count':'count_cl'})
test = test.merge(group, left_on = 'client_id', right_on = 'client_id')

group = train_trans.groupby(['client_id', 'month']).\
    amount_rur.agg(['sum', 'mean', 'std', 'max', 'count', 'median']).astype(np.float32)
group = group.rename(columns={'sum' : 'sum_cl_month', 'std' : 'std_cl_month',
                              'mean': 'mean_cl_month',
                              'max' : 'max_cl_month', 'median':'median_cl_month',
                              'count':'count_cl_month'})
group = group.unstack().fillna(0)
train_target = train_target.merge(group, how='left', left_on = 'client_id', right_on='client_id')

group = test_trans.groupby(['client_id', 'month']).\
    amount_rur.agg(['sum', 'mean', 'std', 'max', 'count', 'median']).astype(np.float32)
group = group.rename(columns={'sum' : 'sum_cl_month', 'std' : 'std_cl_month',
                              'mean': 'mean_cl_month',
                              'max' : 'max_cl_month', 'median':'median_cl_month',
                              'count':'count_cl_month'})
group = group.unstack().fillna(0)
test = test.merge(group, how='left', left_on = 'client_id', right_on='client_id')

### SUM, 75%, std, 25%, count spendings on each group

group = train_trans.groupby(['client_id', 'small_group']).\
    amount_rur.agg(['sum', 'mean', 'std', 'max', 'count', 'median']).astype(np.float32)
group = group.rename(columns={'sum' : 'sum_group', 'std' : 'std_group',
                              'mean': 'mean_group',
                              'max' : 'max_group', 'count':'count_group',
                              'median':'median_group'})
group = group.unstack().fillna(0)
train_target = train_target.merge(group, how='left', left_on = 'client_id', right_on='client_id')

group = test_trans.groupby(['client_id', 'small_group']).\
    amount_rur.agg(['sum', 'mean', 'std', 'max', 'count', 'median']).astype(np.float32)
group = group.rename(columns={'sum' : 'sum_group', 'std' : 'std_group',
                              'mean': 'mean_group',
                              'max' : 'max_group', 'count':'count_group',
                              'median':'median_group'})
group = group.unstack().fillna(0)
test = test.merge(group, how='left', left_on = 'client_id', right_on='client_id')

### Spectral group

group = train_trans.groupby(['client_id', 'spectral_group']).\
    amount_rur.agg(['sum', 'mean', 'std',  'max', 'count', 'median']).astype(np.float32)
group = group.rename(columns={'sum' : 'sum_Sgroup', 'std' : 'std_Sgroup',
                              'mean': 'mean_Sgroup',
                              'max' : 'max_Sgroup', 'count':'count_Sgroup',
                              'median':'median_Sgroup'})
group = group.unstack().fillna(0)
train_target = train_target.merge(group, how='left', left_on = 'client_id', right_on='client_id')

group = test_trans.groupby(['client_id', 'spectral_group']).\
    amount_rur.agg(['sum', 'mean', 'std', 'max', 'count', 'median']).astype(np.float32)
group = group.rename(columns={'sum' : 'sum_Sgroup', 'std' : 'std_Sgroup',
                              'mean': 'mean_Sgroup',
                              'max' : 'max_Sgroup', 'count':'count_Sgroup',
                              'median':'median_Sgroup'})
group = group.unstack().fillna(0)
test = test.merge(group, how='left', left_on = 'client_id', right_on='client_id')

### Average position for every group and aggs for every position

group = train_trans.groupby(['client_id', 'small_group']).\
    number_inside_day.agg(['mean', 'std']).astype(np.float32)
group = group.rename(columns={'mean': 'mean_n',
                              'std':  'std_n'})
group = group.unstack().fillna(0)
train_target = train_target.merge(group, how='left', left_on = 'client_id', right_on='client_id')

group = test_trans.groupby(['client_id', 'small_group']).\
    number_inside_day.agg(['mean', 'std']).astype(np.float32)
group = group.rename(columns={'mean': 'mean_n',
                              'std':  'std_n'})
group = group.unstack().fillna(0)
test = test.merge(group, how='left', left_on = 'client_id', right_on='client_id')

group = train_trans.groupby(['client_id', 'number_inside_day']).\
    amount_rur.agg(['sum', 'mean', 'std',  'max', 'count', 'median']).astype(np.float32)
group = group.rename(columns={'sum' : 'sum_nid', 'std' : 'std_nid',
                              'mean': 'mean_nid',
                              'max' : 'max_nid', 'count':'nid',
                              'median':'median_nid'})
group = group.unstack().fillna(0)
train_target = train_target.merge(group, how='left', left_on = 'client_id', right_on='client_id')

group = test_trans.groupby(['client_id', 'number_inside_day']).\
    amount_rur.agg(['sum', 'mean', 'std', 'max', 'count', 'median']).astype(np.float32)
group = group.rename(columns={'sum' : 'sum_nid', 'std' : 'std_nid',
                              'mean': 'mean_nid',
                              'max' : 'max_nid', 'count':'nid',
                              'median':'median_nid'})
group = group.unstack().fillna(0)
test = test.merge(group, how='left', left_on = 'client_id', right_on='client_id')

### TF-IDF for each client


"""
     берем названия групп трат человека в один большой текст и TF-IDF + dictionary learning.
"""

morph = ph.MorphAnalyzer()

descript.small_group = [' '.join([morph.parse(word)[0].normal_form for word in obj.split()])\
                        for obj in descript.small_group]
desc = pd.Series(descript.small_group, index=descript.small_group_code)

group = train_trans.groupby('client_id')['small_group'].agg(lambda x: ' '.join(map(str, x)))
group2 = test_trans.groupby('client_id')['small_group'].agg(lambda x: ' '.join(map(str, x)))
full = group.append(group2)
total_sents = []
for sent in tqdm(full):
    total_sents += [' '.join(desc[list(map(int, sent.split()))].values.tolist())]

total_sents = TfidfVectorizer().fit_transform(total_sents)
total_sents = TruncatedSVD(n_components=n_tfidf_comps, random_state=42).fit_transform(total_sents).astype(np.float32)
total_sents = pd.DataFrame(total_sents,
                           columns = [f'tfidf_{a}'for a in range(n_tfidf_comps)],
                           index = full.index).astype(np.float32)
train_target = train_target.merge(total_sents, how='left',
                                  left_on='client_id', right_on='client_id')
test = test.merge(total_sents, how='left', on='client_id')

# Fourier for every month and client
"""
    дальше из фич будет:
        1) Фурье по среднему всех трат за 24 месяца
        2) изменение в тратах за 12 месяцев. (новый год - старый)
        3) минимальное количество групп, которые составляют: 30% трат, 50 процентов, 70 и 90 (помогает определить diversity в тратах человека)
        4) разница в тратах между первой категорией и второй, второй и третьей и так 10 категорий (сортируются в порядке потраченной суммы)
        5) dictionary learning опять: берем все категории и суммы трат, раскладываем с TruncatedSVD.
        6) разница между количеством из фичи номер 3
"""

columns = []
for a in range(1, 25):
    columns += [('mean_cl_month', a)]
sums_per_month = train_target[columns]
sums_per_month = StandardScaler().fit_transform(sums_per_month)
fourier = np.fft.rfft(sums_per_month)
fourier = np.concatenate([fourier.real, fourier.imag], axis=1)

# delta sum for every client from month to month

columns = []
for a in range(1, 25):
    columns += [('mean_cl_month', a)]
delta_mean_year = train_target[columns].iloc[:, 12:24].values - \
    train_target[columns].iloc[:, :12].values

columns = [col for col in train_target.columns if col[0] == 'sum_group']
columns2 = [col[1] for col in train_target.columns if col[0] == 'sum_group']
best_groups = np.array(columns2)[train_target[columns].values.argsort(axis=1)[:, -10:]]

percentile_spendings = np.percentile(train_target[columns].values, 95, axis=1)

counts = np.zeros((30000, len(train_trans.small_group.unique())))
ff = train_target[columns].values.argsort(axis=1)[:, ::-1].reshape(-1, len(train_trans.small_group.unique()))
asdf = train_target[columns].values
for a in tqdm(range(30000)):
    counts[a] = asdf[a, ff[a]].cumsum()

counts /= train_target['sum_cl'].values.reshape(-1, 1)

count_30 = (counts >= 0.3).argmax(axis=1)+1
count_50 = (counts >= 0.5).argmax(axis=1)+1
count_70 = (counts >= 0.7).argmax(axis=1)+1
count_90 = (counts >= 0.9).argmax(axis=1)+1

top_groups_delta = np.zeros((30000, 9))
for a in tqdm(range(30000)):
    for b in range(9):
        top_groups_delta[a, b] = train_target[('sum_group', best_groups[a, b+1])].iloc[a] - \
                            train_target[('sum_group', best_groups[a, b])].iloc[a]

columns = [col for col in train_target.columns if col[0] == 'sum_group']
spsvd = TruncatedSVD(n_components=20, random_state=42).fit(
    np.concatenate([train_target[columns].values, test[columns].values], axis = 0)
)

spends_svd = spsvd.transform(train_target[columns].values)

counts = [count_90, count_70, count_50, count_30]
counts_diff = np.zeros((30000, 6))
for n, grp in enumerate(combinations(counts, 2)):
    counts_diff[:, n] = grp[0] - grp[1]

X, y = train_target.drop(columns=['client_id', 'bins']), train_target['bins']
X, y = X.values, y.values

pca_pipe = make_pipeline(
    PCA(n_components=4, random_state=42),
)

testN = test.drop(columns = ['client_id']).values

pca_pipe.fit(np.concatenate([
    X[:, -n_tfidf_comps:],
    testN[:, -n_tfidf_comps:]], axis=0))

"""
    на наборе данных X_full LightGBM начинает ухудшать свои результаты, поэтому он обучается на X_agg
"""

X_agg = np.concatenate([X,
                    pca_pipe.transform(X[:, -n_tfidf_comps:]),
                    top_groups_delta,
                    spends_svd], axis=1).astype(np.float32)

X_full = np.concatenate([X,
                    pca_pipe.transform(X[:, -n_tfidf_comps:]),
                    fourier,
                    delta_mean_year,
                    percentile_spendings.reshape(-1, 1),
                    top_groups_delta, spends_svd, count_30.reshape(-1, 1),
                    count_50.reshape(-1, 1), count_70.reshape(-1, 1),
                    count_90.reshape(-1, 1)], axis=1).astype(np.float32)

"""
    все те же самые фичи, теперь и на тесте
"""


columns = []
for a in range(1, 25):
    columns += [('mean_cl_month', a)]
sums_per_month = test[columns]
sums_per_month = StandardScaler().fit_transform(sums_per_month)
fourier = np.fft.rfft(sums_per_month)
fourier = np.concatenate([fourier.real, fourier.imag], axis=1)

columns = []
for a in range(1, 25):
    columns += [('mean_cl_month', a)]
delta_mean_year = test[columns].iloc[:, 12:24].values - \
    test[columns].iloc[:, :12].values

columns = [col for col in test.columns if col[0] == 'sum_group']
columns2 = [col[1] for col in test.columns if col[0] == 'sum_group']
best_groups = np.array(columns2)[test[columns].values.argsort(axis=1)[:, -10:]]

percentile_spendings = np.percentile(test[columns].values, 95, axis=1)

counts = np.zeros((20000, len(test_trans.small_group.unique())))
ff = test[columns].values.argsort(axis=1)[:, ::-1].reshape(-1, len(test_trans.small_group.unique()))
asdf = test[columns].values
for a in tqdm(range(20000)):
    counts[a] = asdf[a, ff[a]].cumsum()

counts /= test['sum_cl'].values.reshape(-1, 1)

count_30 = (counts >= 0.3).argmax(axis=1)+1
count_50 = (counts >= 0.5).argmax(axis=1)+1
count_70 = (counts >= 0.7).argmax(axis=1)+1
count_90 = (counts >= 0.9).argmax(axis=1)+1

top_groups_delta = np.zeros((20000, 9))
for a in tqdm(range(20000)):
    for b in range(9):
        top_groups_delta[a, b] = test[('sum_group', best_groups[a, b+1])].iloc[a] - \
                            test[('sum_group', best_groups[a, b])].iloc[a]

columns = [col for col in train_target.columns if col[0] == 'sum_group']
spends_svd = spsvd.transform(test[columns].values)

counts = [count_90, count_70, count_50, count_30]
counts_diff = np.zeros((20000, 6))
for n, grp in enumerate(combinations(counts, 2)):
    counts_diff[:, n] = grp[0] - grp[1]

testN = test.drop(columns='client_id').values
test_agg = np.concatenate([testN,
                    pca_pipe.transform(testN[:, -n_tfidf_comps:]),
                    top_groups_delta,
                    spends_svd], axis=1)

test_full = np.concatenate([testN,
                    pca_pipe.transform(testN[:, -n_tfidf_comps:]),
                    fourier,
                    delta_mean_year,
                    percentile_spendings.reshape(-1, 1),
                    top_groups_delta, spends_svd, count_30.reshape(-1, 1),
                    count_50.reshape(-1, 1), count_70.reshape(-1, 1),
                    count_90.reshape(-1, 1)], axis=1)

cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

"""
МОДЕЛЬ!

самое интересное во всем соревновании.

Сразу к нейросетям. Я использовал комбо из QuantileTransformer для подготовки данных + 
                                                        PCA для декоррелирования
Над нейросетью не парился, взял дефолтную из sklearn с ручным тюнингом параметров (еще optuna настраивать для тюнинга, ха)
Но увидев точность в районе 62-63 на cv не обрадовался и пошел думать.
Предположив, что модель плохо экстраполирует, решил попробовать Pseudo-labeling. 
И хотя он работает хорошо при высокой точности на тесте, 
он в идее должен быть полезен для модели, чтобы она просто увидела возможные границы.
В чем заключается эта техника:
    если точность на тесте высокая, то можно сделать предсказания и скормить модели новые данные
    это улучшает точность модели, референс https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52557
В теории, даже при низкой точности результаты модели должны улучшиться.
так и произошло, но есть несколько нюансов:
    1) модель брала на вход предикшны с высокой точностью
    2) на cv модель показывает хорошую точность благодаря тому, что модель начала бить пространство так же, как и бустинги 
    (подразумевается не то, что нейросеть превращается в бустинг), которые обучились на полном наборе данных, что может быть ликейджем.
с Pseudo-labeling точность подскочила почти до точности бустингов.

Лучше всего себя показали кэтбуст и лайтгбм. Хгбуст оверфитит и не приносит никакой пользы. Возможно мало тюнил.
Лайтгбм хотя бы точность на тесте дает неплохую

Я натюнил лайтгбм и перенес параметры на кэтбуст с небольшим ручным тюнингом.

У нас есть два набора данных, X_agg и X_full.
На первом мы обучаем лайтгбм и кэтбуст, на втором нейросеть и кэтбуст
Плюсом к этому на 30% самых лучших фич из X_full по мнению кэтбуста я обучил такую же нейросеть

Теперь у нас есть по две модели для каждого набора данных, их мы объединяем с помощью Dynamic Ensemble Selection.
Про эту методику можете почитать отдельно. Я лишь скажу, что использовал DES-KL, 
        поскольку объединять лайтгбм, который выдает на трейне точность 0.98, и кэтбуст с 0.8 сложно, ведь ORACLE всегда будет брать именно лайтгбм. это логично, но на тесте его точность может быть ниже.
        DESKL не переобучается так сильно, как другие попробованные мною методы.

DESы, обученные на X_agg и на X_full, блендятся без взвешивания.
К ним прибавляется отдельная нейросеть с малым весом, равным 0.2.

"""


model_cat_single = catb.CatBoostClassifier(learning_rate=0.04,
                                    bootstrap_type='Bernoulli',
                                    subsample=0.1,
                                    max_depth=4,
                                    iterations=5000,
                                    rsm=0.09,
                                    task_type='CPU',
                                    l2_leaf_reg=3,
                                    thread_count=12,
                                    use_best_model=False, silent=True,
                                    eval_metric = 'Accuracy')

model_cat_full = catb.CatBoostClassifier(learning_rate=0.04,
                                    bootstrap_type='Bernoulli',
                                    subsample=0.1,
                                    max_depth=4,
                                    rsm = 0.09,
                                    iterations=5000,
                                    task_type='CPU',
                                    l2_leaf_reg=3,
                                    thread_count=12,
                                    use_best_model=False, silent=True,
                                    eval_metric = 'Accuracy')

model_lgb_single = lgm.LGBMClassifier(max_depth=7, num_leaves=31,
                                      learning_rate=0.026701767646119645,
                                    n_estimators=1500,
                                    subsample=0.1362923276112757,
                                    colsample_bytree=0.11688068510900741,
                                    lambda_l2=2.842306161342649,
                                    n_jobs=6, device='CPU', eval_metric='Accuracy',
                                    importance_type='gain')

model_logit = make_pipeline(
    QuantileTransformer(random_state=42),
    PCA(random_state=42),
    # xgb.XGBClassifier(booster='gblinear', nthread=12,
    #                   silent=False, learning_rate=0.1, n_estimators=250, subsample=0.1)
    MLPClassifier(max_iter=7, activation = 'relu', batch_size=64,
                  nesterovs_momentum=False, random_state=42, learning_rate_init=0.0005)
)

model_logit_best = make_pipeline(
    QuantileTransformer(random_state=42),
    PCA(random_state=42),
    MLPClassifier(max_iter=7, activation = 'relu', batch_size=64,
                  nesterovs_momentum=False, random_state=42, learning_rate_init=0.0005)
)

cv_results = []
cv_results2 = []
cv_results3 = []

afff = cv.split(X, y)
# next(afff)

for tr, te in afff:
    print(len(cv_results)+1, 'fold started')

    model_cat_single.fit(X_agg[tr], y[tr])
    print('Solo cat', accuracy_score(y[te], model_cat_single.predict(X_agg[te])))

    model_lgb_single.fit(X_agg[tr], y[tr])
    print('Solo lgb', accuracy_score(y[te], model_lgb_single.predict(X_agg[te])))

    des1 = DESKL([model_lgb_single, model_cat_single], random_state=42)
    des1.fit(X_agg[tr], y[tr])
    print('DES single', accuracy_score(y[te], des1.predict(X_agg[te])))

    pseudo_data_X = np.concatenate([X_full[tr], test_full], axis=0)
    pseudo_data_y = np.concatenate([y[tr], pred_good.bins.values], axis=0)

    model_cat_full.fit(X_full[tr], y[tr])
    print('Solo catfull', accuracy_score(y[te], model_cat_full.predict(X_full[te])))

    model_logit.fit(pseudo_data_X, pseudo_data_y)
    print('Solo linmodel', accuracy_score(y[te],
                        model_logit.predict(X_full[te])))

    des2 = DESKL([model_cat_full, model_logit], random_state=42)
    des2.fit(X_full[tr], y[tr])
    print('DES full', accuracy_score(y[te], des2.predict(X_full[te])))

    pred_des = des1.predict_proba(X_agg[te]) + \
               des2.predict_proba(X_full[te])

    feats = model_cat_full.feature_importances_ >= \
                      np.percentile(model_cat_full.feature_importances_, 30)

    model_logit_best.fit(pseudo_data_X[:, feats], pseudo_data_y)
    pred_des_logit = pred_des + model_logit_best.predict_proba(X_full[te][:, feats]) * 0.2

    print('DES\'s accuracies', accuracy_score(y[te], pred_des.argmax(axis=1)))
    print('DES\'s accuracies + logit_feats', accuracy_score(y[te],
                                             pred_des_logit.argmax(axis=1)), '\n')

    cv_results2 += [accuracy_score(y[te], pred_des.argmax(axis=1))]
    cv_results3 += [accuracy_score(y[te], pred_des_logit.argmax(axis=1))]

print('Results without and with linmodel, and DES+logit')
# print(np.mean(cv_results), '+-', np.std(cv_results))
print(np.mean(cv_results2), '+-', np.std(cv_results2))
print(np.mean(cv_results3), '+-', np.std(cv_results3))


pseudo_data_X = np.concatenate([X_full, test_full], axis=0)
pseudo_data_y = np.concatenate([y, pred_good.bins.values], axis=0)

model_lgb_single.fit(X_agg, y)
model_cat_single.fit(X_agg, y)
model_cat_full.fit(X_full, y)
model_logit.fit(pseudo_data_X, pseudo_data_y)

des1 = DESKL([model_lgb_single, model_cat_single], random_state=42)
des1.fit(X_agg, y)

des2 = DESKL([model_cat_full, model_logit], random_state=42)
des2.fit(X_full, y)

pred_des = des1.predict_proba(test_agg) + des2.predict_proba(test_full)

feats = model_cat_full.feature_importances_ >=\
                  np.percentile(model_cat_full.feature_importances_, 30)

model_logit_best.fit(pseudo_data_X[:, feats], pseudo_data_y)
pred_des_logit = pred_des + model_logit_best.predict_proba(test_full[:, feats]) * 0.2

# neural_best = model_neural.predict(test_full)

pred = pd.DataFrame(pred_des_logit.argmax(axis=1), index = test.client_id, columns = ['bins'])
pred.to_csv('submission_.csv', )


#%% lgb hpopt

logit_full = []
for tr, te in cv.split(X, y):
    pseudo_data_X = np.concatenate([X_full[tr], test_full], axis=0)
    pseudo_data_y = np.concatenate([y[tr], pred_good.bins.values], axis=0)

    model_logit.fit(pseudo_data_X, pseudo_data_y)
    logit_full += [copy.deepcopy(model_logit)]

logit_best = []
for tr, te in cv.split(X, y):
    pseudo_data_X = np.concatenate([X_full[tr], test_full], axis=0)
    pseudo_data_y = np.concatenate([y[tr], pred_good.bins.values], axis=0)

    model_cat_full = cat_full[len(logit_best)]
    feats = model_cat_full.feature_importances_ >= \
                  np.percentile(model_cat_full.feature_importances_, 50)
    model_logit_best.fit(pseudo_data_X[:, feats], pseudo_data_y)
    logit_best += [copy.deepcopy(model_logit_best)]

des2_preds = []
for tr, te in cv.split(X, y):
    model_cat_full = cat_full[len(des2_preds)]
    model_logit = logit_full[len(des2_preds)]
    des2 = DESKL([model_cat_full, model_logit], random_state=42)
    des2.fit(X_full[tr], y[tr])
    des2_preds += [des2.predict_proba(X_full[te])]
    print(accuracy_score(y[te], des2.predict(X_full[te])))

mlp_preds = []
for tr, te in cv.split(X, y):
    model_logit_best = logit_best[len(mlp_preds)]
    model_cat_full = cat_full[len(mlp_preds)]
    feats = model_cat_full.feature_importances_ >= \
                      np.percentile(model_cat_full.feature_importances_, 50)
    mlp_preds += [model_logit_best.predict_proba(X_full[te][:, feats])]

def boost_lgb(trial):
    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
    model_lgb = lgm.LGBMClassifier(max_depth=trial.suggest_int('max_depth', 1, 7),
                                   num_leaves=trial.suggest_int('max_leaves', 8, 50),
                                   learning_rate=trial.suggest_uniform('learning_rate', 0.01, 0.04),
                                   n_estimators=trial.suggest_int('n_estimators', 400, 1500),
                                   colsample_bytree=trial.suggest_uniform('colsample_bytree', 0, 1),
                                   lambda_l2=trial.suggest_uniform('lambda_l2', 0, 5),
                                   random_state=42,
                                   n_jobs=6)
    print(model_lgb)
    cv_results = []
    for tr, te in cv.split(X, y):
        model_lgb.fit(X_agg[tr], y[tr])
        des1 = DESKL([model_lgb, cat_single[len(cv_results)]], random_state=42)
        des1.fit(X_agg[tr], y[tr])
        des_pred = des2_preds[len(cv_results)] + des1.predict_proba(X_agg[te]) + \
                    mlp_preds[len(cv_results)]
        print(accuracy_score(y[te], des_pred.argmax(axis=1)))
        cv_results += [accuracy_score(y[te], des_pred.argmax(axis=1))]
    return np.mean(cv_results)

def boost_cat(trial):
    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
    model_cat = catb.CatBoostClassifier(max_depth=trial.suggest_int('max_depth', 2, 5),
                                  iterations=trial.suggest_int('n_estimators', 1500, 6000),
                                  rsm=trial.suggest_uniform('rsm', 0, 1),
                                  learning_rate=trial.suggest_uniform('learning_rate', 0.01, 0.07),
                                  bootstrap_type='Bernoulli',
                                  l2_leaf_reg=trial.suggest_uniform('l2_leaf_reg', 0, 1),
                                  thread_count=12, eval_metric = 'Accuracy',
                                  use_best_model = False,
                                  random_state=42,
                                  subsample=trial.suggest_uniform('subsample', 0, 1))
    cv_results = []
    for tr, te in cv.split(X, y):
        model_cat.fit(X[tr], y[tr], eval_set = (X[te], y[te]))
        cv_results += [accuracy_score(y[te], model_cat.predict(X[te]))]
    return np.mean(cv_results)

study = optuna.create_study(direction='maximize')
study.optimize(boost_lgb, n_trials=2000)
