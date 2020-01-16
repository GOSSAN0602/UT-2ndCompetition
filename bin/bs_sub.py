from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import sys
sys.path.append('./')
from libs.get_params import get_lgb_params
#from libs.feature_select import kolmogorov_smirnov, adversarial_del_list
import lightgbm as lgb
import numpy as np
import pandas as pd
import tables
import gc

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# load data
data_folder = '../input/'
train = pd.read_csv(data_folder+'train.csv')
train_y = train.loc[:,['quality']]
train_x = train.drop('quality',axis=1)
test_x = pd.read_csv(data_folder+'test.csv')
sub = pd.read_csv(data_folder+'submission.csv')
# for visualizing feature importance
columns = train_x.columns
feature_importances = pd.DataFrame()
feature_importances['feature'] = columns

# make fold
NFOLDS = 5
kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=71)
splits = kf.split(train_x, train_y)

# for pred
y_preds = np.zeros(test_x.shape[0])
y_oof = np.zeros(train_x.shape[0])
score = 0

# get_params
params = get_lgb_params()

for fold_n, (tr_idx, va_idx) in enumerate(splits):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
    dtrain = lgb.Dataset(tr_x, label=tr_y)
    dvalid = lgb.Dataset(va_x, label=va_y)

    clf = lgb.train(params, dtrain, 3000, valid_sets=[dtrain,dvalid], verbose_eval=100, early_stopping_rounds=100)

    feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()

    y_pred_valid = clf.predict(va_x)
    y_oof[va_idx] = y_pred_valid
    print(f"Fold {fold_n + 1} | MSE: {mean_squared_error(va_y, y_pred_valid)}")

    score += mean_squared_error(va_y, y_pred_valid) / NFOLDS
    y_preds += clf.predict(test_x) / NFOLDS

    del tr_x, tr_y, va_x, va_y
    gc.collect()

#print(f"\nMean Train MSE = {score}")
print(f"Out of folds MSE = {mean_squared_error(train_y, y_oof)}")

# make submission
sub['quality'] = y_preds
sub.to_csv("./submission.csv", index=False)

# plot feature importance
feature_importances['average'] = feature_importances[[f'fold_{fold_n + 1}' for fold_n in range(NFOLDS)]].mean(axis=1)
plt.figure(figsize=(16, 16))
sns.barplot(data=feature_importances.sort_values(by='average', ascending=False), x='average', y='feature');
plt.title('feature importance over {} folds average; score {}'.format(NFOLDS,mean_squared_error(train_y, y_oof)))
plt.savefig('./figs/feature_importance.png')
