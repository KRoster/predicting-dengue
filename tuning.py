import pandas as pd
import numpy as np
import datetime
from pandas import Series
import itertools
from scipy import stats


# plotting
import matplotlib.pyplot as plt
import seaborn as sns
from palettable.cartocolors.qualitative import Vivid_4

# evaluation
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

# algorithms
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor



# time series cross-validation
n_splits = 8
tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=None)


###################################################################
### set up the regressors and parameter spaces
###################################################################


# MLP
mlp_reg = MLPRegressor(max_iter=500, solver='adam', alpha = 0.2, activation='relu')
# set up the parameter space
parameters_mlp = {'hidden_layer_sizes': [(64,64,64,), (64,64,), (64,),
                                        (32,32,32,), (32,32,), (32,),
                                        (128,128,128,), (128,128,), (128,),
                  ]}


# RF
rf_reg = RandomForestRegressor()
parameters_rf = {'n_estimators' : [50, 100, 150, 200],
                    'max_depth' : [2,3,6, None]}


# GBM
gbm_reg = GradientBoostingRegressor() 
parameters_gbm = {'n_estimators' : [50, 100, 150, 200],
                    'max_depth' : [2,3,6, None]}

# SVR
svm_reg = svm.SVR()
parameters_svm = {'epsilon' : [0.01, 0.05, 0.1, 0.2]}



###################################################################
### Grid Search
###################################################################


### only dengue data

# MLP
clf_mlp = GridSearchCV(estimator = mlp_reg, cv=tscv, param_grid=parameters_mlp)
search_result_mlp = clf_mlp.fit(train_x_onlyDengue, train_y)

# RF
clf_rf = GridSearchCV(estimator = rf_reg, cv=tscv, param_grid=parameters_rf)
search_result_rf = clf_rf.fit(train_x_onlyDengue, train_y)

# GBM
clf_gbm = GridSearchCV(estimator = gbm_reg, cv=tscv, param_grid=parameters_gbm)
search_result_gbm = clf_gbm.fit(train_x_onlyDengue, train_y)

# SVM
clf_svm = GridSearchCV(estimator = svm_reg, cv=tscv, param_grid=parameters_svm)
search_result_svm = clf_svm.fit(train_x_onlyDengue, train_y)


print("MLP Best Parameters: %s, with a score of: %f" %  ( search_result_mlp.best_params_, search_result_mlp.best_score_), "\n")
print("GBM Best Parameters: %s, with a score of: %f" %  ( search_result_gbm.best_params_, search_result_gbm.best_score_), "\n")
print("RF Best Parameters: %s, with a score of: %f" %  ( search_result_rf.best_params_, search_result_rf.best_score_), "\n")
print("SVM Best Parameters: %s, with a score of: %f" %  ( search_result_svm.best_params_, search_result_svm.best_score_), "\n")


### all variables

# MLP
clf_mlp2 = GridSearchCV(estimator = mlp_reg, cv=tscv, param_grid=parameters_mlp)
search_result_mlp2 = clf_mlp2.fit(train_x, train_y)

# RF
clf_rf2 = GridSearchCV(estimator = rf_reg, cv=tscv, param_grid=parameters_rf)
search_result_rf2 = clf_rf2.fit(train_x, train_y)

# GBM
clf_gbm2 = GridSearchCV(estimator = gbm_reg, cv=tscv, param_grid=parameters_gbm)
search_result_gbm2 = clf_gbm2.fit(train_x, train_y)

# SVM
clf_svm2 = GridSearchCV(estimator = svm_reg, cv=tscv, param_grid=parameters_svm)
search_result_svm2 = clf_svm2.fit(train_x, train_y)


print("MLP Best Parameters: %s, with a score of: %f" %  ( search_result_mlp2.best_params_, search_result_mlp2.best_score_), "\n")
print("GBM Best Parameters: %s, with a score of: %f" %  ( search_result_gbm2.best_params_, search_result_gbm2.best_score_), "\n")
print("RF Best Parameters: %s, with a score of: %f" %  ( search_result_rf2.best_params_, search_result_rf2.best_score_), "\n")
print("SVM Best Parameters: %s, with a score of: %f" %  ( search_result_svm2.best_params_, search_result_svm2.best_score_), "\n")



### PCMCI Parcorr

# MLP
clf_mlp3 = GridSearchCV(estimator = mlp_reg, cv=tscv, param_grid=parameters_mlp)
search_result_mlp3 = clf_mlp3.fit(train_x_parcorr, train_y)

# RF
clf_rf3 = GridSearchCV(estimator = rf_reg, cv=tscv, param_grid=parameters_rf)
search_result_rf3 = clf_rf3.fit(train_x_parcorr, train_y)

# GBM
clf_gbm3 = GridSearchCV(estimator = gbm_reg, cv=tscv, param_grid=parameters_gbm)
search_result_gbm3 = clf_gbm3.fit(train_x_parcorr, train_y)

# SVM
clf_svm3 = GridSearchCV(estimator = svm_reg, cv=tscv, param_grid=parameters_svm)
search_result_svm3 = clf_svm3.fit(train_x_parcorr, train_y)


print("MLP Best Parameters: %s, with a score of: %f" %  ( search_result_mlp3.best_params_, search_result_mlp3.best_score_), "\n")
print("GBM Best Parameters: %s, with a score of: %f" %  ( search_result_gbm3.best_params_, search_result_gbm3.best_score_), "\n")
print("RF Best Parameters: %s, with a score of: %f" %  ( search_result_rf3.best_params_, search_result_rf3.best_score_), "\n")
print("SVM Best Parameters: %s, with a score of: %f" %  ( search_result_svm3.best_params_, search_result_svm3.best_score_), "\n")



### Correlation

# MLP
clf_mlp4 = GridSearchCV(estimator = mlp_reg, cv=tscv, param_grid=parameters_mlp)
search_result_mlp4 = clf_mlp4.fit(train_x_corr, train_y)

# RF
clf_rf4 = GridSearchCV(estimator = rf_reg, cv=tscv, param_grid=parameters_rf)
search_result_rf4 = clf_rf4.fit(train_x_corr, train_y)

# GBM
clf_gbm4 = GridSearchCV(estimator = gbm_reg, cv=tscv, param_grid=parameters_gbm)
search_result_gbm4 = clf_gbm4.fit(train_x_corr, train_y)

# SVM
clf_svm4 = GridSearchCV(estimator = svm_reg, cv=tscv, param_grid=parameters_svm)
search_result_svm4 = clf_svm4.fit(train_x_corr, train_y)


print("MLP Best Parameters: %s, with a score of: %f" %  ( search_result_mlp4.best_params_, search_result_mlp4.best_score_), "\n")
print("GBM Best Parameters: %s, with a score of: %f" %  ( search_result_gbm4.best_params_, search_result_gbm4.best_score_), "\n")
print("RF Best Parameters: %s, with a score of: %f" %  ( search_result_rf4.best_params_, search_result_rf4.best_score_), "\n")
print("SVM Best Parameters: %s, with a score of: %f" %  ( search_result_svm4.best_params_, search_result_svm4.best_score_), "\n")




