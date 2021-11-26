"""
Predicting dengue in Brazilian cities

Author: Kirstin Roster

"""


import pandas as pd
import numpy as np
import itertools

# plotting
import matplotlib.pyplot as plt
from matplotlib.container import BarContainer
from palettable.cartocolors.qualitative import Vivid_4

# evaluation
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

# algorithms
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

from functions import *


plt.style.use('seaborn-darkgrid')


###################################################################
### Data Prep
###################################################################


# read data
# contains location and time identifier (mun_code and date) and all input features
data = pd.read_csv("data.csv", parse_dates=['date'])

data_lags = data_prep(data, max_lags=12, min_date='2007-01-01')

# train / test split
data_train = data_lags[data_lags.date<'2016-01-01']
data_test = data_lags[data_lags.date>='2016-01-01']



# feature selection
features_onlyDengue = [s for s in data_train.columns if "cases_" in s]
features_pcmci = ['cases_lag1', 'cases_lag2', 'cases_lag3', 'cases_lag4', 'cases_lag5', 'cases_lag11', #autocorr vars
                  'Precipitacao_mean_lag4']
features_corr = ['cases_lag1', 'cases_lag2', 'cases_lag11', 'cases_lag3',
                'cases_lag10', 'cases_lag9', 'cases_lag4']



###################################################################
### Run Models - Val Set
###################################################################


# prepare the time series splits (use the same for all the models)
n_splits = 8
tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=None)
cv_indices = tscv.split(data_train)


# models with optimal hyperparameters

models_dict_onlyDengue = {'rf' : RandomForestRegressor(max_depth = None, n_estimators =50), 
                          'svr_rbf' : svm.SVR(epsilon = 0.05),
                          'MLP' : MLPRegressor(max_iter=300, solver='adam', alpha = 0.2, activation='relu',
                                              hidden_layer_sizes = (32, 32)),
                          'GBM' : GradientBoostingRegressor(max_depth = 2, n_estimators = 200)
                         }

models_dict_pcmci = {'rf' : RandomForestRegressor(max_depth = 6, n_estimators =100), 
                          'svr_rbf' : svm.SVR(epsilon = 0.05),
                          'MLP' : MLPRegressor(max_iter=300, solver='adam', alpha = 0.2, activation='relu',
                                              hidden_layer_sizes = (64,)),
                          'GBM' : GradientBoostingRegressor(max_depth = 2, n_estimators = 200)
                         }

models_dict_corr = {'rf' : RandomForestRegressor(max_depth = None, n_estimators =50), 
                          'svr_rbf' : svm.SVR(epsilon = 0.05),
                          'MLP' : MLPRegressor(max_iter=300, solver='adam', alpha = 0.2, activation='relu',
                                              hidden_layer_sizes = (32,)),
                          'GBM' : GradientBoostingRegressor(max_depth = 2, n_estimators = 200)
                         }

models_dict_withClimate = {'rf' : RandomForestRegressor(max_depth = 6, n_estimators =200), 
                          'svr_rbf' : svm.SVR(epsilon = 0.01),
                          'MLP' : MLPRegressor(max_iter=300, solver='adam', alpha = 0.2, activation='relu',
                                              hidden_layer_sizes = (32, 32)),
                          'GBM' : GradientBoostingRegressor(max_depth = 2, n_estimators = 200)
                         }

# run models
errors_onlyDengue, errors_cities_onlyDengue, predictions_onlyDengue = run_models_allcities(data_train.loc[:,['date','mun_code', 'cases']+features_onlyDengue], models_dict_onlyDengue, tscv.split(data_train))
errors_pcmci, errors_cities_pcmci, predictions_pcmci = run_models_allcities(data_train.loc[:,['date','mun_code', 'cases']+features_pcmci], models_dict_pcmci, tscv.split(data_train))
errors_corr, errors_cities_corr, predictions_corr = run_models_allcities(data_train.loc[:,['date','mun_code', 'cases']+features_corr], models_dict_corr, tscv.split(data_train))
errors_withClimate, errors_cities_withClimate, predictions_withClimate = run_models_allcities(data_train, models_dict_withClimate, tscv.split(data_train))


# combine the errors
errors_onlyDengue['fs'] = 'only Dengue'
errors_pcmci['fs'] = 'PCMCI'
errors_corr['fs'] = 'Correlation'
errors_withClimate['fs'] = 'with Climate'

errors_combo = pd.concat([errors_onlyDengue, errors_pcmci, 
                          errors_corr, errors_withClimate], axis=0)

errors_cities_onlyDengue['fs'] = 'only Dengue'
errors_cities_pcmci['fs'] = 'PCMCI'
errors_cities_corr['fs'] = 'Correlation'
errors_cities_withClimate['fs'] = 'with Climate'

errors_cities_combo = pd.concat([errors_cities_onlyDengue, errors_cities_pcmci, errors_cities_corr, 
                                 errors_cities_withClimate], axis=0)

# aggregate over the cv folds
errors_cities_cvagg = errors_cities_combo.drop('cv_fold', axis=1).groupby(['mun_code', 'model', 'fs']).agg('mean')
errors_cities_cvagg.reset_index(inplace=True)



# output table
table_cities = errors_cities_cvagg.drop('mun_code',axis=1).groupby(['model','fs']).agg(['mean', 'median'])

with pd.option_context('display.precision', 3):
    table_cities_style = (table_cities.style.highlight_min(axis=0))
    
table_cities_style


###################################################################
### Model Selection
###################################################################

best_model_percity_mae= errors_cities_cvagg.loc[errors_cities_cvagg.groupby(['mun_code'])['mae_val_real'].idxmin()]
best_model_percity_mae.drop(['mae_val', 'rmse_val', 'mae_val_real', 'rmse_val_real'], axis=1).groupby(['model','fs']).agg('count').plot(kind='barh', xlabel='')





###################################################################
### Test Set Predictions
###################################################################




# get predictions

errors_test_onlyDengue, errors_cities_test_onlyDengue, predictions_test_onlyDengue = test_models_allcities(data_train.loc[:,['date','mun_code', 'cases']+features_onlyDengue], data_test.loc[:,['date','mun_code', 'cases']+features_onlyDengue], models_dict_onlyDengue)
errors_test_pcmci, errors_cities_test_pcmci, predictions_test_pcmci = test_models_allcities(data_train.loc[:,['date','mun_code', 'cases']+features_pcmci], data_test.loc[:,['date','mun_code', 'cases']+features_pcmci], models_dict_pcmci)
errors_test_corr, errors_cities_test_corr, predictions_test_corr = test_models_allcities(data_train.loc[:,['date','mun_code', 'cases']+features_corr], data_test.loc[:,['date','mun_code', 'cases']+features_corr], models_dict_corr)
errors_test_withClimate, errors_cities_test_withClimate, predictions_test_withClimate = test_models_allcities(data_train, data_test, models_dict_withClimate)


# combine the errors
errors_test_onlyDengue['fs'] = 'only Dengue'
errors_test_pcmci['fs'] = 'PCMCI'
errors_test_corr['fs'] = 'Correlation'
errors_test_withClimate['fs'] = 'with Climate'

errors_test_combo = pd.concat([errors_test_onlyDengue, errors_test_pcmci, errors_test_corr, 
                               errors_test_withClimate], axis=0)

errors_cities_test_onlyDengue['fs'] = 'only Dengue'
errors_cities_test_pcmci['fs'] = 'PCMCI'
errors_cities_test_corr['fs'] = 'Correlation'
errors_cities_test_withClimate['fs'] = 'with Climate'

errors_cities_test_combo = pd.concat([errors_cities_test_onlyDengue, errors_cities_test_pcmci, errors_cities_test_corr, 
                                 errors_cities_test_withClimate], axis=0)

# output the error tables
round(errors_test_combo,3)

# city-level errors
round(errors_cities_test_combo.drop('mun_code', axis=1).groupby(['fs','model']).agg(['mean', 'median']), 3)


### city-specific model
# for each city, we choose the best algorithm AND FS type
# apply that algo for that city during test set prediction 

best_model_testerrors_mae = errors_cities_test_combo[errors_cities_test_combo[['mun_code','model','fs']].apply(tuple,1).isin(best_model_percity_mae[['mun_code','model','fs']].apply(tuple,1))]

best_model_testerrors_mae_table = best_model_testerrors_mae.drop(['mun_code', 'model', 'fs'],axis=1).agg(['mean','median'])
round(best_model_testerrors_mae_table,3)






###################################################################
### Plots
###################################################################

### Bar plot: best model selection across cities in val set

best_model_pivot = best_model_percity_mae.drop(['mun_code', 'mae_val', 'rmse_val_real', 'mae_val_real'],axis=1).groupby(['fs','model']).agg('count')
best_model_pivot = best_model_pivot.unstack(level=-1).T
best_model_pivot.index = best_model_pivot.index.droplevel()
best_model_pivot.index = ['GBM', 'MLP', 'RF', 'SVR']

ax = best_model_pivot.plot(kind='bar', stacked=True, rot=0, xlabel='', colormap=Vivid_4.mpl_colormap,
                          figsize=(10,8), fontsize=16)
bars = [thing for thing in ax.containers if isinstance(thing, BarContainer)]

patterns = itertools.cycle(('|','|','|','|', 
                            'x','x','x','x', 
                            '.','.','.','.',
                            '\\','\\','\\','\\'))
for bar in bars:
    for patch in bar:
        patch.set_hatch(next(patterns))
L = ax.legend(prop={'size': 16})




### Lineplot: Predictions vs true cases

# combine predictions
predictions_test_withClimate['fs'] = 'with Climate'
predictions_test_onlyDengue['fs'] = 'only Dengue'
predictions_test_pcmci['fs'] = 'PCMCI' 
predictions_test_corr['fs'] = 'Correlation'

predictions_test_combined = pd.concat([predictions_test_withClimate, predictions_test_onlyDengue, 
                                       predictions_test_pcmci, predictions_test_corr ], axis=0) 

# select predictions from best model for each city only
best_model_predictions_mae = predictions_test_combined[predictions_test_combined[['mun_code','model','fs']].apply(tuple,1).isin(best_model_percity_mae[['mun_code','model','fs']].apply(tuple,1))]


# combine validation and test sets
predictions_onlyDengue['fs'] = 'only Dengue'
predictions_pcmci['fs'] = 'PCMCI'
predictions_corr['fs'] = 'Correlation'
predictions_withClimate['fs'] = 'with Climate'

# combine all the val set predictions
predictions_val_combo = pd.concat([predictions_withClimate, predictions_onlyDengue, 
                                   predictions_pcmci, predictions_corr], axis=0) 
# combine test and val sets
predictions_valtest = pd.concat([predictions_val_combo, predictions_test_combined], axis=0)

# correct the date value format
predictions_valtest.date = pd.to_datetime(predictions_valtest.date)




use_cities_dict = {'Belém' : 150140., 
                   'São Paulo' : 355030., 
                   'Sorocaba': 355220.
                  }


# subset the predictions, selecting only the best kind
predictions1 = predictions_valtest[predictions_valtest.model=='rf']
predictions1 = predictions1[predictions1.fs=='onlyDengue']

ncols=2
nrows=len(use_cities_dict)
plt.figure(figsize=(5*ncols,5*nrows))

for i,mun_name in enumerate(use_cities_dict.keys()):
    mun_code = use_cities_dict[mun_name]
    # subset city data
    predictions_city1 = predictions1[predictions1.mun_code == mun_code]

    min_val = min(predictions_city1.predictions_real)
    max_val = max(predictions_city1.predictions_real.append(predictions_city1.actual_real))
    ax = plt.subplot(nrows,ncols,2*i+1)
    predictions_city1.plot(ax=ax, x='date', y='actual_real', 
                           color='black', label = 'Actual')
    predictions_city1.plot(ax=ax, x='date', y= 'predictions_real', color = Vivid_4.mpl_colors[2], label = 'Predicted')
    plt.axvspan('1-1-2016', '2019-12-31', facecolor='yellow', alpha=0.2)
    i+=1
    
    plt.text(s='Test Set', x='6-1-2016', y=max_val, color = 'gray', fontweight='bold')
    
    plt.xlabel("")
    plt.title(mun_name, fontsize=20)
    
    # ----- scatter -----------
    
    ax = plt.subplot(nrows,ncols,2*i+2)
    
    scatter_data1 =  predictions1[predictions1.date<'1-1-2016']
    scatter_data2 =  predictions1[predictions1.date>='1-1-2016']

    ax.scatter(data=scatter_data1[scatter_data1.mun_code==mun_code], 
           x='predictions_real', y='actual_real', 
           s=25, color=Vivid_4.mpl_colors[2],alpha=0.8, label='validation')
    ax.scatter(data=scatter_data2[scatter_data2.mun_code==mun_code], 
           x='predictions_real', y='actual_real', 
           s=25, color='gold', alpha = 0.9, label='test')
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
           ]
    # now plot both limits against eachother
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)

    i+=1
    plt.xlabel('Predicted', fontsize=15)
    plt.ylabel('Actual', fontsize=15)
    plt.legend()
    plt.title(mun_name, fontsize=20)
    
plt.tight_layout()    
       



