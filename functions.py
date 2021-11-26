



def data_prep(data, max_lags=12, min_date='2007-01-01'):
  """add lagged features as columns

  Arguments
    data : dataframe with input data
    max_lags : int, maximum number of past/lagged values
    min_date : str, delete observations before this date


  """

  # remove rows with NA date
  data.dropna(axis=0, subset=['date'], inplace=True)
  # check 
  data.date = pd.to_datetime(data.date)
  # set identifiers
  data.set_index(["date", "mun_code"], inplace=True)
  
  # prep output with unlagged data:
  data_lags = data.copy()

  # add all of the lags
  for l in np.arange(1,max_lags):
      data_lags[data.columns+'_lag'+str(l)] = data.copy().unstack().shift(l).stack(dropna=False)
  # reset index    
  data_lags.reset_index(inplace=True)
  data_lags.sort_values("date", inplace=True)
  
  # remove lag0 vars
  data_lags.drop(data.columns[1:], axis=1, inplace=True)
  
  # remove rows with missing values
  data_lags.dropna(inplace=True)

  return data_lags




def compute_errors(y_pred, y_true, y_pred_real, y_true_real):
    
    """compute MAE and RMSE (normalized and real) 
    """
    
    # mae
    mae =  mean_absolute_error(y_true, y_pred)
    # rmse
    rmse = mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False)
    
    # mae real
    mae_real =  mean_absolute_error(y_true_real, y_pred_real)
    # rmse
    rmse_real = mean_squared_error(y_true=y_true_real, y_pred=y_pred_real, squared=False)
    
    return mae,rmse,mae_real,rmse_real



def run_models_allcities(data_train, models_dict, cv_indices):
    '''
    Run model for mutliple cities, including:
    - feature selection 
    - time series CV
    - run set of ML models in models_dict
    - compute errors on each CV split (for each city)

    Arguments
      data_train : dataframe, input data
      models_dict : dict, algorithms to use for training
      cv_indices : tran/val splits
    '''
    
    # prep output
    errors = pd.DataFrame()
    errors_cities = pd.DataFrame()
    predictions = pd.DataFrame()
    

    # check that data is sorted by date
    data_train.sort_values(by='date', inplace=True, ascending=True)


    ### CROSS-VALIDATION SPLITS
    # expanding time series CV
    cv_count=1
    for train_index, val_index in cv_indices:
        cv_train, cv_val = data_train.iloc[train_index,:], data_train.iloc[val_index,:]
        # separate date and mun_code columns
        cv_identifiers_val = cv_val.loc[:,['mun_code','date']]
        # remove them from data
        cv_train.drop(['mun_code','date'], axis=1, inplace=True)
        cv_val.drop(['mun_code','date'], axis=1, inplace=True)

        # normalize each cv fold
        mean = np.mean(cv_train,axis=0)
        std = np.std(cv_train,axis=0)
        cv_train_norm = (cv_train - mean)/std
        cv_val_norm = (cv_val - mean)/std


        cv_train_X = cv_train_norm.drop(['cases'],axis=1)
        cv_train_Y = cv_train_norm['cases']
        cv_val_X = cv_val_norm.drop(['cases'],axis=1)
        cv_val_Y = cv_val_norm['cases']



        ### MACHINE LEARNING MODELS
        for p in models_dict:
            pred_model = models_dict[p]
            # fit
            pred_model.fit(cv_train_X, cv_train_Y)

            # predict on val set
            prediction_val = pred_model.predict(cv_val_X)

            # output predictions
            predictions_model = pd.DataFrame({'predictions':prediction_val, 
                                             'actual':cv_val_Y})
            predictions_model['mun_code'] = cv_identifiers_val.loc[:,'mun_code'].values
            predictions_model['date'] = cv_identifiers_val.loc[:,'date'].values
            predictions_model['model'] = p
            predictions_model['cv_fold'] = cv_count

            # non-normalize
            predictions_real = prediction_val*std['cases'] + mean['cases']
            predictions_model['predictions_real'] = predictions_real
            predictions_model['actual_real'] = cv_val['cases']

            # append predictions
            predictions = predictions.append(predictions_model)

            ### COMPUTE ERRORS
            # compute errors
            mae_val, rmse_val, mae_val_real, rmse_val_real = compute_errors(y_pred = prediction_val, 
                                                             y_true = cv_val_Y, 
                                                             y_pred_real = predictions_real,
                                                             y_true_real = cv_val['cases']
                                                             )
            
            
            errors_model = pd.DataFrame({'mae_val': mae_val, 'rmse_val': rmse_val,
                                        'mae_val_real': mae_val_real, 'rmse_val_real': rmse_val_real},index=[cv_count])

            # append the errors
            errors_model['model'] = p
            errors_model['cv_fold'] = cv_count
            # append errors
            errors = errors.append(errors_model)


            # compute errors per city
            for mun_code in predictions_model.mun_code.unique():
                
                predictions_model_city = predictions_model[predictions_model.mun_code == mun_code]
                
                mae_val_city, rmse_val_city, mae_val_city_real, rmse_val_city_real = compute_errors(y_pred = predictions_model_city.predictions, 
                                                                                                     y_true = predictions_model_city.actual, 
                                                                                                     y_pred_real = predictions_model_city.predictions_real,
                                                                                                     y_true_real = predictions_model_city.actual_real
                                                                                                     )
            
                
                errors_model_city = pd.DataFrame({'mae_val': mae_val_city, 'rmse_val': rmse_val_city,
                                                 'mae_val_real':mae_val_city_real, 'rmse_val_real': rmse_val_city_real},index=[cv_count])
                errors_model_city['model'] = p
                errors_model_city['cv_fold'] = cv_count
                errors_model_city['mun_code'] = mun_code
                # append
                errors_cities = errors_cities.append(errors_model_city)
                
        cv_count+=1
            
    return errors, errors_cities, predictions





def test_models_allcities(data_train, data_test, models_dict):
    '''
    Train tuned model on train data and test on hold-out test set.
    
    
    Returns
    - aggregated errors over all cities
    - individual city errors
    - predictions for all cities
    '''
    
    # prep output
    errors = pd.DataFrame()
    errors_cities = pd.DataFrame()
    predictions = pd.DataFrame()
    
    # remove rows with missing values
    data_train.dropna(inplace=True)
    # remove columns with missing values
    data_train.dropna(inplace=True,axis=1)

    # sort data by date
    data_train.sort_values(by='date', inplace=True, ascending=True)



    identifiers_train = data_train.loc[:,['mun_code','date']]
    identifiers_test = data_test.loc[:,['mun_code','date']]

    # remove them from data
    data_train.drop(['mun_code','date'], axis=1, inplace=True)
    data_test.drop(['mun_code','date'], axis=1, inplace=True)

    # normalize
    mean = np.mean(data_train,axis=0)
    std = np.std(data_train,axis=0)
    data_train_norm = (data_train - mean)/std
    data_test_norm = (data_test - mean)/std

    train_X = data_train_norm.drop(['cases'],axis=1)
    train_Y = data_train_norm['cases']
    test_X = data_test_norm.drop(['cases'],axis=1)
    test_Y = data_test_norm['cases']



    ### MACHINE LEARNING MODELS
    for p in models_dict:
        pred_model = models_dict[p]
        # fit
        pred_model.fit(train_X, train_Y)

        # predict on val set
        prediction_test = pred_model.predict(test_X)

        # output predictions
        predictions_model = pd.DataFrame({'predictions':prediction_test, 
                                         'actual':test_Y})
        predictions_model['mun_code'] = identifiers_test.loc[:,'mun_code'].values
        predictions_model['date'] = identifiers_test.loc[:,'date'].values
        predictions_model['model'] = p

        # non-normalize
        predictions_real = prediction_test*std['cases'] + mean['cases']
        actual_real = test_Y*std['cases'] + mean['cases']
        predictions_model['predictions_real'] = predictions_real
        predictions_model['actual_real'] = actual_real

        # append predictions
        predictions = predictions.append(predictions_model)

        ### COMPUTE ERRORS
        mae_test, rmse_test, mae_test_real, rmse_test_real = compute_errors(y_pred = prediction_test, 
                                                             y_true = test_Y, 
                                                             y_pred_real = predictions_real,
                                                             y_true_real = data_test['cases']
                                                             )
            
        
        errors_model = pd.DataFrame({'mae_test': mae_test, 'rmse_test': rmse_test,
                                    'mae_test_real': mae_test_real, 'rmse_test_real': rmse_test_real},index=[0])

        # append the errors
        errors_model['model'] = p
        # append errors
        errors = errors.append(errors_model)


        # compute errors per city
        for mun_code in predictions_model.mun_code.unique():
            predictions_model_city = predictions_model[predictions_model.mun_code == mun_code]
            # errors
            mae_city, rmse_city, mae_city_real, rmse_city_real = compute_errors(y_pred = predictions_model_city.predictions, 
                                                             y_true = predictions_model_city.actual, 
                                                             y_pred_real = predictions_model_city.predictions_real,
                                                             y_true_real = predictions_model_city.actual_real
                                                             )
            
            errors_model_city = pd.DataFrame({'mae_test': mae_city, 'rmse_test': rmse_city,
                                             'mae_test_real':mae_city_real, 'rmse_test_real': rmse_city_real},index=[0])
            errors_model_city['model'] = p
            errors_model_city['mun_code'] = mun_code
            # append
            errors_cities = errors_cities.append(errors_model_city)


    return errors, errors_cities, predictions
