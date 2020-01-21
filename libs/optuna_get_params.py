def get_lgb_params():
    params = {'num_leaves': 128,
          'min_child_weight': 1,
          'bagging_fraction': 0.4181193142567742,
          'min_data_in_leaf': 300,
          'objective': 'regression',
          'max_depth': 5,
          'learning_rate': 0.1,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'rmse',
          "verbosity": -1,
          'random_state': 47,
          'feature_fraction': 0.8,
         }
    return params
