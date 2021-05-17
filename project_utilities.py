# -*- coding: utf-8 -*-
"""
Created on Fri May 14 23:42:52 2021

@author: aidan
"""

import pandas as pd
import numpy as np
import pycountry_convert as pc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV


# function from https://stackoverflow.com/questions/55910004/get-continent-name-from-country-using-pycountry
# input: country name
# output: continent name
def country_to_continent(country_name):
    country_alpha2 = pc.country_name_to_country_alpha2(country_name)
    country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
    country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
    return country_continent_name

# this funtion cleans a single dataset by dropping columns/missing values, 
# mapping the nationality to a continent, label encoding the work rate, 
# and performing one hot encoding for categorical variables
# input: a dataframes for a year of data
# include_gk tells whether or not to include GK in the analysis
# output: scaled X dataset containing cleaned data
def single_set_data_cleaning(df, include_gk = False):
    x = df.drop(['wage_eur', 'sofifa_id', 'mentality_composure', 'player_url', 'club_name', 'short_name', 'long_name', 'value_eur', 'dob', 'player_positions', 'real_face', 'release_clause_eur', 
                 'player_tags', 'team_jersey_number', 'loaned_from', 'joined', 'contract_valid_until',
                 'nation_position', 'nation_jersey_number', 'player_traits', 'ls', 'st', 'rs', 'lw', 
                 'lf', 'cf', 'rf', 'rw', 'lam', 'cam', 'ram', 'lm', 'lcm', 'cm', 'rcm', 'rm', 'lwb', 'ldm',
                 'cdm', 'rdm', 'rwb', 'lb', 'lcb', 'cb', 'rcb', 'rb'], axis = 1)
    y = df['value_eur']
    if not include_gk:
        x = x.drop(['gk_diving', 'gk_handling', 'gk_kicking', 'gk_reflexes',
                'gk_speed', 'gk_positioning', 'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes'], axis = 1)
        x = x[x['team_position'] != 'GK']
        y = y[df['team_position'] != 'GK']
        x = x[~x['pace'].isna()]
        y = y[~df['pace'].isna()]
    else:
        # the non-goalkeeper players have missing goalkeeper stats
        # the goalkeepers have missing player stats
        for col in ['pace', 'shooting', 'passing', 'dribbling', 'defending','physic', 'gk_diving', 'gk_handling', 'gk_kicking', 'gk_reflexes','gk_speed', 'gk_positioning']:
            x[col] = x[col].fillna(0)
    
    work_rate_map = {'Low': 1, 'Medium': 2, 'High': 3}
    x['att_work_rate']= x['work_rate'].apply(lambda x: x.split('/')[0]).replace(work_rate_map)
    x['def_work_rate'] = x['work_rate'].apply(lambda x : x.split('/')[1]).replace(work_rate_map)
    x = x.drop(['work_rate'], axis = 1)
    name_map = {'Wales':'United Kingdom', 'England': 'United Kingdom', 'Bosnia Herzegovina': 'Bosnia and Herzegovina',
            'Republic of Ireland': 'Ireland','Northern Ireland': 'Ireland', 'Korea Republic': 'South Korea',
            'DR Congo': 'Congo', 'Scotland': 'United Kingdom','Guinea Bissau': 'Guinea-Bissau', 
            'Trinidad & Tobago': 'Trinidad and Tobago', 'Kosovo': 'Turkey', 'Curacao':'Netherlands', 
            'Korea DPR': 'North Korea', 'Antigua & Barbuda':'Mexico', 'China PR': 'China', 'São Tomé & Príncipe':'Ghana', 'Chinese Taipei': 'China'}

    x['continent_nationality'] = x['nationality'].replace(name_map).apply(country_to_continent)
    x = x.drop(['nationality'], axis = 1)
    # fill missing league values with 4 (lowest league)
    # this is probably the case for teams that aren't in any popular leagues
    x['league_rank'] = x['league_rank'].fillna(4)
    
    # perform one-hot encoding on remaining categorical variables
    return pd.get_dummies(x), y


# input: 2 dataframes for 2 different years of data (put earlier year first)
# include_gk tells whether or not to include GK in the analysis
# output: scaled X dataset containing cleaned data ready for models and y (relative percent change in value between 2 years)
def horizon_data_prep(df1, df2, include_gk = False):
    df = pd.merge(df1, df2[['sofifa_id', 'value_eur']], how = 'inner', on = 'sofifa_id')
    df = df[(df['value_eur_x'] != 0) & (df['value_eur_y'] != 0)]
    y = (df['value_eur_y'] - df['value_eur_x'])/df['value_eur_x']
    x = df.drop(['value_eur_x', 'value_eur_y', 'wage_eur', 'mentality_composure', 'player_url', 
                 'club_name', 'short_name', 'long_name', 'dob', 'player_positions', 'real_face', 'release_clause_eur', 
                 'player_tags', 'team_jersey_number', 'loaned_from', 'joined', 'contract_valid_until',
                 'nation_position', 'nation_jersey_number', 'player_traits', 'ls', 'st', 'rs', 'lw', 
                 'lf', 'cf', 'rf', 'rw', 'lam', 'cam', 'ram', 'lm', 'lcm', 'cm', 'rcm', 'rm', 'lwb', 'ldm',
                 'cdm', 'rdm', 'rwb', 'lb', 'lcb', 'cb', 'rcb', 'rb', 'league_name', 'body_type'], axis = 1)
    if not include_gk:
        x = x.drop(['gk_diving', 'gk_handling', 'gk_kicking', 'gk_reflexes',
                'gk_speed', 'gk_positioning', 'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes'], axis = 1)
        x = x[x['team_position'] != 'GK']
        y = y[df['team_position'] != 'GK']
        x = x[~x['pace'].isna()]
        y = y[~df['pace'].isna()]
    else:
        # the non-goalkeeper players have missing goalkeeper stats
        # the goalkeepers have missing player stats
        for col in ['pace', 'shooting', 'passing', 'dribbling', 'defending',
                'physic', 'gk_diving', 'gk_handling', 'gk_kicking', 'gk_reflexes',
                'gk_speed', 'gk_positioning']:
            x[col] = x[col].fillna(0)
    work_rate_map = {'Low': 1, 'Medium': 2, 'High': 3}
    x['att_work_rate']= x['work_rate'].apply(lambda x: x.split('/')[0]).replace(work_rate_map)
    x['def_work_rate'] = x['work_rate'].apply(lambda x : x.split('/')[1]).replace(work_rate_map)
    x = x.drop(['work_rate'], axis = 1)
    name_map = {'Wales':'United Kingdom', 'England': 'United Kingdom', 'Bosnia Herzegovina': 'Bosnia and Herzegovina',
            'Republic of Ireland': 'Ireland','Northern Ireland': 'Ireland', 'Korea Republic': 'South Korea',
            'DR Congo': 'Congo', 'Scotland': 'United Kingdom','Guinea Bissau': 'Guinea-Bissau', 
            'Trinidad & Tobago': 'Trinidad and Tobago', 'Kosovo': 'Turkey', 'Curacao':'Netherlands', 
            'Korea DPR': 'North Korea', 'Antigua & Barbuda':'Mexico', 'China PR': 'China', 'São Tomé & Príncipe':'Ghana', 'Chinese Taipei': 'China'}

    x['continent_nationality'] = x['nationality'].replace(name_map).apply(country_to_continent)
    x = x.drop(['nationality'], axis = 1)
    # fill missing league values with 4 (lowest league)
    # this is probably the case for teams that aren't in any popular leagues
    x['league_rank'] = x['league_rank'].fillna(4)
    # perform one-hot encoding on remaining categorical variables
    x = pd.get_dummies(x)
    sofifa_id = x['sofifa_id']
    x = x.drop(['sofifa_id'], axis = 1)
    cols = x.columns
    scaler = StandardScaler()
    x[cols] = scaler.fit_transform(x[cols])
    x['sofifa_id'] = sofifa_id
    return x, y, cols

# takes in a model object and a param_grid
# returns an array of optimal model and the optimal cross validated MAE for each time horizon
def get_model_results(model, param_grid, x, y):
    horizon_scores = []
    horizon_models = []
    for horizon in range(6):
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        reg = GridSearchCV(model, param_grid, cv =kfold, scoring = "neg_mean_absolute_error", n_jobs = -1)
        search = reg.fit(x[horizon].drop(['sofifa_id'], axis = 1), y[horizon])
        horizon_scores.append(search.best_score_)
        horizon_models.append(search.best_estimator_)
    return horizon_models, horizon_scores

# takes in a model object and a param_grid
# returns an array of optimal model and the optimal cross validated MAE for each time horizon
def get_model_results_pca(model, param_grid, x, y):
    horizon_scores = []
    horizon_models = []
    for horizon in range(6):
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        reg = GridSearchCV(model, param_grid, cv =kfold, scoring = "neg_mean_absolute_error", n_jobs = -1)
        search = reg.fit(pca.fit_transform(x[horizon].drop(['sofifa_id'], axis = 1)), y[horizon])
        horizon_scores.append(search.best_score_)
        horizon_models.append(search.best_estimator_)
    return horizon_models, horizon_scores

# takes in a model object and a param_grid
# returns an array of optimal model and the optimal cross validated MAE for each time horizon
# this function will only fit the model on variables chosen by LASSO
def get_model_results_lasso(model, param_grid,x, y):
    horizon_scores = []
    horizon_models = []
    for horizon in range(6):
        lasso = LassoCV(random_state = 42, max_iter = 10000).fit(x[horizon].drop(['sofifa_id'], axis = 1), y[horizon])
        lasso_var_mask = np.where(lasso.coef_ != 0)
        mask_cols = x[horizon].drop(['sofifa_id'], axis = 1).columns[lasso_var_mask]
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        reg = GridSearchCV(model, param_grid, cv =kfold, scoring = "neg_mean_absolute_error", n_jobs = -1)
        search = reg.fit(x[horizon].drop(['sofifa_id'], axis = 1)[mask_cols], y[horizon])
        horizon_scores.append(search.best_score_)
        horizon_models.append(search.best_estimator_)
    return horizon_models, horizon_scores

# function for getting the cross-validated performance metrics for a regression model
def get_performance_metrics(model, x, y):
  metrics_dict = {}
  metrics_dict['Mean_Absolute_Error'] = (cross_val_score(model, x, y, scoring = 'neg_mean_absolute_error', cv =5, n_jobs = -1).mean())*-1
  metrics_dict['Root_Mean_Squared_Error'] = (cross_val_score(model, x, y, scoring = 'neg_root_mean_squared_error', cv =5, n_jobs = -1).mean())*-1
  metrics_dict['R_Squared'] = cross_val_score(model, x, y, scoring = 'r2', cv =5, n_jobs = -1).mean()
  return metrics_dict

# 2021 data_prep
def data_cleaning_test(df_2021, include_gk = False):
    x = df_2021.drop(['value_eur', 'wage_eur', 'mentality_composure', 'player_url', 
                 'club_name', 'short_name', 'long_name', 'dob', 'player_positions', 'real_face', 'release_clause_eur', 
                 'player_tags', 'team_jersey_number', 'loaned_from', 'joined', 'contract_valid_until',
                 'nation_position', 'nation_jersey_number', 'player_traits', 'ls', 'st', 'rs', 'lw', 
                 'lf', 'cf', 'rf', 'rw', 'lam', 'cam', 'ram', 'lm', 'lcm', 'cm', 'rcm', 'rm', 'lwb', 'ldm',
                 'cdm', 'rdm', 'rwb', 'lb', 'lcb', 'cb', 'rcb', 'rb', 'body_type', 'league_name'], axis = 1)
    if not include_gk:
        x = x.drop(['gk_diving', 'gk_handling', 'gk_kicking', 'gk_reflexes',
                'gk_speed', 'gk_positioning', 'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes'], axis = 1)
        x = x[x['team_position'] != 'GK']
        x = x[~x['pace'].isna()]
    else:
        # the non-goalkeeper players have missing goalkeeper stats
        # the goalkeepers have missing player stats
        for col in ['pace', 'shooting', 'passing', 'dribbling', 'defending',
                'physic', 'gk_diving', 'gk_handling', 'gk_kicking', 'gk_reflexes',
                'gk_speed', 'gk_positioning']:
            x[col] = x[col].fillna(0)
    work_rate_map = {'Low': 1, 'Medium': 2, 'High': 3}
    x['att_work_rate']= x['work_rate'].apply(lambda x: x.split('/')[0]).replace(work_rate_map)
    x['def_work_rate'] = x['work_rate'].apply(lambda x : x.split('/')[1]).replace(work_rate_map)
    x = x.drop(['work_rate'], axis = 1)
    name_map = {'Wales':'United Kingdom', 'England': 'United Kingdom', 'Bosnia Herzegovina': 'Bosnia and Herzegovina',
            'Republic of Ireland': 'Ireland','Northern Ireland': 'Ireland', 'Korea Republic': 'South Korea',
            'DR Congo': 'Congo', 'Scotland': 'United Kingdom','Guinea Bissau': 'Guinea-Bissau', 
            'Trinidad & Tobago': 'Trinidad and Tobago', 'Kosovo': 'Turkey', 'Curacao':'Netherlands', 
            'Korea DPR': 'North Korea', 'Antigua & Barbuda':'Mexico', 'China PR': 'China', 'São Tomé & Príncipe':'Ghana', 'Chinese Taipei': 'China'}

    x['continent_nationality'] = x['nationality'].replace(name_map).apply(country_to_continent)
    x = x.drop(['nationality'], axis = 1)
    # fill missing league values with 4 (lowest league)
    # this is probably the case for teams that aren't in any popular leagues
    x['league_rank'] = x['league_rank'].fillna(4)
    # perform one-hot encoding on remaining categorical variables
    x = pd.get_dummies(x)
    sofifa_id = x['sofifa_id']
    x = x.drop(['sofifa_id'], axis = 1)
    cols = x.columns
    scaler = StandardScaler()
    x[cols] = scaler.fit_transform(x[cols])
    x['sofifa_id'] = sofifa_id
    return x, cols


## Optimal params
    
# [XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=1, gamma=0,
#               importance_type='gain', learning_rate=0.05, max_delta_step=0,
#               max_depth=3, min_child_weight=1, missing=None, n_estimators=30,
#               n_jobs=1, nthread=None, objective='reg:squarederror',
#               random_state=42, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
#               seed=None, silent=None, subsample=1, verbosity=1),
#  XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=1, gamma=0,
#               importance_type='gain', learning_rate=0.05, max_delta_step=0,
#               max_depth=3, min_child_weight=1, missing=None, n_estimators=30,
#               n_jobs=1, nthread=None, objective='reg:squarederror',
#               random_state=42, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
#               seed=None, silent=None, subsample=1, verbosity=1),
#  XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=1, gamma=0,
#               importance_type='gain', learning_rate=0.05, max_delta_step=0,
#               max_depth=3, min_child_weight=1, missing=None, n_estimators=30,
#               n_jobs=1, nthread=None, objective='reg:squarederror',
#               random_state=42, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
#               seed=None, silent=None, subsample=1, verbosity=1),
#  XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=1, gamma=0,
#               importance_type='gain', learning_rate=0.05, max_delta_step=0,
#               max_depth=3, min_child_weight=1, missing=None, n_estimators=30,
#               n_jobs=1, nthread=None, objective='reg:squarederror',
#               random_state=42, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
#               seed=None, silent=None, subsample=1, verbosity=1),
#  XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=1, gamma=0,
#               importance_type='gain', learning_rate=0.05, max_delta_step=0,
#               max_depth=3, min_child_weight=1, missing=None, n_estimators=30,
#               n_jobs=1, nthread=None, objective='reg:squarederror',
#               random_state=42, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
#               seed=None, silent=None, subsample=1, verbosity=1),
#  XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=1, gamma=0,
#               importance_type='gain', learning_rate=0.05, max_delta_step=0,
#               max_depth=3, min_child_weight=1, missing=None, n_estimators=30,
#               n_jobs=1, nthread=None, objective='reg:squarederror',
#               random_state=42, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
#               seed=None, silent=None, subsample=1, verbosity=1)]