
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from lag_selector import select_best_lags
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.decomposition import PCA
# import shap
import hyperopt
from hyperopt import tpe, fmin, hp, space_eval, Trials
from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import sample
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.evaluate import PredefinedHoldoutSplit
# import joblib

#
#
# Preparando data para optimización
#
#

def create_target(dataset, target_as_string, timesteps = 1):
    """
    Esta función desfasa la variable target en n timesteps.
    """
    dataset['Target'] = dataset.loc[:, target_as_string].shift(-timesteps)
    cols = dataset.columns.values.tolist()
    cols = cols[-1:] + cols[:-1]
    dataset = dataset[cols]
    return dataset.fillna(0)

# Definir la ruta, nombre y hoja del archivo que se va a importar
file_name = "BD - PBIMensual"
main_path = ("Z:\\SIM Peru\\Research\\Top Down\\Perú\\PBI Mensual")
file_path = (file_name + ".xlsx")
sheet_name = "PythonYoY"

percent_train = 0.90
max_evals = 200 #Número de iteraciones que hará hyperopt para hallar el modelo óptimo
eval_metric = 'mse'
cross_validate = False #Seleccionar si se desea, o no, usar validación cruzada para el entrenamiento del modelo
metrics_names = {'mse': 'neg_mean_squared_error', 'r2': 'r2'}

dataset = pd.read_excel(main_path + "\\" + file_path, sheet_name, index_col = 'Date', parse_dates = True) 
# Prueba de nuevo modelo!
dataset = dataset.loc['31/01/2017':, :] #El modelo se entrenó a partir del 15/01/2010, por eso se selecciona el df a partir de esa fecha.

# Para modelos de PIB/IGAE/IMACEC
# max_missing = 12
# short_series = dataset.isna().sum()
# short_series = short_series[short_series < max_missing + 1]
# dataset = dataset[list(short_series.index)]
# new_obs = dataset.tail(1)
# dataset.drop(dataset.tail(1).index, inplace=True)

# Para modelos de Inflación
dataset = dataset.dropna()
dataset = create_target(dataset, 'PRGDP Index', 1)
# dataset = create_target(dataset, 'PRGOGGDP Index', 1)
# Target series:
# PRGDP Index: nsa
# PRGOGGDP Index: sa
del dataset['PRGDP Index']
del dataset['PRGOGGDP Index']
dataset = select_best_lags(dataset, dataset.iloc[:int(len(dataset)*percent_train)], 3, replicate = True, n_replicas = 3) #Con esta función se seleccionan los lags con mayor poder predictivo sobre la variable target.
# dataset['Month'] = dataset.index.month

# Separando validation dataset
length_opt = int(len(dataset)*percent_train)
total_obs = len(dataset)
dataset = dataset[:int(length_opt-total_obs)]

colnames = dataset.columns.values.tolist()
target = colnames[0]
predictors = colnames[1:]
n_features = dataset.shape[1]

length_train = int(length_opt*percent_train)
# length_train = length_opt
print('Total de observaciones :' + str(int(total_obs)))
print('Observaciones de training: ' + str(length_train))
print('Observaciones de testing: ' + str(length_opt - length_train))
print('Observaciones para validación posterior: ' + str(int(total_obs*(1-percent_train))))

# !!!! Restaurar
# Preparando dataset para entrenamiento, conservando ds_total
ds_total = dataset
dataset = dataset.iloc[:length_opt]

# percent_train = percent_train*0.80

# En caso de haber usado mse como métrica de evaluación y no querer usar validación cruzada, se define la variable piter que almacena la parte del índice del df donde se deberá
# testear el modelo. Además, en la función objective_function, se debe asgignar, en la variable reg, piter a cv.
if eval_metric == 'mse' and cross_validate == False:
    dataset.reset_index(inplace = True, drop = True)
    validation_indices = dataset.iloc[length_train:].index.values
    piter = PredefinedHoldoutSplit(validation_indices)

# Data preprocessing functions

def split_train_test(dataset, length_train):
    """
    Particiona el df en training y testing sin segmentar por variables dependientes e independientes.
    """
    train_set = dataset.iloc[:length_train]
    test_set = dataset.iloc[length_train:]
    return train_set, test_set

def no_transform(dataset, predictors, target):
    """
    Particiona el df en variables independientes y dependiente
    """
    X = dataset[predictors]
    Y = dataset[target]
    return X, Y

def transform(dataset, predictors, target, length_train):
    """
    Particiona el df en training y testing, aplica MinMaxScaler y retorna targets y predictores
    """
    train_set, test_set = split_train_test(dataset, length_train)
    scaler = MinMaxScaler(feature_range = (0,1))
    scaler.fit(train_set)
    scaled_train_set = pd.DataFrame(scaler.transform(train_set), columns = dataset.columns)
    scaled_test_set = pd.DataFrame(scaler.transform(test_set), columns = dataset.columns)
    scaled_df = pd.concat([scaled_train_set, scaled_test_set])
    X = scaled_df[predictors]
    Y = scaled_df[target]
    return X, Y, scaler

def standard_scaler(dataset, predictors, target, length_train):
    """
    Particiona el df en training y testing, aplica StandardScaler y retorna targets y predictores
    """
    train_set, test_set = split_train_test(dataset, length_train)
    scaler = StandardScaler()
    scaler.fit(train_set)
    scaled_train_set = pd.DataFrame(scaler.transform(train_set), columns = dataset.columns)
    scaled_test_set = pd.DataFrame(scaler.transform(test_set), columns = dataset.columns)
    scaled_df = pd.concat([scaled_train_set, scaled_test_set])
    X = scaled_df[predictors]
    Y = scaled_df[target]
    return X, Y, scaler

def robust_scaler(dataset, predictors, target, length_train):
    """
    Particiona el df en training y testing, aplica RobustScaler y retorna targets y predictores
    """
    train_set, test_set = split_train_test(dataset, length_train)
    scaler = RobustScaler()
    scaler.fit(train_set)
    scaled_train_set = pd.DataFrame(scaler.transform(train_set), columns = dataset.columns)
    scaled_test_set = pd.DataFrame(scaler.transform(test_set), columns = dataset.columns)
    scaled_df = pd.concat([scaled_train_set, scaled_test_set])
    X = scaled_df[predictors]
    Y = scaled_df[target]
    return X, Y, scaler

def quantile_transformer(dataset, quantiles, predictors, target, length_train):
    """
    Particiona el df en training y testing, aplica QuintileTransformer y retorna targets y predictores
    """
    train_set, test_set = split_train_test(dataset, length_train)
    scaler = QuantileTransformer(n_quantiles = quantiles)
    scaler.fit(train_set)
    scaled_train_set = pd.DataFrame(scaler.transform(train_set), columns = dataset.columns)
    scaled_test_set = pd.DataFrame(scaler.transform(test_set), columns = dataset.columns)
    scaled_df = pd.concat([scaled_train_set, scaled_test_set])
    X = scaled_df[predictors]
    Y = scaled_df[target]
    return X, Y, scaler

def power_transformer(dataset, predictors, target, length_train):
    """
    Particiona el df en training y testing, aplica PowerTransformer y retorna targets y predictores
    """
    train_set, test_set = split_train_test(dataset, length_train)
    scaler = PowerTransformer()
    scaler.fit(train_set)
    scaled_train_set = pd.DataFrame(scaler.transform(train_set), columns = dataset.columns)
    scaled_test_set = pd.DataFrame(scaler.transform(test_set), columns = dataset.columns)
    scaled_df = pd.concat([scaled_train_set, scaled_test_set])
    X = scaled_df[predictors]
    Y = scaled_df[target]
    return X, Y, scaler

def pca_transform(dataset, components, predictors, target):
    """
    Particiona el df en training y testing, aplica PCA y retorna targets y predictores
    """
    train_set, test_set = split_train_test(dataset, length_train)
    train_set = train_set[predictors]
    test_set = test_set[predictors]
    scaler_standard = StandardScaler()
    scaler_standard.fit(train_set)
    standardized_train_set = scaler_standard.transform(train_set)
    standardized_test_set = scaler_standard.transform(test_set)
    scaler = PCA(n_components = components)
    scaler.fit(standardized_train_set)
    scaled_train_set = pd.DataFrame(scaler.transform(standardized_train_set))
    scaled_test_set = pd.DataFrame(scaler.transform(standardized_test_set))
    scaled_df = pd.concat([scaled_train_set, scaled_test_set])
    X = scaled_df
    Y = dataset[target]
    return X, Y, scaler

def inverse_transform(data, scaler, n_features):
    """
    Aplica la transformada inversa dado un scaler.
    """
    data = data.reshape(-1, 1)
    for i in range(data.shape[1]):
        tmp = np.zeros((data.shape[0], n_features))
        tmp[:, 0] = data[:, i]
        data[:, i] = scaler.inverse_transform(tmp)[:, 0]
    return data

# Hyperparameter search spaces

# La variable search_space define el espacio de búsqueda para que hyperopt evalúe la mejor combinación de hiperparámetros.
search_space = hp.choice('regressor',[
    {
        'model': xgb,
        'preprocessing': hp.choice('preprocessing', ['NoTransform', 'MinMaxScaler',
                         'StandardScaler', 'RobustScaler', 'QuantileTransformer',
                         'PowerTransformer']),
        'k_features': hp.quniform('k_features', 5, 12, 1),
        'n_components': sample(scope.int(hp.quniform('n_components', 1, 3, 1))),
        'params': {'booster': hp.choice('booster', ['gbtree']), #'dart']),
        'quantiles': hp.quniform('quantiles', 4, 7, 1),
        'n_estimators': hp.quniform('n_estimators', 50, 300, 1),
        'eta': hp.loguniform('eta', np.log(0.01), np.log(0.2)),
        'gamma': hp.uniform('gamma', 1, 4),
        'max_depth': hp.quniform('max_depth', 3, 10, 1),
        'min_child_weight': hp.uniform('min_child_weight', 8, 15),
        'random_state': sample(scope.int(hp.quniform('random_state', 4, 8, 1))),
        'subsample': hp.uniform('subsample', 0.2, 1),
        'alpha': hp.uniform('alpha', 4, 8),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 1),
        'sample_type': hp.choice('sample_type', ['uniform', 'weighted']),
        'normalize_type': hp.choice('normalize_type', ['tree', 'forest']),
        'grow_policy': hp.choice('grow_policy', ['depthwise', 'lossguide']),
        'rate_drop': hp.uniform('rate_drop', 0.5, 1),
        'skip_drop': hp.uniform('skip_drop', 0.5, 1),
        'colsample_bylevel':  hp.uniform('colsample_bylevel', 0.7, 1),
        'colsample_bynode': hp.uniform('colsample_bynode', 0.7, 1),
        'reg_lambda':  hp.uniform('reg_lambda', 4, 8)}
    }
])

# Model selection

def objective_function(args):
    # n_components = args['n_components']
    quantiles = int(args['params']['quantiles'])
    if args['preprocessing'] == 'NoTransform':
        X, Y, scaler = transform(dataset, predictors, target, length_train)
    elif args['preprocessing'] == 'MinMaxScaler':
        X, Y, scaler = transform(dataset, predictors, target, length_train)
    elif args['preprocessing'] == 'StandardScaler':
        X, Y, scaler = standard_scaler(dataset, predictors, target, length_train)
    elif args['preprocessing'] == 'RobustScaler':
        X, Y, scaler = robust_scaler(dataset, predictors, target, length_train)
    elif args['preprocessing'] == 'QuantileTransformer':
        X, Y, scaler = quantile_transformer(dataset, quantiles, predictors, target, length_train)
    elif args['preprocessing'] == 'PowerTransformer':
        X, Y, scaler = power_transformer(dataset, predictors, target, length_train)
    if args['model'] == xgb:
        booster = args['params']['booster']
        eta = args['params']['eta']
        gamma = args['params']['gamma']
        max_depth = int(args['params']['max_depth'])
        n_estimators = int(args['params']['n_estimators'])
        min_child_weight = args['params']['min_child_weight']
        subsample = args['params']['subsample']
        alpha = args['params']['alpha']
        random_state = args['params']['random_state']
        colsample_bytree = args['params']['colsample_bytree']
        colsample_bylevel = args['params']['colsample_bylevel']
        colsample_bynode = args['params']['colsample_bynode']
        reg_lambda = args['params']['reg_lambda']
        grow_policy = args['params']['grow_policy']
        if booster == 'dart':
            sample_type = args['params']['sample_type']
            normalize_type = args['params']['normalize_type']
            rate_drop = args['params']['rate_drop']
            skip_drop = args['params']['skip_drop']
        if args['preprocessing'] != 'PCA':
            k_features = int(args['k_features'])
        else:
            k_features = X.shape[1]
        if booster == 'gbtree':
            estimator = xgb.XGBRegressor(objective= 'reg:squarederror', booster = booster, eta = eta, gamma = gamma, max_depth = max_depth, n_estimators = n_estimators,
                              min_child_weight = min_child_weight, subsample = subsample, alpha = alpha, random_state = random_state,
                              colsample_bytree = colsample_bytree, colsample_bylevel = colsample_bylevel, grow_policy = grow_policy,
                              colsample_bynode = colsample_bynode, reg_lambda = reg_lambda, n_jobs = -1)
            reg = SFS(estimator, cv = piter, k_features = k_features, forward = True, floating = False, scoring = metrics_names[eval_metric])
        elif booster == 'dart':
            num_round = 50
            estimator = xgb.XGBRegressor(objective= 'reg:squarederror', booster = booster, eta = eta, gamma = gamma, max_depth = max_depth, n_estimators = n_estimators,
                              min_child_weight = min_child_weight, subsample = subsample, alpha = alpha, random_state = random_state,
                              colsample_bytree = colsample_bytree, sample_type = sample_type, normalize_type = normalize_type,
                              rate_drop = rate_drop, skip_drop = skip_drop, colsample_bylevel = colsample_bylevel, grow_policy = grow_policy,
                              colsample_bynode = colsample_bynode, reg_lambda = reg_lambda, n_jobs = -1)
            reg = SFS(estimator, cv = piter, k_features = k_features, forward = True, floating = False, scoring = metrics_names[eval_metric])
    if eval_metric == 'mse':
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 1 - 0.8, random_state = 1, shuffle = False)
        # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 1 - length_train/len(dataset), random_state = 1, shuffle = False)
        sfsl = reg.fit(X, Y)
        x_sfs = sfsl.transform(X)
        x_train_sfs = x_sfs[:length_train]
        x_test_sfs = x_sfs[length_train:]
        estimator.fit(x_train_sfs, y_train)
        if args['model'] == xgb:
            if booster == "gbtree":
                y_pred = estimator.predict(x_test_sfs)
            elif booster == "dart":
                y_pred = estimator.predict(x_test_sfs, ntree_limit = num_round)
        else:
            y_pred = estimator.predict(x_test_sfs)
        if args['preprocessing'] != 'NoTransform':
            predictions = y_pred.reshape(-1, 1)
            for i in range(predictions.shape[1]):
                if args['preprocessing'] != 'PCA':
                    tmp = np.zeros((predictions.shape[0], n_features))
                else:
                    tmp = np.zeros((predictions.shape[0], X.shape[1]))
                tmp[:, 0] = predictions[:, i]
                predictions[:, i] = scaler.inverse_transform(tmp)[:, 0]
            mse = mean_squared_error(dataset[target][length_train:], predictions)
            print('mse value: {}, Selected variables {}'.format(mse, sfsl.k_feature_names_))
            return mse
        else:
            mse = mean_squared_error(dataset[target][length_train:], y_pred)
            print('mse value: {}, Selected variables {}'.format(mse, sfsl.k_feature_names_))
            return mse
    else:
        #reg.fit(X, Y)
        #print('r2 value: {}, Selected variables {}'.format(reg.k_score_, reg.k_feature_names_))
        #loss_function = 1 - reg.k_score_
        cross_score = cross_val_score(estimator, X, Y, cv = 10)
        print('r2 value: {}'.format(np.average(cross_score)))
        loss_function = 1 - np.average(cross_score)
        return loss_function

def select_model(space):
    best_regressor = fmin(objective_function, space, algo = tpe.suggest, max_evals = max_evals)
    print(hyperopt.space_eval(space, best_regressor))
 
select_model(search_space)
