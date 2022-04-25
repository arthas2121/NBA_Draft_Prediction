import numpy as np
from sklearn.metrics import ndcg_score
from sklearn import metrics
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import time

def all_models():
    model_list = {}
    lr = LinearRegression()
    model_list['Linear Regression'] = lr
    log_r = LogisticRegression(penalty='l1', solver='liblinear')
    model_list['Logistic Regression'] = log_r
    ridge = Ridge(alpha=0.8, solver='lsqr')
    model_list['Ridge'] = ridge
    lasso = Lasso(alpha=0.4, max_iter=200, selection='random')
    model_list['Lasso'] = lasso
    rf = RandomForestRegressor(n_estimators=140, min_samples_split=6, min_samples_leaf=5, max_depth=50)
    model_list['Random Forest'] = rf
    kn = KNeighborsClassifier(n_neighbors=7)
    model_list['KNeighbors'] = kn
    xgb = XGBRegressor(learning_rate=0.05, n_estimators=150)
    model_list['XGBoost'] = xgb
    return model_list

def use_regression_model(X_train, y_train, X_test, y_test):
    start = time.time()
    best_mae = 60
    best_ndcg = 0
    best_mae_model = ''
    best_ndcg_model = ''
    best_rank = np.zeros(y_test.shape)
    model_list = all_models()
    for model_name in model_list.keys():
        model_list[model_name].fit(X_train, y_train)
        y_pred = model_list[model_name].predict(X_test)
        y_pred = y_pred.argsort().argsort()
        y_test = y_test.argsort().argsort()
        mae = metrics.mean_absolute_error(y_pred, y_test)
    #print("MAE: {:,.5f}".format(mae))
        ndcg = ndcg_score([y_test], [y_pred])
    #print("ndcg score: ", ndcg)
        if mae < best_mae:
            best_mae_model = model_name
            best_mae = mae
        ndcg = ndcg_score([y_test], [y_pred])
        if ndcg > best_ndcg:
            best_ndcg_model = model_name
            best_ndcg = ndcg
            best_rank = y_pred
    print(f'Minimum MAE value: {best_mae:.3f} | Using model: {best_mae_model}')
    print(f'Best NDCG score: {best_ndcg:.3f} | Using model: {best_ndcg_model}')
    end = time.time()
    print('Runtime: {:,.5f}s'.format(end - start))
    return best_rank