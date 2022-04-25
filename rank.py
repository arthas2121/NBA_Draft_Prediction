import numpy as np
import utils
import neural_network
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import time

def all_models():
    model_list = {}
    xgb = XGBClassifier(max_depth=5, learning_rate=0.05)
    model_list['XGBoost'] = xgb
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, criterion='entropy')
    model_list['Random Forest'] = rf
    dt = DecisionTreeClassifier()
    model_list['Decision Tree'] = dt
    kn = KNeighborsClassifier(n_neighbors=3)
    model_list['KNeighbors'] = kn
    nb = GaussianNB()
    model_list['Naive Bayes'] = nb
    return model_list
    

def use_rank_model(X_train, y_train, X_test, y_test):
    start = time.time()
    X_new, y_new = utils.transform_pairwise(X_train, y_train)
    y_new = np.where(y_new == -1, 0, y_new)
    X_test_new, y_test_new = utils.transform_pairwise(X_test, y_test)
    y_test_new = np.where(y_test_new == -1, 0, y_test_new)
    best_acc = 0
    best_ndcg = 0
    best_acc_model = ''
    best_ndcg_model = ''
    best_rank = np.zeros(y_test.shape)
    model_list = all_models()
    for model_name in model_list.keys():
        model_list[model_name].fit(X_new, y_new)
        y_pred = model_list[model_name].predict(X_test_new)
        accuracy = accuracy_score(y_test_new, y_pred)
        if accuracy > best_acc:
            best_acc_model = model_name
            best_acc = accuracy
        rank_pred, score = utils.calc_ndcg(y_pred, y_test)
        if score > best_ndcg:
            best_ndcg_model = model_name
            best_ndcg = score
            best_rank = rank_pred
    print(f'Best pairwise accuracy: {best_acc:.3f} | Using model: {best_acc_model}')
    print(f'Best NDCG score: {best_ndcg:.3f} | Using model: {best_ndcg_model}')
    end = time.time()
    print('Runtime: {:,.5f}s'.format(end - start))
    return best_rank