import xgboost as xgb
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, KFold, RandomizedSearchCV, cross_validate
from sklearn.metrics import precision_score, recall_score, accuracy_score, cohen_kappa_score, confusion_matrix, make_scorer

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt import hp
import random
import traceback
import pickle
import csv
import copy

import numpy as np

def precision_s(y_true, y_pred):
    return precision_score(y_true, y_pred, average=None)[1]

def recall_s(y_true, y_pred):
    return recall_score(y_true, y_pred, average=None)[1]

class Classifier:
    
    def best_score(self, results):
        candidate = np.flatnonzero(results['rank_test_score'] == 1)[0]
        mean_score = results['mean_test_score'][candidate]
        params = results['params'][candidate]
        print("Model with rank: 1")
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              results['mean_test_score'][candidate],
              results['std_test_score'][candidate]))
        print("Parameters: {0}".format(results['params'][candidate]))
        print("")
        return mean_score, params
    
    def metrics(self, predicted, test):
        acc = accuracy_score(test, predicted)
        prec = precision_score(test, predicted, average=None)[1]
        rec = recall_score(test, predicted, average=None)[1]
        kpp = cohen_kappa_score(test, predicted)
        return {"acc": acc, "prec": prec, "rec": rec, "kpp": kpp}

class XGBoost_Classifier(Classifier):
    
    def __init__(self, qtd_classes):
        self.num_class = qtd_classes
    
    def search_model(self, X, y, grid, iters):        
        xgb_model = xgb.XGBClassifier(objective='multi:softprob', random_state=42)
        
        search = RandomizedSearchCV(xgb_model, param_distributions=grid, random_state=42, n_iter=iters, cv=3, verbose=1, n_jobs=4, pre_dispatch = 2, return_train_score=True)

        search.fit(X, y)
        
        return self.best_score(search.cv_results_)
    
    def train_single(self, X_train, Y_train, X_test, Y_test, params, steps=200):
        D_train = xgb.DMatrix(X_train, label=Y_train)
        D_test = xgb.DMatrix(X_test, label=Y_test)
        
        model = xgb.train(params, D_train, steps)
        y_predicted = model.predict(D_test)
        y_predicted = np.asarray([np.argmax(line) for line in y_predicted])
    
        response = self.metrics(y_predicted, Y_test)
        response["status"] = STATUS_OK
        response["params"] = params
        return response

class SVM_Classifier(Classifier):
    
    def search_model(self, X, y, grid, iters):        
        svc = SVC(probability=True)
        
        search = RandomizedSearchCV(svc, param_distributions=grid, random_state=42, n_iter=iters, cv=3, verbose=3, n_jobs=4, pre_dispatch = 2, return_train_score=True)
        
        search.fit(X, y)
        
        return self.best_score(search.cv_results_)
    
    def train_single(self, X_train, Y_train, X_test, Y_test, params):
        svc = SVC(probability=True, **params)
        
        model = svc.fit(X_train, Y_train)
        y_predicted = model.predict(X_test)
        
        response = self.metrics(y_predicted, Y_test)
        response["status"] = STATUS_OK
        response["params"] = params
        return response

class KNN_Classifier(Classifier):
    
    def search_model(self, X, y, grid, iters):        
        knn = KNeighborsClassifier()
        
        search = RandomizedSearchCV(knn, param_distributions=grid, random_state=42, n_iter=iters, cv=3, verbose=1, n_jobs=4, pre_dispatch = 2, return_train_score=True)
        
        search.fit(X, y)
        
        return self.best_score(search.cv_results_)
    
    def train_single(self, X_train, Y_train, X_test, Y_test, params):
        knn = KNeighborsClassifier(**params)
        
        model = knn.fit(X_train, Y_train)
        y_predicted = model.predict(X_test)
        
        response = self.metrics(y_predicted, Y_test)
        response["status"] = STATUS_OK
        response["params"] = params
        return response

class RandomForest_Classifier(Classifier):

    def search_model(self, X, y, grid, iters):        
        rfc = RandomForestClassifier()
        
        search = RandomizedSearchCV(rfc, param_distributions=grid, random_state=42, n_iter=iters, cv=3, verbose=1, n_jobs=4, pre_dispatch = 2,return_train_score=True)
        
        search.fit(X, y)
        
        return self.best_score(search.cv_results_)
    
    def train_single(self, X_train, Y_train, X_test, Y_test, params):
        rfc = RandomForestClassifier(**params)
        
        model = rfc.fit(X_train, Y_train)
        y_predicted = model.predict(X_test)
        
        response = self.metrics(y_predicted, Y_test)
        response["status"] = STATUS_OK
        response["params"] = params
        return response
    
class GausianNB_Classifier(Classifier):
    
    def search_model(self, X, y, grid, iters):        
        gss = GaussianNB()
        
        search = RandomizedSearchCV(gss, param_distributions=grid, random_state=42, n_iter=iters, cv=3, verbose=1, n_jobs=4,pre_dispatch = 2, return_train_score=True)
        
        search.fit(X, y)
        
        return self.best_score(search.cv_results_)
    
    def train_single(self, X_train, Y_train, X_test, Y_test, params):
        gss = GaussianNB(**params)
        
        model = gss.fit(X_train, Y_train)
        y_predicted = model.predict(X_test)
        
        response = self.metrics(y_predicted, Y_test)
        response["status"] = STATUS_OK
        response["params"] = params
        return response

class MLP_Classifier(Classifier):
    
    def search_model(self, X, y, grid, iters):
        mlp = MLPClassifier()
        
        search = RandomizedSearchCV(mlp, param_distributions=grid, random_state=42, n_iter=iters, cv=3, verbose=1, n_jobs=4,pre_dispatch = 2, return_train_score=True)
        
        search.fit(X, y)
        
        return self.best_score(search.cv_results_)
    
    def train_single(self, X_train, Y_train, X_test, Y_test, params):
        mlp = MLPClassifier(**params)
        
        model = mlp.fit(X_train, Y_train)
        y_predicted = model.predict(X_test)
        
        response = self.metrics(y_predicted, Y_test)
        response["status"] = STATUS_OK
        response["params"] = params
        return response

def chooseClassifier(features_train, labels_train, features_test, labels_test, params):
    if params['classifier']['method'] == 'SVM':
        c = SVM_Classifier()
    elif params['classifier']['method'] == 'XGB':
        c = XGBoost_Classifier(5)
    elif params['classifier']['method'] == 'KNN':
        c = KNN_Classifier()
    elif params['classifier']['method'] == 'RFC':
        c = RandomForest_Classifier()
    elif params['classifier']['method'] == 'GSS':
        c = GausianNB_Classifier()
    elif params['classifier']['method'] == "MLP":
        c = MLP_Classifier()
    params = copy.deepcopy(params["classifier"])
    del params["method"]
    scores = c.train_single(features_train, labels_train, features_test, labels_test, params)
    return scores

param_space ={
    'classifier': hp.choice('classifier', [
        {
            'method': 'SVM',
            'C': hp.choice('C', [1, 2, 5, 10, 100, 1000]),
            'gamma': hp.choice('gamma', [0.5, 0.1, 0.01, 0.001, 0.0001]),
            'kernel': hp.choice('kernel', ['linear', 'rbf', 'sigmoid'])
        },
        {
            'method': 'XGB',
            'max_depth': hp.choice('max_depth', [ 35, 40]), 
            'min_child_weight': hp.choice('min_child_weight', [ 5, 7 ]), 
            'gamma': hp.choice('gamma2', [ 0.0, 0.1, 0.2 ]), 
            'learning_rate': hp.choice('learning_rate', [ 0.3, 0.6]), 
            'colsample_bytree': hp.choice('colsample_bytree', [ 0.3, 0.4, 0.5])
        },
        {
            'method': 'KNN',
            'n_neighbors': hp.choice('n_neighbors', [ 3, 5, 11, 19, 40]),
            'weights': hp.choice('weights', ['uniform', 'distance']),
            'metric': hp.choice('metric', ['euclidean', 'manhattan'])
        },
        {
            'method': 'RFC',
            'n_estimators': hp.choice('n_estimators', [100, 200, 500]),
            'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2']), 
            'max_depth': hp.choice('max_depth2', [ 4, 6, 8, 10]),
            'criterion': hp.choice('criterion', ['gini', 'entropy'])
        },
        {
            'method': 'GSS',
            'var_smoothing': hp.choice('var_smoothing', [1e-7, 1e-9, 1e-11, 1e-13])
        },
        {
            'method': 'MLP',
            'hidden_layer_sizes': hp.choice('hidden_layer_sizes', [(10, 10, 10), (20,), (50,), (70,), (100,), (200,)]), 
            'activation': hp.choice('activation', ['identity', 'logistic', 'tanh', 'relu']), 
            'learning_rate': hp.choice('learning_rate2', ['constant', 'invscaling', 'adaptive']),
            'learning_rate_init': hp.choice('learning_rate_init', [0.001, 0.01, 0.1, 0.0001])
        }
    ])
}

model_features = "densenet"

def hyperopt_fitness(params: dict):
    
    features_train = np.load("../dataset_featurized/{0}/features.npy".format(model_features))
    labels_train = np.load("../dataset_featurized/{0}/labels.npy".format(model_features))
    features_test = np.load("../dataset_featurized/{0}/features_test.npy".format(model_features))
    labels_test = np.load("../dataset_featurized/{0}/labels_test.npy".format(model_features))
    
    scores = chooseClassifier(features_train, labels_train, features_test, labels_test, params)
    print("Classifier Score: ", scores)
    
    results = {
        'loss': 1-scores['acc'],
        'acurracy': scores['acc'],
        'space': params,
        'status': STATUS_OK
    }
    save_result(results)
    return results

def save_result(resultado):
    with open('resultados_{}.csv'.format(model_features),'a', newline='') as results:
        writer = csv.writer(results)      
        writer.writerow([resultado['acurracy'], resultado['loss'], resultado['space']])

def run_a_trial():
    try:
        trials = pickle.load(open("otimizacao_{}.pkl".format(model_features), "rb"))
        print("Encontrei uma otimização já salva! Carregando...")
        max_evals = len(trials.trials) + 1
        print("Rodando a partir da {} iteração.".format(
        len(trials.trials)))
        
    except:
        trials = Trials()
        print("Começando do zero.")
    
    trials = Trials()
    best = fmin(hyperopt_fitness, 
                param_space,
                algo=tpe.suggest, 
                max_evals=5, 
                trials=trials)
    
    pickle.dump(trials, open("otimizacao_{}.pkl".format(model_features), "wb"))

while True:
    try:
        run_a_trial()
    except Exception as err:
        err_str = str(err)
        print(err_str)
        traceback_str = str(traceback.format_exc())
        print(traceback_str)