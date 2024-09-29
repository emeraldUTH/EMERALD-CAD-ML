from array import array
from enum import auto
from re import sub
from turtle import backward
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn import svm
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from tabpfn import TabPFNClassifier
import xgboost
from sklearn import tree
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import itertools
import sys
import multiprocessing
from tqdm import tqdm #tqmd progress bar

from genetic_selection import GeneticSelectionCV
from sklearn.tree import DecisionTreeClassifier

# function for printing each component of confusion matrix
def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

data_path = 'path_to_csv'
data = pd.read_csv(data_path)
print(data.columns)
print(data.values)
dataframe = pd.DataFrame(data.values, columns=data.columns)
dataframe['CAD'] = data.CAD
x = dataframe.drop(['ID','female','CNN_Healthy','CNN_CAD','Doctor: CAD','HEALTHY','CAD'], axis=1) # Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’).
x_nodoc = dataframe.drop(['ID','female','CNN_Healthy','CNN_CAD','Doctor: CAD', 'Doctor: Healthy','HEALTHY','CAD'], axis=1) # Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’).
print("x:\n",x.columns)
y = dataframe['CAD'].astype(int)

# ml algorithms initialization
svm = svm.SVC(kernel='rbf')

# doc/no_doc parameterization
sel_alg = svm
x = x_nodoc #TODO comment when not testing with doctor
X = x_nodoc

#############################################
#### Genetic Algorithm Feature Selection ####
#############################################
for i in range (0,3):
    print("run no ", i, ":")
    selector = GeneticSelectionCV(
        estimator=sel_alg,
        cv=10,
        verbose=2,
        scoring="accuracy", 
        max_features=26, #TODO change to 27 when testing with doctor, 26 without
        caching=True,
        n_jobs=1)
    selector = selector.fit(x, y)
    n_yhat = selector.predict(x)
    sel_features = x.columns[selector.support_]
    print("Genetic Feature Selection:", x.columns[selector.support_])
    print("Genetic Accuracy Score: ", selector.score(x, y))
    print("Testing Accuracy: ", metrics.accuracy_score(y, n_yhat))

    ###############
    #### CV-10 ####
    ###############
    x = x_nodoc
    for feature in x.columns:
        if feature in sel_features:
            pass
        else:
            X = X.drop(feature, axis=1)
    
    print("cv-10 accuracy: ", cross_val_score(sel_alg, X, y, scoring='accuracy', cv = 10).mean() * 100)

sel_features = x_nodoc

##############
### CV-10 ####
##############
for feature in x.columns:
    if feature in sel_features:
        pass
    else:
        X = X.drop(feature, axis=1)

est = sel_alg.fit(X, y)
n_yhat = est.predict(X)

print("cv-10 accuracy: ", cross_val_score(sel_alg, X, y, scoring='accuracy', cv = 10).mean() * 100)
print("cv-10 accuracy STD: ", cross_val_score(sel_alg, X, y, scoring='accuracy', cv = 10).std() * 100)
print("f1_score: ", cross_val_score(sel_alg, X, y, scoring='f1', cv = 10).mean() * 100)
print("f1_score STD: ", cross_val_score(sel_alg, X, y, scoring='f1', cv = 10).std() * 100)
print("jaccard_score: ", cross_val_score(sel_alg, X, y, scoring='jaccard', cv = 10).mean() * 100)
print("jaccard_score STD: ", cross_val_score(sel_alg, X, y, scoring='jaccard', cv = 10).std() * 100)
scoring = {
    'sensitivity': metrics.make_scorer(metrics.recall_score),
    'specificity': metrics.make_scorer(metrics.recall_score,pos_label=0)
}
print("sensitivity: ", cross_val_score(sel_alg, X, y, scoring=scoring['sensitivity'], cv = 10).mean() * 100)
print("sensitivity STD: ", cross_val_score(sel_alg, X, y, scoring=scoring['sensitivity'], cv = 10).std() * 100)
print("specificity: ", cross_val_score(sel_alg, X, y, scoring=scoring['specificity'], cv = 10).mean() * 100)
print("specificity STD: ", cross_val_score(sel_alg, X, y, scoring=scoring['specificity'], cv = 10).std() * 100)

print("confusion matrix:\n", metrics.confusion_matrix(y, n_yhat, labels=[0,1]))
print("TP/FP/TN/FN: ", perf_measure(y, n_yhat))
