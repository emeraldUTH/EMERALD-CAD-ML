from turtle import backward
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn import metrics
from sklearn import svm
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from tabpfn import TabPFNClassifier
import shap

def cohen_effect_size(X, y):
    """Calculates the Cohen effect size of each feature.
    
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target vector relative to X
        Returns
        -------
        cohen_effect_size : array, shape = [n_features,]
            The set of Cohen effect values.
        Notes
        -----
        Based on https://github.com/AllenDowney/CompStats/blob/master/effect_size.ipynb
    """
    group1, group2 = X[y==0], X[y==1]
    diff = group1.mean() - group2.mean()
    var1, var2 = group1.var(), group2.var()
    n1, n2 = group1.shape[0], group2.shape[0]
    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    d = diff / np.sqrt(pooled_var)
    return d

# function for printing each feature's SHAP value
def shapley_feature_ranking(shap_values, X):
    """Calculates the SHAP value of each feature.
    
        Parameters
        ----------
        shap_values : array-like, shape = [n_samples, n_features]
            vector, where n_samples in the number of samples and
            n_features is the number of features.
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        Returns
        -------
        pd.DataFrame [n_features, 2]
            Dataframe containing feature names and according SHAP value.
    """
    feature_order = np.argsort(np.mean(shap_values, axis=0))
    return pd.DataFrame(
        {
            "features": [X.columns[i] for i in feature_order][::-1],
            "importance": [
                np.mean(shap_values, axis=0)[i] for i in feature_order
            ][::-1],
        }
    )

# function to print SHAP values and plots
# function to print SHAP values and plots
def xai(model, X, val):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    ###
    sv = explainer(X)
    exp = shap.Explanation(sv[:,:,1], sv.base_values[:,1], X, feature_names=X.columns)
    idx_healthy = 2 # datapoint to explain (healthy)
    idx_cad = 9 # datapoint to explain (CAD)
    shap.waterfall_plot(exp[idx_healthy])
    shap.waterfall_plot(exp[idx_cad])
    ###

    shap.summary_plot(shap_values[val], X)
    # shap.summary_plot(shap_values[0], X, plot_type="bar")
    shap.summary_plot(shap_values[0], X, plot_type='violin')
    for feature in X.columns:
        print(feature)
        shap.dependence_plot(feature, shap_values[0], X)
    shap.force_plot(explainer.expected_value[0], shap_values[0][0], X.iloc[0,:], matplotlib=True)
    shap.force_plot(explainer.expected_value[1], shap_values[0][0], X.iloc[0,:], matplotlib=True)

    ###
    shap_rank = shapley_feature_ranking(shap_values[0], X)
    shap_rank.sort_values(by="importance", ascending=False)
    print(shap_rank)

data = pd.read_csv('CAD/src/cad_dset.csv')
dataframe = pd.DataFrame(data.values, columns=data.columns)
dataframe['CAD'] = data.CAD
x = dataframe.drop(['ID','female','CNN_Healthy','CNN_CAD','Doctor: CAD','HEALTHY','CAD'], axis=1) # Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’).
x_nodoc = dataframe.drop(['ID','female','CNN_Healthy','CNN_CAD','Doctor: CAD', 'Doctor: Healthy','HEALTHY','CAD'], axis=1) # Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’).
y = dataframe['CAD'].astype(int)

# ml algorithms initialization
svm = svm.SVC(kernel='rbf')
lr = linear_model.LinearRegression()
dt = DecisionTreeClassifier()
rndF = RandomForestClassifier(max_depth=None, random_state=0, n_estimators=80)
ada = AdaBoostClassifier(n_estimators=150, random_state=0)
knn = KNeighborsClassifier(n_neighbors=20) #TODO n_neighbors=13 when testing with doctor, 20 w/o doctor
tab = TabPFNClassifier(device='cpu', N_ensemble_configurations=26)
catb = CatBoostClassifier(n_estimators=79, learning_rate=0.1, verbose=False)


x = x_nodoc #TODO ucommment when running w/o doctor
X = x
# sel_features = no_doc_catb #TODO here input the array with the feature names
sel_alg = svm


# best performing subset
doc_rdnF_80_none = ['known CAD', 'previous PCI', 'Diabetes', 'Chronic Kindey Disease', 'ANGINA LIKE', 'RST ECG', 'male', '<40', 'Doctor: Healthy'] # rndF 84,41% -> cv-10: 83,02%

sel_features = doc_rdnF_80_none
for feature in x.columns:
    if feature in sel_features:
        pass    
    else:
        X = X.drop(feature, axis=1)

est = sel_alg.fit(X, y)
n_yhat = cross_val_predict(est, X, y, cv=10)
print("Testing Accuracy: {a:5.2f}%".format(a = 100*metrics.accuracy_score(y, n_yhat)))

# cross-validate result(s) 10fold
cv_results = cross_validate(sel_alg, X, y, cv=10)
# sorted(cv_results.keys())
print("Avg CV-10 Testing Accuracy: {a:5.2f}%".format(a = 100*sum(cv_results['test_score'])/len(cv_results['test_score'])))
print("metrics:\n", metrics.classification_report(y, n_yhat, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=True, zero_division='warn'))
print("f1_score: ", metrics.f1_score(y, n_yhat, average='weighted'))
print("jaccard_score: ", metrics.jaccard_score(y, n_yhat,pos_label=1))
print("confusion matrix:\n", metrics.confusion_matrix(y, n_yhat, labels=[0,1]))

print("###### SHAP ######")
# print('Number of features %d' % len(est.feature_names_in_))
effect_sizes = cohen_effect_size(X, y)
effect_sizes.reindex(effect_sizes.abs().sort_values(ascending=False).nlargest(40).index)[::-1].plot.barh(figsize=(6, 10))
plt.title('Features with the largest effect sizes')
plt.show()

xai(est, X, 0)
