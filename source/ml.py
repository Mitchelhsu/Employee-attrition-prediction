
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop('Attrition', axis = 1) #feature matrix
y = df.Attrition # target vector

# standard scaled X matrix
scaler = StandardScaler()
scaler.fit(X)
X_std = scaler.transform(X)

X_train, X_test, y_train, y_test =  train_test_split(X_std, y, test_size=0.20, random_state=111, stratify = y)

# creating logistic regression object
from sklearn.linear_model import LogisticRegression

# Logistic Regression

logreg = LogisticRegression()

# Create regularization penalty space
penalty = ['l1', 'l2']

# Create regularization hyperparameter space
C = np.logspace(0, 4, 10)

# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)

# creating grid search with 5 fold
from sklearn.model_selection import GridSearchCV

clf = GridSearchCV(logreg, hyperparameters, cv=5, verbose=0)

# fit grid search on data
model = clf.fit(X,y)

# view best parameters 
print('Best Penalty:', model.best_estimator_.get_params()['penalty'])
print('Best C:', model.best_estimator_.get_params()['C'])

# calculate test score
from sklearn import metrics

model = model.best_estimator_.fit(X_train, y_train)

# predictions for test set
y_preds = model.predict(X_test)

test_accuracy_log = metrics.accuracy_score(y_test, y_preds)
test_recall_log = metrics.recall_score(y_test, y_preds)
print('Test Accuracy:', test_accuracy_log)
print('Test Sensitivity:', test_recall_log)

# K Nearest Neighbors

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
%matplotlib inline 

# define a range for k values
k_range = list(range(1,31))

# create a parameter grid
param_grid = dict(n_neighbors=k_range)

knn=KNeighborsClassifier()
# instantiate the grid
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False)

# fit the grid with data
model = grid.fit(X,y)

# examine the best model
print('Best Estimator:', model.best_estimator_)
print('CV Score:', model.best_score_)

model=model.best_estimator_.fit(X_train, y_train)

# predictions for test set
y_preds = model.predict(X_test)

# calculate test score
test_accuacy_knn = metrics.accuracy_score(y_test, y_preds)
test_recall_knn = metrics.recall_score(y_test, y_preds)

print('Test Accuracy:', test_accuacy_knn)
print('Test Sensitivity:', test_recall_knn)

# Decision Tree with GridSearchCV

from sklearn.tree import DecisionTreeClassifier 

parameters={'min_samples_split' : range(10,500,50),'max_depth': range(1,16,2)}

clf_tree=DecisionTreeClassifier()

clf=GridSearchCV(clf_tree,parameters)

model = clf.fit(X,y)

# examine the best model
print('Best Estimator:', model.best_estimator_)
print('Best Score:', model.best_score_)

model=model.best_estimator_.fit(X_train, y_train)

# predictions for test set
y_preds = model.predict(X_test)

# calculate test score
test_accuracy_dt = metrics.accuracy_score(y_test, y_preds)
test_recall_dt = metrics.recall_score(y_test, y_preds)

print('Test Accuracy:', test_accuracy_dt)
print('Test Sensitivity:', test_recall_dt)

#Random Forest with GridSearchCv

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

param_grid = { 
    'n_estimators': [10, 20,50],
    'max_features': ['auto', 'sqrt', 'log2']
}

grid = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 10)
model = grid.fit(X, y)

# examine the best model
print('Best Estimator:',model.best_estimator_)
print('Best Random Forest Score:',model.best_score_)
model=model.best_estimator_.fit(X_train, y_train)

# predictions for test set
y_preds = model.predict(X_test)

# calculate test score
test_accuracy_rf = metrics.accuracy_score(y_test, y_preds)
test_recall_rf = metrics.recall_score(y_test, y_preds)

print('Test Accuracy:', test_accuracy_rf)
print('Test Sensitivity:', test_recall_rf)

# Extra Trees Classifier

from sklearn.ensemble import ExtraTreesClassifier

clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)

scores=cross_val_score(clf, X, y, cv=10)
print('CV Score:',scores.mean())

model = clf.fit(X_train, y_train)

# predictions for test set
y_preds = model.predict(X_test)

# calculate test score

test_accuracy_et = metrics.accuracy_score(y_test, y_preds)
test_recall_et = metrics.recall_score(y_test, y_preds)

print('Test Accuracy:', test_accuracy_et)
print('Test Sensitivity:', test_recall_et)

# AdaBoost with Decision Trees

from sklearn.ensemble import AdaBoostClassifier

# create adaboost classifier with decision trees as base estimator
clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=50)

scores = cross_val_score(clf, X, y, cv=5)

print('CV Score:',scores.mean())
model = clf.fit(X_train, y_train)

# predictions for test set
y_preds = model.predict(X_test)

# calculate test score
test_accuracy_ada = metrics.accuracy_score(y_test, y_preds)
test_recall_ada = metrics.recall_score(y_test, y_preds)

print('Test Accuracy:', test_accuracy_ada)
print('Test Sensitivity:', test_recall_ada)

# Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1)

scores = cross_val_score(clf, X, y, cv=10)

print('CV Score:',scores.mean())
model = clf.fit(X_train, y_train)

# predictions for test set
y_preds = model.predict(X_test)

# calculate test score
test_accuracy_gb = metrics.accuracy_score(y_test, y_preds)
test_recall_gb = metrics.recall_score(y_test, y_preds)

print('Test Accuracy:', test_accuracy_gb)
print('Test Sensitivity:', test_recall_gb)

#Light GBM Classifier

import lightgbm as lgb
from lightgbm import LGBMClassifier

lgb = LGBMClassifier(max_depth=7,
                        num_leaves=36,
                        learning_rate=0.4)

model = lgb.fit(X_train, y_train)

y_preds = model.predict(X_test)


model = lgb.fit(X_train, y_train)
y_preds=model.predict(X_test)

lgb_cv_score = cross_val_score(lgb, X,y,cv=5)
test_accuracy_lgb = metrics.accuracy_score(y_preds, y_test)
test_recall_lgb = metrics.recall_score(y_preds, y_test)

print('CV Score:', lgb_cv_score.mean())
print('Test Accuracy:', test_accuracy_lgb)
print('Test Sensitivity:', test_recall_lgb)

# XGBoost

import xgboost as xgb
from xgboost import XGBClassifier  

xgb = XGBClassifier(silent=False, 
                      scale_pos_weight=1,
                      learning_rate=0.1,  
                      colsample_bytree = 0.4,
                      subsample = 0.8,
                      objective='binary:logistic', 
                      n_estimators=1000, 
                      reg_alpha = 0.3,
                      max_depth=9, 
                      gamma=0.1)
model = xgb.fit(X_train, y_train)
y_preds=model.predict(X_test)

# xgb_cv_score = cross_val_score(xgb, X,y,cv=5)
test_accuracy_xgb = metrics.accuracy_score(y_preds, y_test)
test_recall_xgb = metrics.recall_score(y_preds, y_test)

# print('Cross Validation Score:', xgb_cv_score.mean())
print('Test Accuracy:', test_accuracy_xgb)
print('Test Sensitivity:', test_recall_xgb)