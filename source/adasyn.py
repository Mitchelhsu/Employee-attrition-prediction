#!/usr/bin/env python
# coding: utf-8

# In[40]:


import imblearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from imblearn.pipeline import make_pipeline as imb_pipline
from sklearn.naive_bayes import GaussianNB, BernoulliNB, CategoricalNB, MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_val_predict
from sklearn.metrics import f1_score, precision_score, recall_score, mean_squared_error, r2_score
from sklearn import svm, tree
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import fbeta_score, make_scorer, precision_score, recall_score
from math import sqrt
import seaborn as sns
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import NearMiss, CondensedNearestNeighbour, EditedNearestNeighbours, AllKNN


# In[41]:


data = pd.read_csv('../p_train.csv')
st = pd.read_csv('../PerStatus.csv')
data.drop('yyyy', axis=1, inplace=True)


# In[42]:


X = preprocessing.scale(data.iloc[:, :].values)
Y = st.iloc[:, 0].values


# In[43]:


classifier = GaussianNB()


# In[31]:


#Make pipeline of clf
clf_pipeline = make_pipeline(classifier)
y_pred = cross_val_predict(clf_pipeline, X, Y, cv=10)
print('Accuracy score:', metrics.accuracy_score(Y, y_pred))
print('RMSE:', round(sqrt(mean_squared_error(Y, y_pred)), 2))
print('R_squared:', round(r2_score(Y, y_pred), 2))
print('Recall score:', metrics.recall_score(Y, y_pred))
print('Fbeta score:', fbeta_score(Y, y_pred, beta=1.5))


# In[33]:


#Make SMOTE pipeline
smote_pipline = imb_pipline(SMOTE(), classifier)
smote_pred = cross_val_predict(smote_pipline, X, Y, cv=10)
print('Accuracy score:', metrics.accuracy_score(Y, smote_pred))
print('RMSE:', round(sqrt(mean_squared_error(Y, smote_pred)), 2))
print('R_squared:', round(r2_score(Y, smote_pred), 2))
print('Recall score:', metrics.recall_score(Y, smote_pred))
print('Fbeta score:', fbeta_score(Y, smote_pred, beta=1.5))


# In[35]:


#Make NearMiss pipeline
nearmiss_pipline = imb_pipline(NearMiss(), classifier)
nearmiss_pred = cross_val_predict(nearmiss_pipline, X, Y, cv=10)
print('Accuracy score:', metrics.accuracy_score(Y, nearmiss_pred))
print('RMSE:', round(sqrt(mean_squared_error(Y, nearmiss_pred)), 2))
print('R_squared:', round(r2_score(Y, nearmiss_pred), 2))
print('Recall score:', metrics.recall_score(Y, nearmiss_pred))
print('Fbeta score:', fbeta_score(Y, nearmiss_pred, beta=1.5))


# In[36]:


#Make ADASYN pipeline
ada_pipline = imb_pipline(ADASYN(), classifier)
ada_pred = cross_val_predict(ada_pipline, X, Y, cv=10)
print('Accuracy score:', metrics.accuracy_score(Y, ada_pred))
print('RMSE:', round(sqrt(mean_squared_error(Y, ada_pred)), 2))
print('R_squared:', round(r2_score(Y, ada_pred), 2))
print('Recall score:', metrics.recall_score(Y, ada_pred))
print('Fbeta score:', fbeta_score(Y, ada_pred, beta=1.5))


# In[37]:


#Make AllKNN pipeline
alknn_pipline = imb_pipline(AllKNN(), classifier)
alknn_pred = cross_val_predict(alknn_pipline, X, Y, cv=10)
print('Accuracy score:', metrics.accuracy_score(Y, alknn_pred))
print('RMSE:', round(sqrt(mean_squared_error(Y, alknn_pred)), 2))
print('R_squared:', round(r2_score(Y, alknn_pred), 2))
print('Recall score:', metrics.recall_score(Y, alknn_pred))
print('Fbeta score:', fbeta_score(Y, alknn_pred, beta=1.5))


# In[38]:


#Make CondensedNearestNeighbour pipeline
CN_pipline = imb_pipline(CondensedNearestNeighbour(), classifier)
CN_pred = cross_val_predict(CN_pipline, X, Y, cv=10)
print('Accuracy score:', metrics.accuracy_score(Y, CN_pred))
print('RMSE:', round(sqrt(mean_squared_error(Y, CN_pred)), 2))
print('R_squared:', round(r2_score(Y, CN_pred), 2))
print('Recall score:', metrics.recall_score(Y, CN_pred))
print('Fbeta score:', fbeta_score(Y, CN_pred, beta=1.5))


# In[ ]:


#Make EditedNearestNeighbours pipeline
ENN_pipline = imb_pipline(EditedNearestNeighbours(), classifier)
ENN_pred = cross_val_predict(ENN_pipline, X, Y, cv=10)
print('Accuracy score:', metrics.accuracy_score(Y, ENN_pred))
print('RMSE:', round(sqrt(mean_squared_error(Y, ENN_pred)), 2))
print('R_squared:', round(r2_score(Y, ENN_pred), 2))
print('Recall score:', metrics.recall_score(Y, ENN_pred))
print('Fbeta score:', fbeta_score(Y, ENN_pred, beta=1.5))


# In[50]:


kf = KFold(n_splits=10)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
X_train, Y_train = CondensedNearestNeighbour().fit_resample(x_train, y_train)
accuracy = []
precision = []
recall = []
fbeta = []
for train, test in kf.split(X_train, Y_train):
    pipline = make_pipeline(classifier)
    model = pipline.fit(X_train[train], Y_train[train])
    prediction = model.predict(X_train[test])
    
    accuracy.append(pipline.score(X_train[test], Y_train[test]))
    precision.append(precision_score(Y_train[test], prediction))
    recall.append(recall_score(Y_train[test], prediction))
    fbeta.append(fbeta_score(Y_train[test], prediction, beta=1.5))


# In[51]:


print(np.mean(accuracy))
print(np.mean(precision))
print(np.mean(recall))
print(np.mean(fbeta))


# In[209]:


def store_csv(prediction, filename):
    sub = pd.read_csv('../submission.csv')
    new = {'PerStatus':prediction}
    sub.update(new)
    sub.to_csv(filename, index=False)


# In[210]:


test = pd.read_csv('../E_data/stest.csv')
test.drop(['Unnamed: 0', 'PerStatus', 'PerNo', 'yyyy'], axis=1, inplace=True)
test.ffill(inplace=True)
test.fillna(0, inplace=True)


# In[211]:


test = preprocessing.scale(test)
result = model.predict(test)


# In[212]:


result.sum()


# In[213]:


store_csv(result, 'smote_BN.csv')


# In[ ]:





# In[ ]:




