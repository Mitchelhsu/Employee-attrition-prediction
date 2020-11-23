#!/usr/bin/env python
# coding: utf-8

# In[169]:


import random
import pickle
import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB, BernoulliNB, CategoricalNB, MultinomialNB
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_val_predict
from sklearn.metrics import f1_score, precision_score, recall_score, mean_squared_error, r2_score
from sklearn import svm, tree
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import fbeta_score, make_scorer
from math import sqrt
import seaborn as sns

from joblib import dump
from collections import Counter


# In[170]:


df = pd.read_csv('p_train.csv')
st = pd.read_csv('PerStatus.csv')

data = df.T.append(st.T).T


# In[171]:


X = df.iloc[:, :df.shape[1]]
Y = st.iloc[:,0]


# In[163]:


#Make pipeline of clf
clf_pipeline = make_pipeline(StandardScaler(), tree.DecisionTreeClassifier())
y_pred = cross_val_predict(clf_pipeline, X, Y, cv=10)
print('Accuracy score:', metrics.accuracy_score(Y, y_pred))
print('RMSE:', round(sqrt(mean_squared_error(Y, y_pred)), 2))
print('R_squared:', round(r2_score(Y, y_pred), 2))
print('Recall score:', metrics.recall_score(Y, y_pred))
print('Fbeta score:', fbeta_score(Y, y_pred, beta=1.5))
print('F1-score:', f1_score(Y, y_pred))


# # Filter Features by Variance

# In[164]:


var = df.var()
idx = []
for i in range(len(var)):
    if var[i] < 0.75:
        print('{:50} {}'.format(var.index[i], var[i]))
        idx.append(var.index[i])


# In[165]:


print(X.shape)
X.drop(columns=idx, inplace=True)
print(X.shape)


# In[166]:


y_pred = cross_val_predict(clf_pipeline, X, Y, cv=10)
print('Accuracy score:', metrics.accuracy_score(Y, y_pred))
print('RMSE:', round(sqrt(mean_squared_error(Y, y_pred)), 2))
print('R_squared:', round(r2_score(Y, y_pred), 2))
print('Recall score:', metrics.recall_score(Y, y_pred))
print('Fbeta score:', fbeta_score(Y, y_pred, beta=1.5))
print('F1-score:', f1_score(Y, y_pred))


# # Filter Features by Correlation

# In[172]:


abs(data.corr()['PerStatus']).mean()


# In[173]:


vals = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
for val in vals:
    features = abs(data.corr()['PerStatus'][abs(data.corr()['PerStatus']) > val].drop('PerStatus')).index.tolist()
    
    testc = data.drop(columns='PerStatus')
    testc = X[features]
    
    print(features)
    
    y_pred = cross_val_predict(clf_pipeline, testc, Y, cv=10)
    print('Accuracy score:', metrics.accuracy_score(Y, y_pred))
    print('RMSE:', round(sqrt(mean_squared_error(Y, y_pred)), 2))
    print('R_squared:', round(r2_score(Y, y_pred), 2))
    print('Recall score:', metrics.recall_score(Y, y_pred))
    print('Fbeta score:', fbeta_score(Y, y_pred, beta=1.5))


# In[27]:


X = X[['yyyy', 'sex', 'Job classification', 'Job level', 'Work experience1', 'Work experience5', 'Project Hours', 'Project Numbers', 'Special project', 'Training hours B', 'Training hours C', 'Promotion speed', 'leave this three mon. A', 'leave this year A', 'leave this three mon. B', 'leave this year B', 'Business Trip A', 'Business Trip B', 'Annual performance C', 'Age level', 'Marital Status', 'Job tenure level A', 'Job tenure level B', 'Average working years', 'Graduated School', 'Family numbers']]


# # Sequential Feature Selector

# In[28]:


X.shape


# In[29]:


sfsl = SFS(clf_pipeline, 
           k_features = 26, 
           forward=True, 
           scoring=make_scorer(fbeta_score, beta=1.5), 
           cv=10)

sfsl.fit(X, Y)
sfsl.subsets_


# In[36]:


X = X[['yyyy',
   'Job classification',
   'Work experience5',
   'Special project',
   'Training hours B',
   'Training hours C',
   'leave this three mon. A',
   'leave this year A',
   'leave this three mon. B',
   'leave this year B',
   'Annual performance C',
   'Job tenure level A',
   'Job tenure level B',
   'Family numbers']]

y_pred = cross_val_predict(clf_pipeline, X, Y, cv=10)
print('Accuracy score:', metrics.accuracy_score(Y, y_pred))
print('RMSE:', round(sqrt(mean_squared_error(Y, y_pred)), 2))
print('R_squared:', round(r2_score(Y, y_pred), 2))
print('Precision:', metrics.precision_score(Y, y_pred))
print('Recall score:', metrics.recall_score(Y, y_pred))
print('Fbeta score:', fbeta_score(Y, y_pred, beta=1.5))
print('F1-score:', f1_score(Y, y_pred))


# In[159]:


plt.figure(figsize=(30, 30))
sns.heatmap(X[['yyyy', 'Factory code', 'Layers of management', 'Work experience5', 'Project Hours',
              'Working place', 'Training hours B', 'Number of leave in recent three months A', 
              'Number of leave in the past year B', 'Business Trip A', 'Annual performance level A',
              'Annual performance level C', 'Age level', 'Job tenure level A', 'Job tenure level C',
              'Average number of years of work before employment', 'Affiliated department']].corr(), 
             annot=True, fmt='.1%')


# In[ ]:


X['AD*FC'] = X['Affiliated department']*X['Factory code']
X['Al*JTA'] = X['Age level']*X['Job tenure level A']


# In[46]:


selected = X

y_pred = cross_val_predict(clf_pipeline, X, Y, cv=10)
print('Accuracy score:', metrics.accuracy_score(Y, y_pred))
print('RMSE:', round(sqrt(mean_squared_error(Y, y_pred)), 2))
print('R_squared:', round(r2_score(Y, y_pred), 2))
print('Recall score:', metrics.recall_score(Y, y_pred))
print('Fbeta score:', fbeta_score(Y, y_pred, beta=1.5))


# # Training

# In[85]:


selected = X

#Scaling
scaler = StandardScaler()
scaler.fit(selected)
selected = scaler.transform(selected)
print(selected)

#Splitting trainset
x_train, x_test, y_train, y_test = train_test_split(selected, Y, test_size=0.2)

#Classifier
model = ExtraTreesClassifier(max_depth=10, min_samples_leaf=5, min_samples_split=5)

#Cross_val_score(fbeta)
scoring = cross_val_score(model, selected, Y, cv=10, 
                         scoring=metrics.make_scorer(fbeta_score, beta=1.5))
print('Cross_val_score(mean):', scoring.mean())
print('Corss_val_score(std):', scoring.std())

#Fitting trainset
model = model.fit(x_train, y_train)
y_pred = model.predict(x_test)

#Showing Score
print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
print('Recall:', metrics.recall_score(y_test, y_pred))
print('fbeta:', metrics.fbeta_score(y_test, y_pred, beta=1.5))


# In[86]:


dump(model, 'last_ext_tunned.joblib')


# In[67]:


X


# In[56]:


st.sum()


# In[57]:


796/14392


# In[58]:


idx


# In[ ]:





# In[ ]:





# In[ ]:




