#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import pickle
import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn import svm, tree
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import fbeta_score, make_scorer
import seaborn as sns
from joblib import dump
from collections import Counter
sns.set(font_scale=1)


# In[2]:


#Norm funciton
def normalize(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler.transform(data)

#Plotting feature importance
def plot_feature(fi):
    select = []
    idx = np.where(fi > np.percentile(fi, 65))[0]
    val = fi[fi > np.percentile(fi, 65)]
    for i in idx:
        select.append(features[i])
    
    plt.title('Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    
    imp = pd.DataFrame({'Feature_names':select, 
                 'Importance':val})
    plt.figure(figsize=(8, 5))
    sns_plot = sns.barplot(x='Importance', y="Feature_names", data=imp, color='b')
    fig = sns_plot.get_figure()
    fig.savefig('feature_importance.png')
    
    return imp


# In[3]:


df = pd.read_csv('p_train.csv')
ps = pd.read_csv('PerStatus.csv')


# In[4]:


#Getting values from DataFrame
X = df.iloc[:, :df.shape[1]].values
Y = ps.iloc[:,0].values
print(X.shape, Y.shape)

#Normalizing data
norm_X = normalize(X)


# In[5]:


#Getting all features
features = [col for col in df.columns]
print(len(features))


# # Random Forest

# In[6]:


#Splitting data
x_train, x_test, y_train, y_test = train_test_split(norm_X, Y, test_size=0.2)

rf_clf = RandomForestClassifier()

#Fitting model
rf_model = rf_clf.fit(norm_X, Y)

#Cross_val_score(fbeta)
scoring = cross_val_score(rf_model, x_train, y_train, cv=10, 
                         scoring=metrics.make_scorer(fbeta_score, beta=1.5))
print('Cross_val_score(mean):', scoring.mean())
print('Corss_val_score(std):', scoring.std())

#Fitting training set
rf_model = rf_model.fit(x_train, y_train)
y_pred = rf_model.predict(x_test)

#Prnting scores
print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
print('Recall:', metrics.recall_score(y_test, y_pred))
print('fbeta:', metrics.fbeta_score(y_test, y_pred, beta=1.5))
rf_importance = plot_feature(rf_model.feature_importances_)


# # Decision Tree

# In[12]:


#Splitting data
x_train, x_test, y_train, y_test = train_test_split(norm_X, Y, test_size=0.2)

dt_clf = tree.DecisionTreeClassifier(min_samples_split=10, min_samples_leaf=10)

#Fitting model
dt_model = dt_clf.fit(norm_X, Y)

#Cross_val_score(fbeta)
score = cross_val_score(dt_clf, norm_X, Y, cv=10, 
                        scoring=metrics.make_scorer(fbeta_score, beta=1.5))
print('Cross_val_score(mean):', score.mean())
print('Cross_val_score(std)', score.std())

#Ftting train set
dt_model.fit(x_train, y_train)
y_pred = dt_model.predict(x_test)

#Printing socres
print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
print('Recall:', metrics.recall_score(y_test, y_pred))
print('fbeta:', metrics.fbeta_score(y_test, y_pred, beta=1.5))
dt_importance = plot_feature(dt_model.feature_importances_)


# # Gradient boosting

# In[7]:


#Splitting data
x_train, x_test, y_train, y_test = train_test_split(norm_X, Y, test_size=0.2)

gb_clf = GradientBoostingClassifier()

#Fitting model
gb_model = gb_clf.fit(norm_X, Y)

#Cross val score(fbeta)
score = cross_val_score(gb_model, norm_X, Y, cv=10, 
                       scoring=make_scorer(fbeta_score, beta=1.5))
print('Cross val score(mean):', score.mean())
print('Cross val score(std):', score.std())

#Fitting train set
gb_model.fit(x_train, y_train)
y_pred = gb_model.predict(x_test)

#Printing socres
print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
print('Recall:', metrics.recall_score(y_test, y_pred))
print('fbeta:', metrics.fbeta_score(y_test, y_pred, beta=1.5))
gb_importance = plot_feature(gb_model.feature_importances_)


# # Extra Trees

# In[97]:


#Splitting data
x_train, x_test, y_train, y_test = train_test_split(norm_X, Y, test_size=0.2)

ext_clf = ExtraTreesClassifier()

#Fitting model
ext_model = ext_clf.fit(norm_X, Y)

#Cross val score(fbeta)
score = cross_val_score(ext_model, norm_X, Y, cv=10, 
                       scoring=make_scorer(fbeta_score, beta=1.5))
print('Cross val score(mean):', score.mean())
print('Cross val score(std):', score.std())

#Fitting train set
ext_model.fit(x_train, y_train)
y_pred = ext_model.predict(x_test)

#Printing socres
print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
print('Recall:', metrics.recall_score(y_test, y_pred))
print('fbeta:', metrics.fbeta_score(y_test, y_pred, beta=1.5))
ext_importance = plot_feature(ext_model.feature_importances_)


# # Feature importances

# In[98]:


f_list = []
for f in rf_importance.loc[:, 'Feature_names']:
    f_list.append(f)
    
for f in dt_importance.loc[:, 'Feature_names']:
    f_list.append(f)
    
for f in gb_importance.loc[:, 'Feature_names']:
    f_list.append(f)
    
for f in ext_importance.loc[:, 'Feature_names']:
    f_list.append(f)


# In[99]:


cnt = pd.DataFrame.from_dict(dict(Counter(f_list)), orient='index', columns=['Frequent'])
cnt = cnt.sort_values(by='Frequent', ascending=False)

cnt


# In[44]:


selected = cnt.index


# In[134]:


for i in range(len(df.var())):
    if df.var()[i] > 2:
        print(df.var().index[i] + ': ' + str(df.var()[i]))


# In[131]:





# In[8]:


select = []
idx = np.where(gb_model.feature_importances_ > np.percentile(gb_model.feature_importances_, 65))[0]
val = rf_model.feature_importances_[gb_model.feature_importances_ > np.percentile(gb_model.feature_importances_, 65)]
for i in idx:
    select.append(features[i])


# In[9]:


len(select)


# In[10]:


select


# In[ ]:





# In[ ]:





# In[ ]:




