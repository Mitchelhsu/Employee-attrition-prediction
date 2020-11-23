#!/usr/bin/env python
# coding: utf-8

# In[152]:


import pandas as pd
import numpy as np
import random as rd
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[153]:


data = pd.read_csv('../p_train.csv')
st = pd.read_csv('../PerStatus.csv')


# In[154]:


scaled_data = preprocessing.scale(data)


# In[155]:


pca = PCA(n_components=25)
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)


# In[156]:


pca_DF = pd.DataFrame(data=pca_data, columns=['pca' + str(x) for x in range(1, 26)])
pca_DF


# In[157]:


X = pca_DF.values
Y = st.iloc[:, 0].values


# In[158]:


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


# In[159]:


#Make pipeline of clf
clf_pipeline = make_pipeline(StandardScaler(), RandomForestClassifier())
y_pred = cross_val_predict(clf_pipeline, X, Y, cv=10)
print('Accuracy score:', metrics.accuracy_score(Y, y_pred))
print('RMSE:', round(sqrt(mean_squared_error(Y, y_pred)), 2))
print('R_squared:', round(r2_score(Y, y_pred), 2))
print('Recall score:', metrics.recall_score(Y, y_pred))
print('Fbeta score:', fbeta_score(Y, y_pred, beta=1.5))


# In[ ]:




