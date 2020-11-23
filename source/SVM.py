#!/usr/bin/env python
# coding: utf-8

# In[25]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.utils import shuffle
sns.set()


# # Load Preprocessed Data

# In[44]:


dataset = pd.read_csv(os.path.join(os.pardir, 'p_season.csv'))
target = pd.read_csv(os.path.join(os.pardir, 'test_season.csv'))
target = target.iloc[:,0]


# In[45]:


dataset.drop(['Unnamed: 0', 'periodQ'], axis=1, inplace=True)
target = dataset.pop('PerStatus')


# In[46]:


dataset.shape


# In[47]:


#Shuffle data&target
dataset, target = shuffle(dataset, target)


# # SVM

# In[48]:


from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

kfold = KFold(10, True)
predicted = []
expected = []

for train, test in kfold.split(dataset):
    x_train = dataset.iloc[train]
    y_train = target.iloc[train]
    x_test = dataset.iloc[test]
    y_test = target.iloc[test]
    svm = OneVsOneClassifier(SVC(gamma='scale')).fit(x_train, y_train)
    expected.extend(y_test)
    predicted.extend(svm.predict(x_test))


# In[49]:


print('Macro-average: {0}'.format(metrics.f1_score(expected, predicted, average='macro')))
print('Micro-average: {0}'.format(metrics.f1_score(expected, predicted, average='micro')))
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
accuracy = metrics.accuracy_score(expected, predicted)
print('Accuracy: %.2f%%' % (accuracy*100))

print('\n')
print('precision:', metrics.precision_score(expected, predicted))
print('recall:', metrics.recall_score(expected, predicted))

print('Average = macro')
print('precision:', metrics.precision_score(expected, predicted, average='macro'))
print('recall:', metrics.recall_score(expected, predicted, average='macro'))
print('F1-score:', metrics.f1_score(expected, predicted, average='macro'))

print('\n')
print('Average = micro')
print('precision:', metrics.precision_score(expected, predicted, average='micro'))
print('recall:', metrics.recall_score(expected, predicted, average='micro'))
print('F1-score:', metrics.f1_score(expected, predicted, average='micro'))

print('\n')
print('Average = weighted')
print('precision:', metrics.precision_score(expected, predicted, average='weighted'))
print('recall:', metrics.recall_score(expected, predicted, average='micro'))
print('F1-score:', metrics.f1_score(expected, predicted, average='weighted'))

print('\n')
print('Fbeta score:', metrics.fbeta_score(expected, predicted, beta=1.5))
print('F1-score:', metrics.f1_score(expected, predicted))


# In[ ]:





# In[ ]:




