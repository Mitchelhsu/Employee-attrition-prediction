#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

# In[2]:


dataset = pd.read_csv(os.path.join(os.pardir, 'p_train.csv'))
target = pd.read_csv(os.path.join(os.pardir, 'PerStatus.csv'))
target = target.iloc[:,0]


# In[3]:


#Shuffle data&target
dataset, target = shuffle(dataset, target)


# # NaiveBayes

# In[4]:


from sklearn.naive_bayes import GaussianNB

kfold = KFold(10, True)
predicted = []
expected = []

for train, test in kfold.split(dataset):
    x_train = dataset.iloc[train]
    y_train = target.iloc[train]
    x_test = dataset.iloc[test]
    y_test = target.iloc[test]
    nb = GaussianNB()
    nb.fit(x_train, y_train)
    expected.extend(y_test)
    predicted.extend(nb.predict(x_test))


# In[6]:


print('Macro-average: {0}'.format(metrics.f1_score(expected, predicted, average='macro')))
print('Micro-average: {0}'.format(metrics.f1_score(expected, predicted, average='micro')))
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
accuracy = metrics.accuracy_score(expected, predicted)
print('Accuracy: %.2f%%' % (accuracy*100))

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


# In[ ]:




