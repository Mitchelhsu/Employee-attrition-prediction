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

# In[3]:


dataset = pd.read_csv(os.path.join(os.pardir, 'p_train.csv'))
target = pd.read_csv(os.path.join(os.pardir, 'PerStatus.csv'))
target = target.iloc[:,0]


# In[4]:


#Shuffle data&target
dataset, target = shuffle(dataset, target)


# # DNN

# In[ ]:




