#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler
from keras import models
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[23]:


def read(file):
    df = pd.read_csv(file)
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)
    df.drop('PerNo', axis=1, inplace=True)
    df.drop('PerStatus', axis=1, inplace=True)
    return df
    
def load_model(file):
    return load(file)

def load_keras(file):
    return models.load_model(file)

def store_csv(prediction, filename):
    sub = pd.read_csv('submission.csv')
    new = {'PerStatus':prediction}
    sub.update(new)
    sub.to_csv(filename, index=False)


# In[24]:


data = read('E_data/stest.csv')
model = load_keras('New_DNN3.h5')

data.drop(['yyyy', 'Unnamed: 0'], axis=1, inplace=True)
data


# In[25]:


data = data.values

scaler = StandardScaler()
scaler.fit(data)
test = scaler.transform(data)


# In[26]:


prediction = model.predict(test)
thresh = prediction.mean()*3/2

for i in range(len(prediction)):
    if prediction[i] > thresh:
        prediction[i] = 1
    else:
        prediction[i] = 0

store_csv(prediction[:,0], 'New_DNN3.csv')


# In[27]:


data.shape


# In[28]:


prediction.sum()


# In[29]:


prediction[:,0].sum()


# In[ ]:




