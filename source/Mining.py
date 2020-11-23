#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# In[2]:


df = pd.read_csv('data/train.csv')
df


# In[3]:


df.info()


# In[7]:


len(df['Job tenure level A'].value_counts().index)


# In[6]:


df.sex.isna().sum()/14392


# In[3]:


len(df.PerNo.value_counts().index)


# In[13]:


df.isna().sum().sum()/(14392*48)


# In[30]:


pernos = df.PerNo.value_counts().index
no_group = df.groupby('PerNo')
for n in pernos:
    group = no_group.get_group(n)
    if group.sex.isna().any():
        print(group)
        break
        
no_group.get_group(7601)


# In[18]:


df.info()


# In[4]:


ps = df.pop('PerStatus')


# In[5]:


pn = df.pop('PerNo')


# In[6]:


#df.drop('highest education level', axis=1, inplace=True)


# In[7]:


df.fillna(method='ffill', inplace=True)
df.fillna(0, inplace=True)


# In[8]:


wo = pd.read_csv('p_train.csv')
wo['Work Overtime']
df['Work Overtime'] = wo['Work Overtime']


# In[9]:


df


# In[10]:


df.to_csv('p_train.csv', index=False)


# In[79]:


ps.to_csv('PerStatus.csv', index=False)


# In[80]:


pn.to_csv('PerNo.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:




