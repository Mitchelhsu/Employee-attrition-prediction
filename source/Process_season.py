#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[2]:


df = pd.read_csv('E_data/season.csv')
train = pd.read_csv('E_data/train.csv')


# In[9]:


df['Number of leave A'].value_counts().index


# In[4]:


len(df.PerNo.value_counts().index)


# In[7]:


df.isna().sum()


# In[97]:


df.sort_values(by=['PerNo', 'periodQ'], inplace=True)
df.index = range(len(df))
df


# In[98]:


val = [0 for _ in range(72684)]
df.insert(2, 'PerStatus', val)
df


# In[99]:


af_per = []

for row in train.iterrows():
    row = row[1]
    if row[2] == 1:
        af_per.append(int(row[1]))


# In[100]:


group = df.groupby(['PerNo'])


# In[101]:


for i in range(len(df)):
    row = df.iloc[i]
    g = group.get_group(row.PerNo)
    p_max = g.periodQ.max()
    if row.PerNo in af_per and row.periodQ == p_max:
        df.at[i, 'PerStatus'] = 1


# In[113]:


df.PerStatus.sum()


# In[106]:


df.sort_values(by=['yyyy'], inplace=True)
df.index = range(len(df))
df


# In[110]:


test_season = df.drop([i for i in range(0, 57728)], axis=0)
test_season
df.drop([i for i in range(57728, len(df))], axis=0, inplace=True)


# In[115]:


test_season.index = range(len(test_season))
test_season


# In[116]:


df.to_csv('p_season.csv')
test_season.to_csv('test_season.csv')


# In[155]:


idx = []
for i in range(len(df)):
    row = df.iloc[i]
    if row.PerNo not in af_per:
        idx.append(i)
        
        
idx


# In[156]:


df.drop(index=idx, inplace=True)


# # Work overtime

# In[158]:


df = pd.read_csv('E_data/season.csv')
train = pd.read_csv('E_data/train.csv')


# In[159]:


for i in range(len(df)):
    if df.iloc[i].yyyy == 2018:
        df.drop([p for p in range(i, len(df))], inplace=True)
        break


# In[124]:


df.drop('periodQ', axis=1, inplace=True)


# In[125]:


no_group = df.groupby(['PerNo', 'yyyy']).sum()
no_group


# In[126]:


no_group.reset_index(inplace=True)
no_group


# In[127]:


no_group.drop(['Business Trip A', 'Business Trip B', 'Number of leave A', 'Number of leave B'], axis=1, inplace=True)


# In[128]:


train['Work Overtime'] = wo = pd.Series([np.nan for i in range(len(train))])


# In[133]:


train.set_index(['yyyy', 'PerNo'], inplace=True)
no_group.set_index(['yyyy', 'PerNo'], inplace=True)


# In[136]:


train.update(no_group)


# In[140]:


train.reset_index(inplace=True)


# In[219]:


train.info()


# In[143]:


train.to_csv('E_data/strain.csv', index=False)


# # WorkOvertime test

# In[205]:


df = pd.read_csv('E_data/season.csv')
test = pd.read_csv('E_data/test.csv')


# In[206]:


df.drop(['Business Trip A', 'Business Trip B', 'Number of leave A', 'Number of leave B', 'periodQ'], 
          axis=1, inplace=True)


# In[207]:


for i in range(len(df)):
    if df.iloc[i].yyyy == 2018:
        df = df.drop([p for p in range(0, i)])
        break


# In[208]:


group = df.groupby(['yyyy', 'PerNo']).sum()


# In[209]:


group.reset_index(inplace=True)


# In[216]:


test['Work Overtime'] = group.loc[:, 'Work Overtime']


# In[218]:


test.to_csv('E_data/stest.csv')


# In[220]:


test


# In[ ]:




