#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.ticker import MaxNLocator
import math
import seaborn as sns


# In[6]:


df = pd.read_csv('p_train.csv')
st = pd.read_csv('PerStatus.csv')
df


# In[4]:


for i in range(len(df)):
    if df.loc[i, :].isna().sum() == 44:
        df.loc[i, 'PerNo'] = 0


# In[7]:


attri = []
for atr in df:
    attri.append(atr)
    
attri


# In[128]:


stat = data.loc[:, 'PerStatus']
perno = data.loc[:, 'PerNo']
year = data.loc[:, 'yyyy']


# In[22]:


sex = data.loc[:, 'sex']

sex.fillna('ffill')

res_f = 0
res_m = 0
for (s, st) in zip(sex, stat):
    if s == 0 and st == 1:
        res_f += 1
    elif s == 1 and st == 1:
        res_m += 1
        
print(res_f, res_m)


# In[23]:


plt.barh(['Res_Female', 'Res_Male'], [res_f, res_m], color=['pink', 'navy'])
plt.xlabel('Number of Resignation')
plt.ylabel('Gender')

plt.tight_layout()
plt.show()


# In[24]:


year_v = {2014:0, 2015:0, 2016:0, 2017:0}

for i, s in enumerate(stat):
    if s == 1:
        year_v[year[i]] += 1
        
key = [int(y) for y in year_v]
val = [int(v) for v in year_v.values()]
plt.barh(key, val)
print(year_v)


# In[25]:


wp = data.loc[:, '廠區代碼']

wp = wp.ffill().astype(int)

s = data.pivot_table(index = '廠區代碼', aggfunc='size')
wps = []
for val in s.keys().astype(int):
    wps.append(val)

wp_cnt = dict.fromkeys(wps, 0)

for i, s in enumerate(stat):
    if s == 1:
        wp_cnt[int(wp[i])] += 1
        
key = [int(k) for k in wp_cnt.keys()]
val = [int(val) for val in wp_cnt.values()]

plt.barh(key, val)
plt.xlabel('# of resignation')
plt.ylabel('Work place')
plt.show()
print(wp_cnt)


# In[29]:


rank = data.loc[:, '職等']
rank = rank.ffill().astype(int)


# In[247]:


res = pd.DataFrame()

for i, s in enumerate(stat):
    if s == 1:
        res = res.append(df.loc[i, :])
        
print(len(res))


# In[236]:


res.describe()


# In[237]:


res.iloc[:, 2:14].describe()


# In[238]:


res.iloc[:, 14:26].describe()


# In[239]:


res.iloc[:, 26:38].describe()


# In[240]:


res.iloc[:, 38:48].describe()


# In[7]:


staying_rate = (df['PerStatus'].value_counts()[0] - df['PerStatus'].value_counts()[1]) / df['PerStatus'].value_counts()[0]
staying_rate


# In[8]:


#for a in attri:
    #fig = sns.countplot(x=a, hue='PerStatus', data=df)
    #fig.figure.savefig('plots/{}_bar.png'.format(a))


# In[9]:


plt.figure(figsize=(50, 50))
fig = sns.heatmap(df.corr(), annot=True, fmt='.1%')
fig.figure.savefig('plots/heatmap.png')


# In[6]:


sns.countplot(x='yyyy', hue='PerStatus', data=df)


# In[10]:


#sns.relplot(x='出差數B', y='PerStatus',kind='line', data=df)


# In[11]:


new = df.T.append(st.T).T


# In[19]:


abs(new.corr()['PerStatus']).sort_values()


# In[ ]:




