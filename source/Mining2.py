#!/usr/bin/env python
# coding: utf-8

# In[113]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[117]:


df = pd.read_csv('E_data/train.csv')
df.drop('PerNo', axis=1, inplace=True)
df.ffill(inplace=True)
df.fillna(0, inplace=True)


# In[122]:


def chart(feature, df):
    stayed = df[df['PerStatus'] == 0][feature].value_counts()
    left = df[df['PerStatus'] == 1][feature].value_counts()
    df = pd.DataFrame([left])
    df.index=['Left']
    df.plot(kind='bar', stacked=True, figsize=(10, 5))


# In[123]:


for col in df.columns:
    chart(col, df)
    plt.title(col)
    plt.show()


# In[126]:


df.drop(columns=['yyyy','Work experience5', 'Training hours C', 'Factory code', 'Total production', 
         'Number of leave in recent three months A', 'Number of leave in the past year A', 
         'Annual performance level A', 'Annual performance level B','Annual performance level C', 'Age level', 
         'Marital Status', 'Job tenure level A', 'Graduated Department category', 'Family numbers', 
         'Affiliated department'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




