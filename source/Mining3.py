#!/usr/bin/env python
# coding: utf-8

# In[112]:


import pandas as pd
import numpy as np


# In[113]:


df = pd.read_csv('E_data/train.csv')


# In[114]:


df.ffill(inplace=True)
df.fillna(0, inplace=True)


# In[115]:


no = df.PerNo.value_counts().index
no


# In[116]:


new = pd.DataFrame()
for n in no:
    group = df.groupby(by='PerNo')
    cur = group.get_group(n)
    new = new.append(cur[cur.yyyy == cur.yyyy.max()])


# In[117]:


new


# In[118]:


new.index = range(len(new))


# In[119]:


lst = new.pop('PerStatus')
new.drop('PerNo', axis=1, inplace=True)


# In[120]:


new.to_csv('last.csv', index=False)
lst.to_csv('last_Status.csv', index=False)


# In[121]:


new


# In[122]:


lst


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




