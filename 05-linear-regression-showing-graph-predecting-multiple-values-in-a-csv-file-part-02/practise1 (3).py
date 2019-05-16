#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[11]:


df = pd.read_csv(r"C:\Users\nandika\Desktop\homeprices.csv")


# In[12]:


df


# In[27]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('area')
plt.ylabel('price')


# In[14]:


plt.scatter(df.area,df.price,color = 'red',marker = '+')


# In[15]:


plt.show()


# In[20]:


reg = linear_model.LinearRegression()
reg.fit(df[['area']],df.price)


# In[30]:


reg.coef_


# In[32]:


reg.intercept_


# In[33]:


3300 * 135.78 + 180616


# In[34]:


d = pd.read_csv(r"C:\Users\nandika\Desktop\area.csv")


# In[35]:


d


# In[36]:


d.head(3)


# In[38]:


p = reg.predict(d)


# In[39]:


d['prices'] = p #assign a new column for the data frame
d


# In[ ]:




