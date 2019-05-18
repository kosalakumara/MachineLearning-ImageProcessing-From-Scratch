#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv(r"C:\Users\nandika\Desktop\homeprices.csv")
df

get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('area')
plt.ylabel('price')

plt.scatter(df.area,df.price,color = 'red',marker = '+')

plt.show()

reg = linear_model.LinearRegression()
reg.fit(df[['area']],df.price)

reg.coef_

reg.intercept_

3300 * 135.78 + 180616

d = pd.read_csv(r"C:\Users\nandika\Desktop\area.csv")

d
d.head(3)

p = reg.predict(d)


d['prices'] = p #assign a new column for the data frame
d



# In[10]:


d.to_csv(r"C:\Users\nandika\Desktop\area.csv") #upload data to original csv file


# In[ ]:




