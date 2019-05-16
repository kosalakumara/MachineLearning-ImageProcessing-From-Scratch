import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("homeprices.csv")
plt.xlabel('price($)')
plt.ylabel('area')
plt.scatter(df.area,df.price,color = 'red',marker='+')
plt.show()
