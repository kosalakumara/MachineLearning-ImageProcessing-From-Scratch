import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("homeprices.csv")
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,color = 'red',marker='+')
plt.show()


reg = linear_model.LinearRegression() #creating an object
reg.fit(df[['area']], df.price)

reg.predict(3400) #predict the relevent price for this area

reg.coef_  #shows the coeefficient

reg.intercept_ #showing the intercept
