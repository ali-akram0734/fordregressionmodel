#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import statsmodels.api as stats
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[31]:


df = pd.read_csv('/Users/aliakram/Downloads/ford.csv')


# In[32]:


df.head()


# In[33]:


df.isnull().sum()


# In[57]:


y = df['price']
x = df[['year','mileage','engineSize']]


# In[35]:


df.describe()


# In[36]:


fig=plt.figure(figsize=(15,8))
sns.heatmap(df.corr(), linewidths=3, annot=True)


# In[37]:


sns.lmplot(x="price", y="engineSize", line_kws={"color":"r"}, data=df, aspect = 2, height = 7)
plt.title("price vs engineSize", fontweight="bold")


# In[ ]:





# In[10]:


sns.lmplot(x="price", y="year", line_kws={"color":"r"}, data=df, aspect = 2, height = 7)
plt.title("price vs year", fontweight="bold")


# In[22]:


sns.lmplot(x="price", y="mpg", line_kws={"color":"r"}, data=df, aspect = 2, height = 7)
plt.title("price vs mpg", fontweight="bold")


# In[23]:


sns.lmplot(x="price", y="tax", line_kws={"color":"r"}, data=df, aspect = 2, height = 7)
plt.title("price vs tax", fontweight="bold")


# In[ ]:





# In[11]:


df.info()


# In[39]:


gear = pd.get_dummies(df['transmission'])
fuel = pd.get_dummies(df['fuelType'])
mod = pd.get_dummies(df['model'])


# In[40]:


df = pd.concat([df,gear,fuel,mod], axis = 1)
df.head()


# In[41]:


df = df.drop(['transmission','fuelType','model'], axis = 1)


# In[42]:


df.head()


# In[63]:


y = df['price']
x = df.drop('price', axis = 1)


# In[69]:


df.shape[0]


# In[70]:


df.shape[1]


# In[64]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
LR=LinearRegression()
LR.fit(x_train,y_train)


# In[65]:


LR=LinearRegression()
LR.fit(x_train,y_train)
y_prediction = LR.predict(x_test)


# In[66]:


test_set_rmse = (np.sqrt(mean_squared_error(y_test, y_prediction)))
test_set_r2 =r2_score(y_test, y_prediction)


# In[67]:


test_set_rmse


# In[68]:


test_set_r2


# In[56]:


formula='price ~ C(year) + C(mpg) + C(tax) + C(engineSize)'
model=ols(formula, df).fit()
print(np.round(anova_lm(model, typ=2),3))
print(model.summary())
if np.round(model.f_pvalue,2)<0.05:
    print("Reject Null Hypothesis and accept the alternate hypothesis")
else:
    print("Accept the Null Hypothesis")


# In[ ]:




