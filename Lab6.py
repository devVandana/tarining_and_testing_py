#!/usr/bin/env python
# coding: utf-8

# # Lab 7

# ## Imports
# Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Import Dataset
# 
# 
# Datafile Name: Enrollment Forecast
# 
# Number of cases: 29
# Variable Names:
# 
# * 1.YEAR: 1961 = 1, 1989 = 29
# * 2.ROLL: Fall undergraduate enrollment
# * 3.UNEM: January unemployment rate (%) for New Mexico
# * 4.HGRAD: Spring high schoolgraduates in New Mexico
# * 5.INC: Per capita income in Albuquerque (1961 dollars)

# In[5]:


data = pd.read_csv('C:/Users/Hp/Downloads/enrollment_forecast.csv')


# **Check the head of customers, and check out its info() and describe() methods.**

# In[6]:


data.head(5)


# In[3]:


data.info()


# In[4]:


data.describe()


# In[29]:





# In[30]:





# In[31]:





# In[32]:





# In[7]:


sns.pairplot(data)


# ## Apply Training and Testing algorithm
# 
# 
#  X equal to the numerical features of the customers and a variable y equal to the "roll" column.

# In[54]:


X = data[['year','unem','hgrad','inc']]
y = data['roll']


# In[55]:


from sklearn.model_selection import train_test_split


# In[56]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)


# In[57]:


from sklearn.linear_model import LinearRegression


# In[58]:


lr = LinearRegression()


# In[59]:


lr.fit(X_train,y_train)


# ## Predicting Test Data
# 

# In[60]:


predict = lr.predict(X_test)


# In[61]:


plt.scatter(y_test,predict)


# In[63]:


sns.distplot((y_test),bins=10);


# ## Evaluating the Model
# 
# Let's evaluate our model performance by calculating the residual sum of squares and the explained variance score (R^2).
# 
# ** Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error. Refer to the lecture or to Wikipedia for the formulas**

# In[51]:


from sklearn import metrics


# In[52]:


print('MAE:', metrics.mean_absolute_error(y_test, predict))
print('MSE:', metrics.mean_squared_error(y_test, predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predict)))


# In[ ]:




