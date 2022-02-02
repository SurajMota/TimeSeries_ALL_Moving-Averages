#!/usr/bin/env python
# coding: utf-8

# In[57]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from datetime import datetime
import plotly.express as px

mpl.rcParams['figure.figsize'] = (10,8)
mpl.rcParams['axes.grid'] = False


# In[58]:


df = pd.read_csv("https://raw.githubusercontent.com/srivatsan88/YouTubeLI/master/dataset/nyc_energy_consumption.csv")
df.head()


# In[59]:


# convert timeStamp from string to datetime64 type
df['timeStamp'] = pd.to_datetime(df['timeStamp'])


# In[60]:


df.info()


# In[61]:


df.tail()


# In[62]:


fig = px.line(df, x='timeStamp', y='demand', title='New York Energy demand')

fig.update_xaxes(rangeslider_visible=True, 
                 rangeselector=dict(
                 buttons=list([
                     dict(count=1, label="1y", step="year", stepmode="backward"),
                     dict(count=2, label="3y", step="year", stepmode="backward"),
                     dict(count=3, label="5y", step="year", stepmode="backward"),
                     dict(step="all")
                 ])
                 
                 ))
fig.show()


# In[63]:


## setting timeStamp as index
ny_df = df.set_index('timeStamp')


# In[64]:


# graph of all columns
ny_df.plot(subplots=True)


# In[65]:


print("Rows    :",df.shape[0])
print("Columns :",df.shape[1])
print("\nMissing :",df.isnull().sum())
print()
print("\nUnique  :",df.nunique)


# In[66]:


# It will show all nan values in demand columny
df.query('demand != demand')


# In[67]:


# I am filling with forward fill. It will take the past value and fill the data
df['demand'] = df['demand'].fillna(method='ffill')
df['temp'] = df['temp'].fillna(method='ffill')


# In[68]:


print("\nMissing values :",df.isnull().sum())


# In[69]:


ny_df = df.set_index('timeStamp')


# In[70]:


# graph of all columns (After imputing missing values)
ny_df.plot(subplots=True)


# In[71]:


ny_df.resample('M').mean()


# In[72]:


ny_df.resample('M').mean().plot(subplots=True)


# In[73]:


ny_df_monthly = ny_df.resample('M').mean()


# In[74]:


ny_df_monthly.head()


# In[75]:


get_ipython().system('pip install pmdarima')


# In[76]:


## (auto_arima) is like grid search as per the values of range (p, q, r) which we have given
## (adf) is argumanted dickerfuller test. It checks the stationary of data.
## (stepwise) do random search rather than trying each and every combination
## lower value of AIC means its closer to actual data points
import pmdarima as pm

model = pm.auto_arima(ny_df_monthly['demand'],
                      m=12,
                      seasonal=True,
                      start_P=0,
                      start_q=0,
                      max_order=4,
                      test='adf',
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True,
                      trace=True)


# In[77]:


model.summary()


# In[78]:


ny_df_monthly


# In[84]:


## train test split

train = ny_df_monthly[(ny_df_monthly.index.get_level_values(0) >= '2012-01-31') & (ny_df_monthly.index.get_level_values(0) <= '2017-04-30')]
test = ny_df_monthly[(ny_df_monthly.index.get_level_values(0) > '2017-04-30')]


# In[85]:


test


# In[86]:


test.shape


# In[87]:


model.fit(train['demand'])


# In[90]:


# n_periods --> is predict 4 periods
forecast = model.predict(n_periods=4,return_conf_int=True)


# In[91]:


forecast


# In[92]:


forecast_df = pd.DataFrame(forecast[0],index=test.index,columns=['Prediction'])


# In[93]:


forecast_df


# In[94]:


pd.concat([ny_df_monthly['demand'],forecast_df],axis=1).plot()


# In[96]:


forecast1 = model.predict(n_periods=8, return_conf_int=True)


# In[97]:


forecast1


# In[98]:


forecast_range = pd.date_range(start='2017-05-31', periods=8, freq='M')


# In[99]:


forecast1_df = pd.DataFrame(forecast1[0],index=forecast_range,columns=['Prediction'])


# In[100]:


pd.concat([ny_df_monthly['demand'],forecast1_df],axis=1).plot()


# In[101]:


lower = pd.Series(forecast1[1][:,0], index=forecast_range)
upper = pd.Series(forecast1[1][:,1], index=forecast_range)


# In[103]:


plt.plot(ny_df_monthly['demand'])
plt.plot(forecast1_df, color='yellow')
plt.fill_between(forecast_range,
                lower,
                upper,
                color='k', 
                alpha=.15)


# In[104]:


out = model.plot_diagnostics()


# In[ ]:





# In[ ]:





# In[ ]:




