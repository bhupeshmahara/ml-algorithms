#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
get_ipython().system('pip install seaborn')
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")


# In[2]:


df = load_boston()
df


# In[3]:


dataset = pd.DataFrame(df.data)
dataset.columns = df.feature_names
dataset.head()


# In[4]:


dataset["Price"] = df.target
dataset.head()


# In[5]:


# dividing the dataset into indepedent & dependent features

X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]


# In[6]:


## train test split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## Linear Regression

# In[7]:


from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score
mse = cross_val_score(linreg, X_train, y_train, scoring = 'neg_mean_squared_error', cv=5)
mean_mse = np.mean(mse)
print(mean_mse)


# ## Ridge Regression

# In[8]:


from sklearn.linear_model import Ridge
ridge = Ridge()

from sklearn.model_selection import GridSearchCV
params = {'alpha' : list(range(1,200))}
ridge_reg = GridSearchCV(ridge, params, scoring = 'neg_mean_squared_error', cv=10)
ridge_reg.fit(X_train, y_train)


# In[9]:


print(ridge_reg.best_params_)
print(ridge_reg.best_score_)


# ## Lasso Regresion

# In[10]:


from sklearn.linear_model import Lasso
lasso = Lasso()

from sklearn.model_selection import GridSearchCV
params = {'alpha' : [1e-15, 1e-10, 1e-5, 1e-3, 1e-2, 1e-1, 1, 2, 3, 4, 5, 10]}
lasso_reg = GridSearchCV(lasso, params, scoring = 'neg_mean_squared_error', cv=10)
lasso_reg.fit(X_train, y_train)


# In[11]:


print(lasso_reg.best_params_)
print(lasso_reg.best_score_)


# In[12]:


y_pred = linreg.predict(X_test)
from sklearn.metrics import r2_score

r2_score = r2_score(y_pred, y_test)
print(r2_score)

adjusted_r2 = (1 - (1 - r2_score)*(506 - 1))/(506 - 13 -1)
print(adjusted_r2)


# In[13]:


y_pred = ridge_reg.predict(X_test)
from sklearn.metrics import r2_score

r2_score = r2_score(y_pred, y_test)
print(r2_score)

adjusted_r2 = (1 - (1 - r2_score)*(506 - 1))/(506 - 13 -1)
print(adjusted_r2)


# In[14]:


y_pred = lasso_reg.predict(X_test)
from sklearn.metrics import r2_score

r2_score = r2_score(y_pred, y_test)
print(r2_score)

adjusted_r2 = (1 - (1 - r2_score)*(506 - 1))/(506 - 13 -1)
print(adjusted_r2)


# # Logistic Regression

# In[15]:


from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

df = load_breast_cancer()
dataset = pd.DataFrame(df.data, columns = df.feature_names)
dataset.head()


# In[16]:


dataset.shape


# In[17]:


# y = pd.DataFrame(df.target, columns = ['Target'])
dataset["Target"] = df.target
dataset.head()


# In[18]:


dataset['Target'].value_counts()


# In[19]:


## dependent & independent features

X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]


# In[20]:


## train test split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[21]:


params = [{'C' : [1,2,3,5,10]}, {'max_iter' : [100,150,200]}]


# In[22]:


model1 = LogisticRegression(C=100, max_iter=100)


# In[23]:


model = GridSearchCV(model1, param_grid=params, scoring = 'f1', cv=5)


# In[24]:


model.fit(X_train, y_train)


# In[25]:


model.best_params_


# In[26]:


model.best_score_


# In[27]:


y_pred = model.predict(X_test)
y_pred


# In[28]:


from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, accuracy_score
confusion_matrix(y_pred, y_test)


# In[29]:


accuracy_score(y_test, y_pred)


# In[30]:


print(classification_report(y_test, y_pred))


# ## Happy Learning

# In[ ]:




