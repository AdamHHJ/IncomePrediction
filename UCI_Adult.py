#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
# Data Loading
dataset = pd.read_csv('adult.csv')
dataset.info()


# In[2]:


# Data Pre-processing
dataset = dataset.fillna(np.nan)
# Drop the data I don't want to use
dataset.drop(labels=["workclass","fnlwgt", "education","occupation","relationship","race","native.country"], axis = 1, inplace = True)
# Reformat Column We Are Predicting: 0 means less than 50K. 1 means greater than 50K.
dataset['income']=dataset['income'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})
# Convert Sex value to 0 and 1
dataset["sex"] = dataset["sex"].map({"Male": 0, "Female":1})
# Create Married Column - Binary Yes(1) or No(0)
dataset["marital.status"] = dataset["marital.status"].replace(['Never-married','Divorced','Separated','Widowed'], 'Single')
dataset["marital.status"] = dataset["marital.status"].replace(['Married-civ-spouse','Married-spouse-absent','Married-AF-spouse'], 'Married')
dataset["marital.status"] = dataset["marital.status"].map({"Married":1, "Single":0})
dataset["marital.status"] = dataset["marital.status"].astype(int)
array = dataset.values
X = array[:,0:7]
Y = array[:,7]
print(X.shape)
print(Y.shape)


# In[5]:


# Data Splitting
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X,Y,train_size=0.7,random_state=2021,stratify=Y)
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)


# In[7]:


# logistic regression
# D1F
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=2021).fit(train_x,train_y)
print('prediction class')
print(lr.predict([test_x[2021]]))
print('prediciton probability')
print(lr.predict_proba([test_x[2021]]))


# In[8]:


# D2F
print("The training accuracy is ", lr.score(train_x, train_y))
print("The test accuracy is ", lr.score(test_x, test_y))


# In[10]:


from sklearn.metrics import precision_score, recall_score, f1_score
train_pred = lr.predict(train_x)
print("The precision on train set is ", precision_score(train_y, train_pred))
print("The recall on train set is ", recall_score(train_y, train_pred))
print("The f1 score on train set is ", f1_score(train_y, train_pred))
test_pred = lr.predict(test_x)
print("The precision on test set is ", precision_score(test_y, test_pred))
print("The recall on test set is ", recall_score(test_y, test_pred))
print("The f1 score on test set is ", f1_score(test_y, test_pred))


# In[12]:


# Support Vector Machine
# D1F
from sklearn.svm import SVC
svm = SVC(random_state=2021,probability=True).fit(train_x,train_y)
print('prediction class')
print(svm.predict([test_x[2021]]))
print('prediciton probability')
print(svm.predict_proba([test_x[2021]]))


# In[13]:


# D2F
print("The training accuracy is ", svm.score(train_x, train_y))
print("The test accuracy is ", svm.score(test_x, test_y))


# In[14]:


from sklearn.metrics import precision_score, recall_score, f1_score
train_pred = svm.predict(train_x)
print("The precision on train set is ", precision_score(train_y, train_pred))
print("The recall on train set is ", recall_score(train_y, train_pred))
print("The f1 score on train set is ", f1_score(train_y, train_pred))
test_pred = svm.predict(test_x)
print("The precision on test set is ", precision_score(test_y, test_pred))
print("The recall on test set is ", recall_score(test_y, test_pred))
print("The f1 score on test set is ", f1_score(test_y, test_pred))


# In[ ]:





# In[ ]:




