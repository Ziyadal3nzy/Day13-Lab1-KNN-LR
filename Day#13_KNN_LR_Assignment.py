#!/usr/bin/env python
# coding: utf-8

# ## Importing the libraries

# In[1]:


import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


# ## Loading the dataset

# In[2]:


dataset = pd.read_csv('Social_Network_Ads.csv')


# In[3]:


dataset.head()


# In[4]:


dataset.info()


# ## Define X by selecting only the age and EstimatedSalary, and y with purchased column

# In[5]:


X = dataset[['Age', 'EstimatedSalary']]


# In[6]:


X.head()


# In[7]:


y = dataset['Purchased']


# In[8]:


y.head()


# ## Print count of each label in Purchased column

# In[9]:


dataset.Purchased.value_counts().plot(kind="barh");
dataset.Purchased.value_counts()


# ## Print Correlation of each feature in the dataset

# In[10]:


cor = dataset.corr()
cor


# # First: Logistic Regression model

# ## Split the dataset into Training set and Test set with test_size = 0.25 and random_state = 0

# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# ## Train the model with random_state = 0

# In[12]:


TR = LogisticRegression()


# In[13]:


TR.fit(X_train,y_train)


# ## Print the prediction results

# In[14]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)


# In[15]:


predictions = TR.predict(X_test)


# In[16]:


predictions


# In[18]:


comp_result = pd.DataFrame(X_test)
comp_result['pred']= predictions
comp_result


# ## Create dataframe with the Actual Purchased and Predict Purchased

# In[19]:


test_dataset = pd.DataFrame(X_test, columns=['Age','EstimatedSalary'])


# In[20]:


test_dataset['Actual_Purchased'] = y_test


# In[22]:


test_dataset['Predict_Purchased'] = predictions


# In[23]:


test_dataset


# ## Print Confusion Matrix and classification_report

# In[25]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(classification_report(y_test,predictions))


# In[26]:


print(confusion_matrix(y_test,predictions))


# In[27]:


print(accuracy_score(y_test, predictions))


# In[33]:


plot_confusion_matrix(TR, X=X_test, y_true=y_test, cmap='Blues');


# ## Use StandardScaler() to improved performance and re-train your model

# In[34]:


sc = StandardScaler()


# In[35]:


X_train = sc.fit_transform(X_train)


# In[37]:


tr = LogisticRegression()


# In[38]:


tr.fit(X_train, y_train)


# In[40]:


print(classification_report(y_test, predictions))


# In[41]:


tr


# ## Try to Predicting a new result - e.g: person with Age = 30 and Salary = 90,000

# In[42]:


print(tr.predict([[30, 90000]]))


# ## Try to Predicting a new result - e.g: person with Age = 40 and Salary = 90,000

# In[43]:


print(tr.predict([[40, 90000]]))


# # Second: k-nearest neighbors model

# In[44]:


from sklearn.neighbors import KNeighborsClassifier


# In[45]:


knn = KNeighborsClassifier()


# In[46]:


knn.fit(X_train,y_train)


# In[49]:


print('class')
print('\n')
print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))

