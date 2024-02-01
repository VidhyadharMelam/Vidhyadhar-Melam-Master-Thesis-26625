#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing necessary libraries
import numpy as np
import matplotlib.pyplot as mplt
import pandas as pd
import seaborn as sns
import matplotlib.style as style
import pickle


# In[2]:


ckd_df = pd.read_csv('Final_kidney_disease_complete_cleaned.csv')


# In[3]:


#Diplay first five records
ckd_df.head()


# In[4]:


x = ckd_df.drop("class", axis=1)
y = ckd_df["class"]


# In[5]:


# Import train test split
from sklearn.model_selection import train_test_split
# train test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[6]:


# Shape of the dataset
print(x_train.shape)
print(x_test.shape)


# In[7]:


# Importing metrics used for evaluation of our models
from sklearn import metrics
from sklearn.metrics import classification_report


# In[8]:


# Chi Square
from sklearn.feature_selection import chi2
import scipy.stats as stats
from scipy.stats import zscore


# In[9]:


# Hyperparameter tuner and Cross Validation
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


# # Decision Tree

# In[10]:


# import DecisionTreeClassifer
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

#number of features to be considered at every split
max_features = ['auto', 'sqrt', 'log2']

#maximum number levels in the tree
max_depth = [3,5,7,10]

#Splitter
splitter = ['best', 'random']

#number of samples
max_samples = [0.5,0.75,1.0]

#minimum number of samples need to split a node
min_samples_split = [1,2,3,5,7]

#minimum number of samples need at each leaf node
min_samples_leaf = [1,2,3,5,7]

criterion = ['gini', 'entropy', 'log_loss']


# In[11]:


#import GridSearchCV

from sklearn.model_selection import GridSearchCV

param_grid = {'max_features': max_features,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf,
              'criterion': criterion
              }

print(param_grid)


# In[12]:


from sklearn.model_selection import GridSearchCV
Decision_grid = GridSearchCV(estimator = dtc,
                                 param_grid = param_grid, # using all parameters
                                 cv = 5, #train each randomforest 5 times
                                 verbose = 2, #can see th eoutput
                                 n_jobs = 1) #usig all cores of machines to make process faster


# In[13]:


Decision_grid.fit(x_train, y_train)


# In[14]:


# import DecisionTreeClassifer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

dtc = DecisionTreeClassifier(
    criterion = 'entropy',
    max_depth = 10,
    #splitter = 'best',
    min_samples_leaf = 2,
    min_samples_split = 2,
    max_features = 'log2'
    )
dtc_ACCURACY = DecisionTreeClassifier()
dtc.fit(x_train,y_train)

# accuracy score, confusion matrix and classification report of DECISIONTREE

dtc_ACCURACY = accuracy_score(y_test, dtc.predict(x_test))

ACC = accuracy_score(y_train, dtc.predict(x_train))
print(f'Training Accuracy of DECISION TREE CLASSIFIER is {ACC}')

print(f'Testing Accuarcy of DECISION TREE CLASSFIER is {dtc_ACCURACY} \n')

CONF = confusion_matrix(y_test, dtc.predict(x_test))
print(f'Confusion Matrix  \n {CONF} \n')

CLASS = classification_report(y_test, dtc.predict(x_test))
print(f'Classification Report \n\n {CLASS} \n\n')


# In[15]:


# Creating a pickle file for the classifier
filename = 'Kidney.pkl'
pickle.dump(dtc, open(filename, 'wb'))


# In[ ]:




