#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import numpy
import pandas
import matplotlib
import seaborn
import sklearn

print('Python: {}'.format(sys.version))
print('Numpy: {}'.format(numpy.__version__))
print('Pandas: {}'.format(pandas.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Seaborn: {}'.format(seaborn.__version__))
print('Sklearn: {}'.format(sklearn.__version__))


# In[3]:


#import necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


#load the dataset from the csv file using pandas
data = pd.read_csv('creditcard.csv')


# In[6]:


#explore the dataset
print(data.columns)


# In[7]:


print(data.shape)


# In[8]:


print(data.describe())


# In[9]:


#since this is a large data set to save time and computational requirements we are using only fraction of a dataset i.e 10%
data = data.sample(frac = 0.1, random_state = 1)

print(data.shape)


# In[10]:


#plot histogram of each parameter
data.hist(figsize = (20, 20))
plt.show()


# In[11]:


#dtermine number of fraud cases in dataset
Fraud = data[data['Class'] == 1]
Valid = data[data['Class'] == 0]

outlier_fraction = len(Fraud) / float(len(Valid))
print(outlier_fraction)

print('Fraud Cases {}'.format(len(Fraud)))
print('Valid Cases {}'.format(len(Valid)))


# In[12]:


#correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize = (12,9))

sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()


# In[14]:


#Get all the columns from dataframe
columns = data.columns.tolist()

#filter the columns to remove the data we dont want
columns = [c for c in columns if c not in ["Class"]]

#Show the variable we'll be predictiong on 
target = "Class"

X = data[columns]
Y = data[target]

#print the shapes on X and Y
print(X.shape)
print(Y.shape)


# In[20]:


from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# define a random state
state = 1

# define the outlier detection methods
classifiers = {
    "Isolation Forest": IsolationForest(max_samples=len(X),
                                       contamination = outlier_fraction,
                                       random_state = state),
    "Local Outlier Factor": LocalOutlierFactor(
    n_neighbors = 20,
    contamination = outlier_fraction)
}


# In[23]:


# Fit the model
n_outliers = len(Fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):
    
    # fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
        
    # reshape the prediction values to 0 for valid and 1 to fraud
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    
    n_errors = (y_pred != Y).sum()
    
    # Run Classification metrics
    print('{}: {}'.format(clf_name, n_errors))
    print(accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))


# In[ ]:




