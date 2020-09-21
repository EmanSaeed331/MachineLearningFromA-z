#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: eman-saeed
#Simple Linear Regression
"""
#importing librarieres

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from  sklearn.preprocessing import StandardScaler


#import the dataset 
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values


#Splitting the dataset
X_traing,X_test , y_train , y_test = train_test_split(X, y,test_size=0.3,random_state=0)
"""
SC_X = StandardScaler()
X_traing = SC_X.fit_transform(X_traing)
X_test = SC_X.fit_transform(X_test)
y_train = SC_X.fit_transform(y_train)
y_test = sc_X.fit_transform(y_test)
"""
#Fitting Simple Linear Regression to Training set.
"""
- import th LinearRegression class from sklearn.linear_model
"""
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(X_traing,y_train)

#Predicting the Test set results
y_pred = regressor.predict(X_test)



#Visualising the Training set results.
plt.scatter(X_traing,y_train, color='red' )
plt.plot(X_traing,regressor.predict(X_traing), color='blue')
plt.title('Salary  vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualizing the Test set results.

plt.scatter(X_test,y_test, color='blue' )
plt.plot(X_test,regressor.predict(X_test), color='green')
plt.title('Salary  vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()







