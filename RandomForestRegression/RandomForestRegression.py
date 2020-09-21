#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: eman-saeed
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from  sklearn.preprocessing import StandardScaler


#import the dataset 
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values


#Fitting the Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(X,y)

#Predicting a new result
y_pred = regressor.predict([[6.5]])

#Visualizing the Regression result
plt.scatter(X,y , color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

