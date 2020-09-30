#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: eman-saeed
"""
#KNN -> K Nearest Neigbors
#import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap 
#import the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

#in the X dataset we neglect the UserID and Gender.
X = dataset.iloc[:,[2,3]].values
Y = dataset.iloc[:,4].values

#Splitting the dataset.
X_train ,X_test , Y_train , Y_test = train_test_split(X, Y , test_size = 0.25 , random_state=0)

"""
# 1-Features Scalling.
- we need  to apply the feature scalling becouse we need accurate prediction,
  as we need to predict which users are going to targeted.
"""
sc_x    = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test  =sc_x.fit_transform(X_test) 

"""
2- Fitting the K-NN to the Training set.
   2.1 we import the KNeighborsClassifierlibrary .
       2.1.1 the most important arguments that we take care:
           1- k neighbors numbers .
           2- p is the power parameter.
           3- metric is the distance based on distance algorithm.
   2.2 use the fit function to fit with the training set.
   2.3 predicting the Test set Result by using the predict function.
     
"""
#Fitting Logistic regression to the Training set.
classifier = KNeighborsClassifier(n_neighbors = 10 , metric = 'minkowski', p=2)
classifier.fit(X_train, Y_train)
#Predicting the Test set Results.
  
y_predict = classifier.predict(X_test)

"""
3- Evaluating the model by using Confusing matrix : 
    3.1 importing the class confusion_matrix from sklearn.metrics import  .
    3.2 create a object cm from a confusion_matrix
    
"""
cm = confusion_matrix(Y_test,y_predict)

"""
4- Visualising the Trainig set results
    4.1 Visulizing the Trainng set results.
    4.2 Visulizing the test set Results.
"""
X_set, y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('KNN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('KNN(Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()