#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: eman-saeed
"""

# K-mean clustering 
"""
1- import libraries.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
"""
- we have to segment the client to a diffrent groups based on the annual income and the spending score.
- the main idea is know the number of k-mean to the clusters.
- so we have to know the optimal number of clusters so we will use the elbow method.
"""
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values
"""
- using the elbow method to find the optimal number of clusters:
    1-we import the k-mean class from sklearn.
    2-plot the elbow graph by: 
        - computing the within cluster sum of square fot 10 diffrent numbers of clusters,
          so since we're going to have 10 iterations we are going to write a full loop to create a list of the 10 diffrent within cluster sum of squares for the numbers of the clusters.
        - start by initializing this list we call wcss.
    3- in each iteration in this loop we are going to do two things: 
        1- fit the K-mean algorithm to our data x.
        2- compute the within cluster sum of squares and append to our WCSS list.
- Kmeans class has several parameter : 
   1- n_clusters -> the number of clusters .
   2- init       -> the random intialization method .
   3- max_iter   -> there can be to find the final clusters when the K-mean algorithm is running ,
                     the deafult value for this parameter is 300.
   4- n_init     -> the number of times the k-means algorithm will be run with diffrent initial centroid, 
                     the default value for this parameter is 10.
   5- random_state -> fix all the random factors of the k-means process.
- wcss says there is actually another name for the sum of squares called -> inertia.
- inertia attribute : do with the cluster of squares.
- plotting we make in x-axis the range from 1~11 and in y-axis wcss.
"""
from sklearn.cluster import KMeans

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init ='k-means++', max_iter = 300 ,n_init = 10 , random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11) , wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

#after implementation of Elbow method we find the opitmal number of clusters is 5.
"""
- The next step is : 
    -> apply the k-means algorithm on our data X.
- use the fit.predict method that returns for each observation which cluser it belongs to .
    
"""

kmeans = KMeans(n_clusters=5, init = 'k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans = kmeans.fit_predict(X)

#Visualising the clusters.
"""
- the final step : 
    - show the chart of the five clusters well-represented.
- we are will cluster the observations that belongs to cluster 1 .
"""

plt.scatter(X[y_kmeans == 0 ,  0 ] , X[y_kmeans == 0 , 1], s = 100 , color = 'red'     , label = 'Careful')
plt.scatter(X[y_kmeans == 1 ,  0 ] , X[y_kmeans == 1 , 1], s = 100 , color = 'blue'    , label = 'Standard')
plt.scatter(X[y_kmeans == 2 ,  0 ] , X[y_kmeans == 2 , 1], s = 100 , color = 'green'   , label = 'Target')
plt.scatter(X[y_kmeans == 3 ,  0 ] , X[y_kmeans == 3 , 1], s = 100 , color = 'cyan'    , label = 'Careless')
plt.scatter(X[y_kmeans == 4 ,  0 ] , X[y_kmeans == 4 , 1], s = 100 , color = 'magenta' , label = 'Sensible')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s=300, color = 'yellow',label = 'Centroids' )
plt.title('Clusters of clients')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score(1-00)')
plt.legend()
plt.show()


























