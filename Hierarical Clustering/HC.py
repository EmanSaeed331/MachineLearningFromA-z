#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: eman-saeed
"""
#Hierarchical clustering

#1- importig the libraries  
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values
#find the optimal number of clusters by dendrogram.
"""
1- import  library -> scipy.cluster.hierarchy .
2- dendogram method has muliple arguments : 
    1- sch.linkage : is an algorithm itsself of hierarchal clustering . 
    2- sch.linkage has multiple parameter :
        1- X as an array .
        2- method -> is the method that tries to minimize the variance within each cluster.
"""
import scipy.cluster.hierarchy as sch
dendogram  = sch.dendrogram(sch.linkage(X,method ='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distance')
plt.show()

#Fitting the hierarichal clustering to the mall dataset.
"""
1- import sklearn.cluster import AgglomerativeClustering to use the aggomerativeclustering.
2- AgglomerativeClustring has multiple parameter like :
    1- n_clusters which now is 5.
    2- affinity which is actually like the linkage , we use it as a distance parameter.
    3- linkage .
"""
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc = hc.fit_predict(X)

#Visualizing the Clusters.

plt.scatter(X[y_hc == 0 ,  0 ] , X[y_hc == 0 , 1], s = 100 , color = 'red'     , label = 'Careful')
plt.scatter(X[y_hc == 1 ,  0 ] , X[y_hc == 1 , 1], s = 100 , color = 'blue'    , label = 'Standard')
plt.scatter(X[y_hc == 2 ,  0 ] , X[y_hc == 2 , 1], s = 100 , color = 'green'   , label = 'Target')
plt.scatter(X[y_hc == 3 ,  0 ] , X[y_hc == 3 , 1], s = 100 , color = 'cyan'    , label = 'Careless')
plt.scatter(X[y_hc == 4 ,  0 ] , X[y_hc == 4 , 1], s = 100 , color = 'magenta' , label = 'Sensible')
plt.title('Clusters of clients')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score(1-00)')
plt.legend()
plt.show()







