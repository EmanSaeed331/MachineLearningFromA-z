#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 21:56:44 2020

@author: eman-saeed
"""
# Artificial Neural Network.
# Import the libraries .
"""
1- Theano is a  numerical computation library,
   This library can be run in CPU and GPU. 
2- Tesnsorflow is another open source numerical computations library,
   That runs very fast and again can run on your CPU or on GPU . 
3- keras is amazing  library to build deeper models like deep neural networkd in a very few lines of code.
   keras is based on Theano and Tensorflow.
"""
#part 1 -> Data Preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
tf.__version__ 
#Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X =dataset.iloc[:,3:13].values
y = dataset.iloc[:, 13].values
#Encoding the categorical data.
#Enconding the independent variable..
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
label_encoder_X_1 = LabelEncoder()
label_encoder_X_2 = LabelEncoder()
X[:, 1] = label_encoder_X_1.fit_transform(X[:, 1])
X[:, 2] =label_encoder_X_2.fit_transform(X[:, 2])
columntransformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
columntransformer2 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2])], remainder='passthrough')
X = np.array(columntransformer.fit_transform(X))
#Avoiding the dummy variable trap 
X = X[:, 1:]
#Importing the dataset into Training set and Test set.
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=0)

#Features Scalling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.fit_transform(X_test)

#Part 2 - Now let's make the ANN.
"""
1- improting keras libraries and package.
2- keras is using tensorflow backend and that mean keras will build the deep neural networks based on tensorflow.
3- we are going to import 2 modules :
    1- sequential module that is required to initialize our neaural network .
    2- Dense module that is required to build a layers of out NN.
"""
import keras
from keras.models import Sequential
from keras.layers import Dense

#Intialising the ANN by using the Sequential class.
classifier = Sequential()

"""
- we are going to add first layer ANN which is the input layer in the first hidden layer.
- to build a Neural Network there are 7 steps with stochastics gradient descent: 
    1- Randomly initialise the weights to small numbers close to 0
    2- Input the first observation of your dataset in the input layer , each feature in one input node.
    3- Forward-Propagation .
    4- COmpare the predicted value with the actual value.
    5- Back propgation .
    6- Repeat Steps 1 ~ 5 and update weights by (Batch Learning).
    7- Redo epochs.
- use dense class to apply the first step.
- we will use the rectifier actication funtion in the hidden layers.
- choose the number of the node in the layer.
- we will use the sigmoid   activation function in the output layers. 
 """
#Adding the input layer and the first hidden layer.
"""
- Dence function has multiple arguments: 
    1- (output_Dim) or (unit)  that is simply the number of nodes you want to add in this hidden layer,
                       How many nodes can we add in this hidden layer?
                       there are some rules :
                        1- if your data is linearly separable we don't need a hiddent layer 
                           , and you don't even need NN
                           , we can characterize it as a rule rather than a tip ( this step based on experments)
                        2- we can choose the number of nodes in hidden layer as the average of the  number of nodes in the input layer and the number of nodes in the output layer that is the type you want to use.
                        3- this type we want to use with technique called parameter tunning.
                        4- parameter tunning is about using techniques like (K4-Cross-validation) ,
                           this technique is consists of creating a separate set in your dataset.
    2- (init) is the next argument , it is randomly intialized the weights as small numbers close to zero, 
       so we can see that we have the glow red uniform function to initialize the weights , 
       intializing the weights will ve according the uniform distribution.
    3- activate -> which is the activation function that we are going to choose in our hidden layer ( rectifier activation function ) called relu.
    4- input_dim -> the number of nodes in the input layer that is number of independent variable.
""" 
"""
- we calculated the number of nodes in the input layer is 11 (the number of independent variable).
- the number of the output layer is (1) , as we find the dependent variable has a binary outcomes (one or zero).
- the average equale (11+1)/2 = 6.
- 6 nodes will be in the hidden layer .

"""
#classifier.add(Dense( output_Dim=6,init = 'uniform',activation='relu',input_dim=11))
classifier.add(tf.keras.layers.Dense(units=6, activation='relu'))
#Adding the second hidden layer 
classifier.add(tf.keras.layers.Dense(units=6, activation='relu'))
#Adding the output layer 
"""
- we use in output layers the sigmoid activation function , sigmoid is the probabilistic approach .
- the output layer will contain just one node as it is a categorical probelm,
  which is depend on variable is categotical horrible with binary outcome.
  0 if the custome stays in a bank and 1 if the customer leaves the bank.
"""
#HINT 
"""
- if the output layer has multiple categories we use activation function (SOFT-MAX).
- (SOFT-MAX) is works for multiple categories output layer.
"""
classifier.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#Compiling the ANN 
"""
- we are going to compile all the ANN 
- we use the compile method .
- compile methods has several parameters :
    1- Optimizer : the algorithm that we are use to find the optimal number of state of weights,
                   in the Neural Network, 
                   so we need to apply some sort of algorithm to find the best weights, 
                   that will make our NN more powerfuel.
                   the very effiecnt algorithm is called (ADAM).
                   so it is actually will be the optimizier of the parameter.
    2- loss : which is the loss function within the stochastic algorithm  but with the categorical
              dependent variable the used loss function is (binary_crossentropy).
    3- metrix : it just used to evaluate your model             
"""
classifier.compile(optimizer = 'adam',loss='binary_crossentropy',metrics =['accuracy'])

#Fitting the ANN to TRaining set.
"""
- we fit the model with fit method which take multiple arguments: 
    1-Batch size : is the number of the observation after which you want to update the weights.
    2- epoch     : is basically around when the whole training set passed through the ANN .
    there is no rule of choosing the Batch size and epoch .
    but from the exprements the best choise of batch_size = 10 , 
    and number of epochs is 1000 . 
- the accuracy will be imporoved over the rounds that is over the different epochs.

"""
classifier.fit(X_train,y_train,batch_size = 10,epochs=10)

#Part 3 -Making the predictions and evaluating the model 
#Prediction the Test set results.
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
#Make a confusion matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
