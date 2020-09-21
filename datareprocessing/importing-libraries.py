#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: eman-saeed
"""

"""
Importing libraries: 
    1-Numpy is a (library) mathmatical tool and it is the tool that we need to include any types og mathematics in our code.
    2-Matplotlib.pyplot (library) is a tool that help us to plot a nice charts.
    3-pandas (library) is a tool that help us to import data sets and manage data sets.
    4-sklrearn.preprocessing is used for processing any type of data.
    5-import Imputer library for solve the missing data problem by some parameters:
        1-missing_Values ='Nan' -> we put nan instead of empty value..
        2-Strategy -> we replace the empty value on a collumn based on a mean.
        3-axis -> axes equal to 0 then impute along the columns ,
                  axes equal to 1 then the imput along the rows.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from sklearn.preprocessing import Imputer 
#from sklearn.impute import SimpleImputer 
from sklearn.impute import SimpleImputer


"""
Importing the dataset.
(firstly) -> set the working environment by choose the write path of your dataset.
1- we will distinguish the dependent variables(actual output) and independent variables.
"""
dataset = pd.read_csv('Data.csv')
independted_var= dataset.iloc[:,:-1].values
dependent_var = dataset.iloc[:,3].values

"""
Taking care of missing data.

"""
imputer = SimpleImputer(missing_values =np.nan , strategy='mean',fill_value=None)
#we check the null values in independent_vars then we replace it with null.
imputer = imputer.fit(independted_var[:,1:3])
#we can check the column with the missing data .
"""
we use the transform function to : 
    -replace the missing data with the mean(average) of data 
    -
"""
independted_var[:,1:3] = imputer.transform(independted_var[:,1:3])

#Categorical data -> Encoding data
"""
- we need to categorical data :
    -to encode the text that we have into a number.
- sklearn.prepeocssing is the library that used for categorical of data.
- LabelEncoder is a class from the scaler and preprocessing library.
- we use from LabelEncoder Library fit_transform to apply it on the independent variable.
"""

from sklearn.preprocessing import LabelEncoder

labelencoder_independent_var =LabelEncoder() 


labelencoder_independent_var.fit(independted_var[:,0])
independted_var[:,0] = labelencoder_independent_var.fit(independted_var[:,0])

#We need to create a dummy variables 
"""
-how to create a dummy variables.?
- we use OneHotEncoder function to creat a dummy variable.
- onHotEncode has multiple arguments:
    - n_values
    - categorical_features: Specify what features are treated as categorical,
      actually it means that is need to specify the index of the columns of the categorical variable .
- OnHotEncoder can be replaced with ColumnTransformer due to sklearn updates, as categorical_features is depreacted feature.
"""
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

#onehotencoder = OneHotEncoder(categorical_features=[0])
#independent_var = onehotencoder.fit_transform(independent_var).toarray()
#columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])],  )
#independted_var[:,0]=np.array(columnTransformer.fit_transform(independted_var))
columnTransformer = ColumnTransformer([("Country",OneHotEncoder(),[0])],remainder ="drop")
independted_var[:,0] = labelencoder_independent_var.fit_transform(independted_var[:,0])


label_encoder_depenent_var = LabelEncoder()
dependent_var = label_encoder_depenent_var.fit_transform(dependent_var)


#splite the data into the Training set and Test set
"""
- we need to split the data 
- we use cross-validation library from sklearn to split the data into train and test.
- we have to splite the trained data into indepndent_vars_train , dependent_var_trian , independted_var_test , dependent_var_test 
- train_test_split function is take a multiple arguments : 
    1-independent variables.
    2-dependent variables .
    3-test_size.
"""
from sklearn.model_selection import train_test_split


independted_var_train , independted_var_test , dependent_var_train,dependent_var_test = train_test_split(
    independted_var
    ,dependent_var 
    , test_size =0.3 
    ,random_state=0) 

#Feature Scalling 
"""
1-one of the most popular problems on Machine learning is featurs scalling problem.
2- the used library to solve this problem is : StandardScaler.
3- rescaling each of : 
    1-trained independent_data , 
    2-tested dependent_data.
    3-trained dependent_data.
    4-trained featured_data.


"""
from  sklearn.preprocessing import StandardScaler

sc_independent_var = StandardScaler()
independted_var_train = sc_independent_var.fit_transform(independted_var_train)

independted_var_test =sc_independent_var.transform(independted_var_test)










