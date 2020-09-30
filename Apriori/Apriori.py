#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: eman-saeed
"""
#Apriori

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Market_Basket_Optimisation.csv',header=None)
"""
- we will use two for loops becouse we're going to loop over all the transactions in the data
  and the second loop will be about loop over all products in each of the transaction.

"""
transactions = []
products = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])

#Training Apriori on the dataset.
"""
- rule is the output of the Apriori function.
- Apriori function take the transactions as input and will give us the rules as output.
- Apirori function take multiple arguments: 
    1- transactions.
    2- keywords arguments : 
        1-min_support : The minimum support of relations.
        2-min_confidence : the minimum confidence of relations.
        3-min_lift : the minimum lift of relations.
        4-maximum length : min length of the relation.
"""
from apyori import apriori

rules =apriori(transactions,min_support =0.003 ,min_confidence=0.2,min_lift =3 ,min_length=2)

#Visualising the results.
results = list(rules)

    
