#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 20:04:24 2020

@author: eman-saeed
"""

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#importing the dataset 

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
"""
#Implementation of random selection
import random
N = 10000 
d = 10
ads_selected =[]
total_reward = 0
for n in range(0,N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n,ad]
    total_reward = total_reward + reward

#Visualizing the results - Histogram
    
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
"""
#Implementation of UCB
#- this algorithm take 3 steps: 
"""
- Step 1:
    1- each round we consider two numbers for each ad i,
       that is for each version of the add these(two numbers are (i) which is the number of times that we select up to round (n)),
       and some of rewards of the adversion I have two round .
       then now we need to declare these two variables 
           1-i-> the number of times.
           2- sum of rewards .
       the (number_of_selections) will be created as vector of size D that containing only zeros becouse the first round the sums of reward  of each version of the ad is zeros.     
"""
N = 10000
d = 10
numbers_of_selections = [0] * d 
sums_of_reward = [d] * d

"""
- Step 2:
    - From these two numbers we compute first the average reward of at i at up to round .
    - we will make a for loop for making calculte the above numbers in each iteration.
    - N is the totel number of round.
    - we need to compute for each version of the ad the average word and the confidence interval.

"""
"""
- Step 3: 
    - Select the ad with the adversion i that has the maximum upper bound.
    - we make a vector (a huge list) that contain diffrent versions of the ADD that were selected at each round.
    
"""

import math

ads_selected = []
total_reward = 0
for n in range(0,N):
    ad = 0
    max_upper_bound=0
    for i in range(0,d):
        if(numbers_of_selections[i] > 0 ):
            average_reward = sums_of_reward[i]/numbers_of_selections[i]
            delta_i = math.sqrt(3/2 *math.log(n+1)/numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound>max_upper_bound:
            max_upper_bound=upper_bound
            ad = i
        ads_selected.append(ad)
        numbers_of_selections[ad] = numbers_of_selections[ad]+1
        reward = dataset.values[n, ad]
        sums_of_reward[ad]=sums_of_reward[ad]+reward
        total_reward = total_reward+reward
            
  #Visualizing the results
        
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()













