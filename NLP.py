#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 07:20:35 2020

@author: eman-saeed
"""
#Natural Language Processing 
"""
#introduction about NLP .
- NLP is about analyzing the text 
    , this text can be books ,
      reviews,HTML web pages that we extract from a web scrapping ,
      and all sorts of texts.
- NLP is a branch of Machine Learning that used for predictive analysis on text mostly.
"""
"""
- in our business problem these texts are going to be reviews .
- and we will maje some machine models that predict if the review is positive or negative.
"""
#importing the libraries.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset.
"""
- we import a diffrent type of dataset ->tsv file.
- pd.read_Csv will take a multiple parameters : 
    1- name of the dataset.
    2- delimiter that we want to make a separation by tap space.
    3-quoting  -> we ignore the double quots.
"""

dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting =3)

#Cleaning the data
"""
# Step2
- Cleaning the texts to make it ready for our future machine learning algorithm
  ( we will clean the diffrent reviews).
- the purpose of cleaning the data is to make a bag of word representation,
  this will consist of getting the relevant words and the diffrent reviews.
- cleaning will be by clearing the punctuations
 ,the equavalent word in meaning 
  and the unrealted word to the negative review and positive review.
- we will make a regroup to the same versions of a same word like love and loved or even loving .
"""
"""
1- we will import the library that help us to clean the texts reviews efficently,
   this library is -> re .
2- the method sub uused for the cleaning process, 
   this cleaning process will be only keeping the letters in the reviews,
   it will remove the numbers and punctuations.
3- sub method will take a multiple parameters  : 
    1- first parameters is what we don't want to remove in the text.
    2- to avoid the meaning less after removing the punctuations we put a space.
    3- the reviews.
4- we will put all the letters of the review in lowercase.
"""
"""
#Step 3 
- Remove the non-significant words by using the nltk library.
- we will download stockard list form nltk library , 
  stockard is a list of data that we will remove from the reviews dataset.
"""
"""
#Step 4 
- stemming : is finding the word with the same relevant meaning  like ->(Love,Loved,Loving).
- importing the library of steaming (from nltk.stem.porter import PorterStemmer)
- 
"""
"""
#step 5
- we will make a cotpus which is common word and basically 
  corpus is a collection of text that can be any thing,
  brieffly a corpus is a collection of text of the same type.
- corpus is an empty list.
- we will create a loop to fitt a corpus (empty list) with a a cleared reviews.


"""
import re
import nltk
nltk.download('stopwords') 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1000):
    
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

#Creating the Bag of words model.
"""
- what is a bag of word model , why we need to creat it ..?
  we need bag of word model to prevent duplicate the words,
  each column will contain a word .
- we will create a sparse matrix which is actually the bag of words models itself.
- we will make a model of bag of model through the process of tokenization.
- we will use the Class "CountVectorizer" that will used for tokenization that is the creation of our bag of words.
- CountVectorizer has multiple parameter : 
    1-stop_words    : remove all the words in the corpus becouse we will rhen apply our object with the transfer method to our corpus.
    2-lowecase      : it will put the reviews in the corpus in lowercase befor creating this huge sparse matrix.
    3-token_pattern : taking the words that have letters from A~Z removing other charachters ,
                      and the pattern is what you want to keep in the reviews as in the letters from A~Z.
    4-max_features  : to filter the non relevant words.
- sparce matrix->(X) is a matrix of features containing the diffrent independent variabels which we'll use to train our machine.
"""
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X  =cv.fit_transform(corpus).toarray() 

"""
- we will use the classification machine learning algorithms , so we will use a dependent variable (Y).
- we will use the naive bayes classification model.
"""
y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
acurracy = (55+91)/200




































