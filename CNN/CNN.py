#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: eman-saeed
"""
#CNN
#Part -1 -Building the CNN
"""
- A convolutional neural network is just a artificial neural network , 
  in which you use this convolution trick to add some convolutional layers.
- we make the convolution trick to preserve the special structure in images and classify some images.
- we are going to classify the image either cat or dog.
"""
#Prepare the work environment.
"""
- in this problem we have an images ,so we need to preprocessing the image,
  to be able to input these images in our Convolutional neural network.
- this data set has a diffrent structure : we split the folders of training set and test set,
  as the independent variables now are pixels distributed in 3D arrays .

-we are use keras to import some images in a very efficient way :
    1- prepare a very special structrue for our data set,
       in this structure we make a in training folder another two folders :
           1- cat folder .
           2- dog folder . 
         and the same structure for the test folder.
    * keras will understand how to differentiate the labels of your depdent variable.
"""
#Importing the Keras libraries and packages 
"""
- First package is sequential used to intialize our neural network , 
  as there are two ways of initializing a neural network : 
      1- sequence of layers .
      2- graph.

- Second package is Convolution2D is the package that we'll use for the first step of making the CNN,
  which is we add the convolutional layeres .
  the Convolution2D package deal with the images.
  
- the third package  : MaxPooling2D that used in add our pooling layers .
- the fourth package : Flatten that used for convert the pooled feature maps ,
                       that we created through convolution and max pooling into this large feature vector ,
                       and it will be input of our fully connectedd layers.
-the last package : Dense is a package we use to add the fully connected layers and a classic ANN .
 """
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D 
from keras.layers import Flatten
from keras.layers import Dense
import tensorflow as tf

#Intialising the CNN
Classifier = Sequential()
#Step 1 -Convolution =
"""
- in this step we are applying a several feature detectors on the input image(cat/dog)
- after applying the input image with the feature detector (filter) we find the Feature Map.
- a feature map contains sum numbers and the highest numbers of the feature map is where in the feature detector could detect a specific feature in the input image.
- the result of the feature map is the convolution operation .
"""
"""
- we use add() method to create a convolutional layer .
- add method has multiple parameters: 
    1-Convolution2D () has miltple arguments : 
        1- nb_filter which is number of filter ->32,
           then we composed of 32 feature maps.
        2- the dimensions of the feature detector (filter) 3 x 3 .
        3- input shape : the shape of your input image ,
           not all the images in the same format so we have to force 
           all images in the same format,so we will make all image in the same format 
           and the same fixed size, it will be in another part of code,
           but in this argument we enter the expected format of our format input images,
           is that B/W image or Colored image.
       4- the activation function :  we use the Rectifier activation function to increase non-linearity.
"""
Classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))
#Step 2 - Pooling 
"""
- we applying the pooling layer becouse we want to reduce the number of nodes we'll get in the next step.
- we use the MaxPooling2D class with some parameters: 
    1- pool_size the most default is 2 X 2 .
"""

Classifier.add(MaxPooling2D(pool_size=(2,2)))
#adding a second convolutional neural Layer.


Classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))
Classifier.add(MaxPooling2D(pool_size=(2,2)))
   
#step 3 -Flattening 
"""
1- in this step we are going to take the pooled feature map and convert it to a huge single vector.
"""
Classifier.add(Flatten())
#Step 4 -Full Connection
"""
- Dense has multiple arguments : 
	1- output_dim -> 100 is a good choice but a good practice 2^() so we use 128 as output nodes.
	2-activation function -> in  hidden layer we use (Rectifier function ) .
	3- we use the sigmoid in the output layer to give the probabilities to each class.
"""
 #Classifier.add(Dense(output_dim = 128,activation='relu'))
Classifier.add(tf.keras.layers.Dense(units=128, activation='relu'))
#Classifier.add(Dense(output_dim = 1  ,activation='sigmoid'))
Classifier.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#Compile the CNN
"""
- Comile method has multiple parameters: 
    - Optimizer is like a stochastic gradient descent to find the optimal number of the weights.
    - loss function.
"""

Classifier.compile(optimizer='adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

#Part 2 - Fitting the CNN to the images.
"""
- keras process to images is called Image augmentation that consists of pre-preocessing your images,
  to prevent overfitting ..
- if we don't make the image augmentation we will have a high accuracy in a training set ,
  but in the test set we will have much lower accuracy undet test set,
  and that is exactly is overfitting-> great results in trainig situation and poor results in testing situation.
  so befor fitting the CNN to our images let's proceed to this image augmentation process.
  
- what is Image augmentation and how will it prevent overfitting .??
  the first situation that lead to overfitting is 
      : a model find some correlations in the few observation of the training set ,
        but fails to generalize these correlations on some new observations.
        so write now we are working with 10000 images -> 8000 image training,
                                                      -> 2000 image testing.
        but this is actually not much to get some great performance results,
        we either need sime images or we can use a trick ..? 
        so that is data augmentation comes into play, that is the trick 
        becouse it create many batches of our images ,
        and in each batch it will apply some random transformations on a random selection of our image just like
        rotating them , shifting them , or even shearing them ,
        so we will find a lot more material to train.
#BRIEF : 
        why Image augmentation : 
            - That's becouse the amount of our training images is augmented 
              besides becouse the transformations are random transformations,
              so our model will never find the same picture across the batches,
              so all this image augmentation trick can only reduce overfitting. 
        Image augmentation is a technique that allows us to enrich our data set,our training set
        without adding more images and therefore that allows us to get good performance results with
        little or no overfitting even with small amount of images .
"""

"""
there are 2 ways to applying image augmentation on them:
    1- by using code based on flow method.
    2- by using the code based on flow_from_directory method.
"""
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
                    rescale    = 1./255,
                    zoom_range = 0.2,
                     shear_range = 0.2,
                    horizontal_flip=True)

test_datagen    = ImageDataGenerator(rescale=1./255)

"""
 - train_set variable use the  train_datagen.flow_from_directory with multiple arguments:
     1- the directory that we will extract the images from.
     2- target_size -> is the size of your image that is expected in your CNN model.
     3- batch_size  -> that is the size of the batches in which some random samples of our images will be included .
     4- class_mode  -> that is the parameter indicating if your class ,your dependent variabel in binary or has more than 
                       two catehories and therefor since we have two classes here(CAT/DOG) then this class is binary.
"""
train_set = train_datagen.flow_from_directory(
                        'archive/training_set',
                        target_size= (64,64),
                        batch_size = 32,
                        class_mode = 'binary')
"""
 -   test_set variable use the  train_datagen.flow_from_directory with multiple arguments:
         1- the directory that we will extract the images from.
         2- target_size will be the same of the training set.
         3- batch_size  will be the same of the training set.
         4- class_mode  will be the same of the training set.
"""
test_set = test_datagen.flow_from_directory(
                        'archive/test_set',
                        target_size = (64,64),
                        batch_size  = 32,
                        class_mode  = 'binary' )
"""
- the fitting of the model :
- bt using the fit_generator method with multiple parameters: 
    1- training set .
    2- sample_per_epoch (steps_per_epoch =)-> which is simply the number of images we have in our 
                            training sets becouse all the observation of the 
                            training set pass through the Convolutional neural network during each epoch,
                            and since we have 8000 images in our trainig set .
    3- nb_epoch (epoches) -> the number of epochs you want to choose to train our CNN.
    4- validation_data -> that corresponds to the test set which we want to evaluate the performance of our CNN.
    5- nb_val_samples  -> that corresponds to the number of images in our test set.
"""
Classifier.fit_generator(train_set,
                    steps_per_epoch =8000,
                    epochs=25,
                    validation_data=test_set,
                    validation_steps=2000)
 
