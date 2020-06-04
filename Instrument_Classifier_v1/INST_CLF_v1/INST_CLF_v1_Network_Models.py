"""
Landon Buell
Instrument Classifier v1
Classifier - Neural Network Models
4 June 2020
"""

import numpy as np
import pandas as pd
import os

import tensorflow as tf
import tensorflow.keras as keras

            #### PREPROCESSING FUNCTIONS ####

def Design_Matrix_Scaler (X):
    """
    Scale design matrix to have 0 mean, & unit variance
    --------------------------------
    X (arr) : (n_samples x n_features) standard Design matrix to scale
    --------------------------------
    Return scaled design matrix
    """
    scaler = StandardScaler()   # obj inst
    X = scaler.fit_transform(X) # fit the inst & tranf X
    return X

def one_hot_encoder (y,n_classes):
    """
    One-Hot-Encode array y
    --------------------------------
    y (arr) : (n_samples x 1) sample target array
    n_classes (int) : number of target classes
    --------------------------------
    Return matrix Y, (n_samples x n_classes)
    """
    y = np.array(y).ravel()     # make into arr & flatten
    Y = keras.utils.to_categorical(y,n_classes,dtype='int')

            #### NEURAL NETWORK MODELS ####

def CNN_2D_Classifier(name,n_classes,path=None,metrics=None): 
    """
    Create Keras Sequential Object as Convolution Neural Network Model
        Model inspired from LeNet-5 Architecture
    --------------------------------
    name (str) : Name to attack the CNN instance
    n_classes (int) : Number of target class in data
    path (str) : if specified, Local directory to store model data, (None by default)
    metric (iter) : metrics to include in compile method, (None by default)
    --------------------------------
    Return Untrained CNN instance
    """
    # Create Model & add Input Layer
    model = keras.models.Sequential(name=name)
    model.add(keras.layers.InputLayer(input_shape=(None,None,1)))
    # Convolution layers
    model.add(keras.layers.Conv2D(filters=2,kernel_size=(3,3)))
    model.add(keras.layers.AveragePooling2D(pool_size=(2,2),strides=(2,2)))
    # Dense Layers
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=40,activation='relu'))
    # Output
    model.add(keras.layers.Dense(unitss=n_classes,activation='softmax'))
    model.compile(optimizer=keras.optimizers.SGD(),
                  loss='categorical_crossentropy',
                  metrics=metrics)

    # Save Locally?
    if path != None:
        filepath = path + '/' + model.name
        model.save(filepath=filepath,saveformat='tf')
        setattr(model,'filepath',filepath)

    # return model instance
    return model

def MLP_Classifier (name,n_features,n_classes,path=None,metrics=None):
    """
    Create Keras Sequential Object as Convolution Neural Network Model
        Model inspired from LeNet-5 Architecture
    --------------------------------
    name (str) : Name to attack the CNN instance
    n_features (int) : Number of features in data design matrix
    n_classes (int) : Number of target class in data
    path (str) : if specified, Local directory to store model data, (None by default)
    metric (iter) : metrics to include in compile method, (None by default)
    --------------------------------
    Return Untrained Network instance
    """
    # Create Model & add Input Layer
    model = keras.models.Sequential(name=name)
    model.add(keras.layers.InputLayer(input_shape=(n_features,1)))
    # Dense Layers
    model.add(keras.layers.Dense(units=20,activation='relu'))
    model.add(keras.layers.Dense(units=20,activation='relu'))
    # Output
    model.add(keras.layers.Dense(unitss=n_classes,activation='softmax'))
    model.compile(optimizer=keras.optimizers.SGD(),
                  loss='categorical_crossentropy',
                  metrics=metrics)

    # Save Locally?
    if path != None:
        filepath = path + '/' + model.name
        model.save(filepath=filepath,saveformat='tf')
        setattr(model,'filepath',filepath)

    # return model instance
    return model
