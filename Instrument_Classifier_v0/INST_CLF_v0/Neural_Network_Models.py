"""
Landon Buell
PHYS 799
Instrument Classifier v0
10 June 2020
"""

            #### IMPORTS ####

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow.keras as keras

            #### FUNCTION DEFINITIONS ####

def Multilayer_Perceptron (name,n_features,n_classes,layer_units=[40,40],
                           metrics=['Precision','Recall']):
    """
    Create Tensorflow Multilayer Perceptron Model
    --------------------------------
    name (str) : Name to attatch to Network Model
    n_features (int) : Number of features in data set (input neurons)
    n_classes (int) : Number of classes in data set (output neurons)
    layer_units (iter) : Array-like of ints where i-th int is units 
        in i-th hidden layer
    metrics (iter) : Array-like of strs contraining metrics to track
    --------------------------------
    Retrun compiled, untrained Keras MLP Sequential Model Object
    """
    model = keras.models.Sequential(name=name) 
    model.add(keras.layers.InputLayer(input_shape=n_features,name='Input_Layer'))
    # Add hidden layers
    for L,N in enumerate(layer_units):
        model.add(keras.layers.Dense(units=N,activation='relu',
                                     name='Hidden_'+str(L+1)))
    # ouput layer & compile
    model.add(keras.layers.Dense(units=n_classes,activation='softmax',
                                name='Output_Layer'))
    model.compile(optimizer=keras.optimizers.SGD(),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=metrics)
    # Summary & return
    print(model.summary())
    return model

def Convolutional_Neural_Network (name,in_shape,n_classes,layer_units=[40,40],
                            metrics=['Precision','Recall']):
    """
    Create Tensorflow Convolutional Neural Network Model
    --------------------------------
    name (str) : Name to attatch to Network Model
    in_shape (int) : Iter of ints (n_rows x n_cols) of input images
    n_classes (int) : Number of classes in data set (output neurons)
    layer_units (iter) : Array-like of ints where i-th int is units 
        in i-th hidden layer
    metrics (iter) : Array-like of strs contraining metrics to track
    --------------------------------
    Retrun compiled, untrained Keras CNN Sequential Model Object
    """
    model = keras.models.Sequential(name=name) 
    model.add(keras.layers.InputLayer(input_shape=in_shape,name='Input'))
    # Add convolution & pooling
    model.add(keras.layers.Conv2D(filters=4,kernel_size=(3,3),strides=(1,1),
                                  activation='relu',name='C1'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2),name='P1'))
    model.add(keras.layers.Conv2D(filters=2,kernel_size=(3,3),strides=(1,1),
                                  activation='relu',name='C2'))
    model.add(keras.layers.MaxPooling2D(pool_size=(8,8),name='P2'))
    # Add Dense Layers
    model.add(keras.layers.Flatten())       # flatten for dense layers
    for L,N in enumerate(layer_units):
        model.add(keras.layers.Dense(units=N,activation='relu',
                                     name='D'+str(L+1)))
    # ouput layer & compile
    model.add(keras.layers.Dense(units=n_classes,activation='softmax',
                                name='Output'))
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=metrics)
    # Summary & return
    print(model.summary())
    return model
