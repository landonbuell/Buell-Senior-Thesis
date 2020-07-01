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

            #### VARIABLE DECLARATIONS ####
    
sepectrogram_shape = (560,256,1)

            #### FUNCTION DEFINITIONS ####

def Multilayer_Perceptron (name,n_features,n_classes,layerunits=[40,40],
                           metrics=['Precision','Recall']):
    """
    Create Tensorflow Multilayer Perceptron Model
    --------------------------------
    name (str) : Name to attatch to Network Model
    n_features (int) : Number of features in data set (input neurons)
    n_classes (int) : Number of classes in data set (output neurons)
    layerunits (iter) : Array-like of ints where i-th int is 
        number of units in i-th hidden layer
    metrics (iter) : Array-like of strs contraining metrics to track
    --------------------------------
    Retrun compiled, untrained Keras MLP Sequential Model Object
    """
    model = keras.models.Sequential(name=name) 
    model.add(keras.layers.InputLayer(input_shape=n_features,name='Input_Layer'))
    # Add hidden layers
    for i,nodes in enumerate(layerunits):
        model.add(keras.layers.Dense(units=nodes,activation='relu',name='D'+str(i+1)))
    # ouput layer & compile
    model.add(keras.layers.Dense(units=n_classes,activation='softmax',
                                name='Output_Layer'))
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=metrics)
    # Summary & return
    #print(model.summary())
    return model

def Convolutional_Neural_Network_2D (name,in_shape,n_classes,kernelsizes=[2,2],
                                layerunits=[40,40],metrics=['Precision','Recall']):
    """
    Create Tensorflow Convolutional Neural Network Model
    --------------------------------
    name (str) : Name to attatch to Network Model
    in_shape (int) : Iter of ints (n_rows x n_cols) of input images
    n_classes (int) : Number of classes in data set (output neurons)
    layerunits (iter) : Array-like of ints where i-th int is 
        number of units in i-th hidden layer
    metrics (iter) : Array-like of strs contraining metrics to track
    --------------------------------
    Retrun compiled, untrained Keras CNN Sequential Model Object
    """
    model = keras.models.Sequential(name=name) 
    model.add(keras.layers.InputLayer(input_shape=in_shape,name='Input'))

    # Add Convolution & Pooling
    for i,size in enumerate(kernelsizes):              # each group of layers
        model.add(keras.layers.Conv2D(filters=16,kernel_size=size,strides=(1,1),
            padding='same',activation='relu',name='C'+str(i+1)+'A'))            # Conv Layer A
        model.add(keras.layers.Conv2D(filters=16,kernel_size=size,strides=(1,1),
            padding='same',activation='relu',name='C'+str(i+1)+'B'))            # Conv Layer B
        model.add(keras.layers.MaxPool2D(pool_size=(8,8),strides=(4,4),name='P'+str(i+1)))

    # Add Dense Layers
    model.add(keras.layers.Flatten())       # flatten for dense layers
    for i,nodes in enumerate(layerunits):   # each Dense Layer
        model.add(keras.layers.Dense(units=nodes,activation='relu',name='D'+str(i+1)))

    # ouput layer & compile
    model.add(keras.layers.Dense(units=n_classes,activation='softmax',
                                name='Output'))
    model.compile(optimizer=keras.optimizers.Adam(),
                    loss=keras.losses.CategoricalCrossentropy(),
                    metrics=metrics)
    # Summary & return
    #print(model.summary())
    return model
