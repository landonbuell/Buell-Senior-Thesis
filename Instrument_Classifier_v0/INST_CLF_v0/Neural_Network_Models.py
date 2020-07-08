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

"""
Neural_Network_Models.py - "Neural Network Models"
    Contains Definitions to wrap and contain creation of
    Tensorflow/Keras Sequential Neural Network Models
"""

            #### VARIABLE DECLARATIONS ####
    
sepectrogram_shape = (560,256,1)
phasespace_shape = (4096,4096,1)
n_features = 15

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
    print(model.summary())
    return model

def Convolutional_Neural_Network_2D (name,in_shape,n_classes,filtersizes=[16,16],
        kernelsizes=[(3,3),(3,3)],poolsizes=[(2,2),(2,2)],layerunits=[128],
        metrics=['Precision','Recall']):
    """
    Create Tensorflow Convolutional Neural Network Model
    --------------------------------
    name (str) : Name to attatch to Network Model
    in_shape (int) : Iter of ints (n_rows x n_cols) of input images
    n_classes (int) : Number of classes in data set (output neurons)
    filtersizes (iter) : Array-like of ints indicate number of feature maps
        in i-th conv layer-group
    kernelsizes (iter) : Array-like of ints or tuples where i-th obj is
        shape of 2D kernel in each layer group
    poolsizes (iter) : Array-like of ints where i-th obj is
        shape of 2D pool size in each layer group
    layerunits (iter) : Array-like of ints where i-th int is 
        number of units in i-th hidden layer
    metrics (iter) : Array-like of strs contraining metrics to track
    --------------------------------
    Retrun compiled, untrained Keras CNN Sequential Model Object
    """
    assert len(kernelsizes) == len(poolsizes)   # same number of layer groups
    assert len(kernelsizes) == len(filtersizes) # same number of layer groups

    model = keras.models.Sequential(name=name) 
    model.add(keras.layers.InputLayer(input_shape=in_shape,name='Input'))

    # Add Convolution & Pooling
    for i,(kernel,pool) in enumerate(zip(kernelsizes,poolsizes)):
        model.add(keras.layers.Conv2D(filters=filtersizes[i],kernel_size=kernel,strides=(1,1),
            padding='same',activation='relu',name='C'+str(i+1)+'A'))            # Conv Layer A
        model.add(keras.layers.Conv2D(filters=filtersizes[i],kernel_size=kernel,strides=(1,1),
            padding='same',activation='relu',name='C'+str(i+1)+'B'))            # Conv Layer B
        model.add(keras.layers.MaxPool2D(pool_size=pool,strides=(4,4),name='P'+str(i+1)))

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
    print(model.summary())
    return model

def Convolutional_Neural_Network_1D (name,in_shape,n_classes,filtersizes=[16,16],
        kernelsizes=[(4,),(4,)],poolsizes=[(2,),(2,)],layerunits=[128],
        metrics=['Precision','Recall']):
    """
    Create Tensorflow Convolutional Neural Network Model
    --------------------------------
    name (str) : Name to attatch to Network Model
    in_shape (int) : Iter of ints (n_rows x n_cols) of input images
    n_classes (int) : Number of classes in data set (output neurons)
    filtersizes (iter) : Array-like of ints indicate number of feature maps
        in i-th conv layer-group
    kernelsizes (iter) : Array-like of ints or tuples where i-th obj is
        shape of 2D kernel in each layer group
    poolsizes (iter) : Array-like of ints where i-th obj is
        shape of 2D pool size in each layer group
    layerunits (iter) : Array-like of ints where i-th int is 
        number of units in i-th hidden layer
    metrics (iter) : Array-like of strs contraining metrics to track
    --------------------------------
    Retrun compiled, untrained Keras CNN Sequential Model Object
    """
    assert len(kernelsizes) == len(poolsizes)   # same number of layer groups
    assert len(kernelsizes) == len(filtersizes) # same number of layer groups

    model = keras.models.Sequential(name=name) 
    model.add(keras.layers.InputLayer(input_shape=in_shape,name='Input'))

    # Add Convolution & Pooling
    for i,(kernel,pool) in enumerate(zip(kernelsizes,poolsizes)):
        model.add(keras.layers.Conv1D(filters=filtersizes[i],kernel_size=kernel,strides=(2,),
            padding='same',activation='relu',name='C'+str(i+1)+'A'))            # Conv Layer A
        model.add(keras.layers.Conv1D(filters=filtersizes[i],kernel_size=kernel,strides=(2,),
            padding='same',activation='relu',name='C'+str(i+1)+'B'))            # Conv Layer B
        print(model.summary())
        model.add(keras.layers.MaxPool1D(pool_size=pool,strides=(2,),name='P'+str(i+1)))

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
    print(model.summary())
    return model