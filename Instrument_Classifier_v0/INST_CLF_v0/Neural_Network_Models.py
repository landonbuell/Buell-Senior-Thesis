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

model_names = ['JARVIS','VISION','ULTRON']                  # names for models

perceptron_shape = 15
spectrogram_shape = (560,256,1)
phasespace_shape = (512,512,1)

input_shapes = [perceptron_shape,spectrogram_shape,phasespace_shape]

class Network_Models ():
    """
    Object that creates and contains Neural Network Model objects
    --------------------------------
    model_names (iter) : list-like of 3 strings indicating names for models
    input_shapes (iter) : list-like of 3 tuples indicating input shape of modles
    n_classes (int) : number of discrete classes for models
    path (str) : Local parent path object where models are stored

    --------------------------------
    Return Instantiated Neural Network Model Object
    """
    def __init__(self,model_names,n_classes,path,new=True):
        """ Instantiate Class Object """
        assert len(model_names) == 3
        assert len(input_shapes) == 3

        self.MLP_name = model_names[0]      # Perceptron
        self.SXX_name = model_names[1]      # Spectrogram
        self.PSC_name = model_names[2]      # Phase-Space

        self.MLP_savepath = os.path.join(path,self.MLP_name)    # where to save model
        self.SXX_savepath = os.path.join(path,self.SXX_name)    # where to save model
        self.PSC_savepath = os.path.join(path,self.PSC_name)    # where to save model

        self.n_classes = n_classes      # number of output classes
        self.new_models = new           # create new models

        if self.new_models == True:     # create new networks?
            self.__createmodels__()     # create them
            self.__savemodels__()       # save them locally, and erase from ram

    def __createmodels__(self):
        """ Create & name Neural Network Models """
        self.MLP_Classifier = self.Multilayer_Perceptron(name=self.MLP_name,
                                n_features=input_shapes[0])
        self.SXX_Classifier = self.Conv2D_Network(name=self.SXX_name,
                                    inputshape=input_shapes[1])
        self.PSC_Classifier = self.Conv2D_Network(name=self.PSC_name,
                                    inputshape=input_shapes[2])
        # NOTE: 'input_shape' (list) is a global variable in this namespace
        return self

    def __loadmodel__(self,model):
        """ Load Local model Parameter into RAM """
        return model

    def __savemodel__(self):
        """ Save model to Local Disk """
        self.MLP_Classifier.save(self.path)
        retuen self

    def update_pathmap (self,map):
        """ Update paths-dictionary to include paths to models """
        map.update({str(self.MLP_name),self,})
        return map

    def Multilayer_Perceptron (self,name,n_features,layerunits=[40,40],
                               metrics=['Precision','Recall']):
        """
        Create Mutlilayer Perceptron and set object as attribute
        --------------------------------
        name (str) : Name to attatch to Network Model
        n_features (int) : Number of input features into Network
        layerunits (iter) : List-like of ints. I-th element is nodes in I-th hiddenlayer
        metrics (iter) : Array-like of strs contraining metrics to track
        --------------------------------
        Return Compiled, unfit model instance
        """
        model = keras.models.Sequential(name=name)      # create instance & attactch name
        model.add(keras.layers.InputLayer(input_shape=n_features),name='Input') # input layer
        
        # Add Hidden Dense Layers
        for i,nodes in enumerate(layerunits):           # Each hidden layer
            model.add(keras.layers.Dense(units=nodes,activation='relu',name='D'+str(i+1)))

        # Add Output Layer
        model.add(keras.layers.Dense(units=self.n_classes,activation='softmax'),name='Output')

        # Compile, Summary & Return
        model.compile(optimizier=keras.optimizers.Adam(learning_rate=0.01),
                      loss=keras.losses.CategoricalCrossentropy(),
                      metrics=metrics)
        print(model.summary())
        return model

    def Conv2D_Network (self,name,inputshape,filtersizes=[(32,32),(32,32)],
                        kernelsizes=[(3,3),(3,3)],kernelstrides=[(1,1),(1,1)],
                        poolsizes=[(2,2),(2,2)],layerunits=[128],metrics=['Precision','Recall']):
        """
        Create Tensorflow Convolutional Neural Network Model
        --------------------------------
        name (str) : Name to attatch to Network Model
        inputshape (iter) : List-like of ints, indicating dimensionality of input figures
        filtersizes (iter) : List-like of ints.
            i-th element is number of filters in i-th layer group
        kernelsizes (iter) : List-like of tups of ints.
            i-th element is shape of kernel in i-th layer group
         kernelsizes (iter) : List-like of tups of ints.
            i-th element is shape of kernel in i-th layer group
        layerunits (iter) : List-like of ints. I-th element is nodes in I-th hiddenlayer
        metrics (iter) : Array-like of strs contraining metrics to track
        --------------------------------
        """
        model = keras.models.Sequential(name=name)      # create instance & attactch name
        model.add(keras.layers.InputLayer(input_shape=inputshape),name='Input') # input layer

        # Convolution Layer Groups
        n_layergroups = len(filtersizes)
        for i in range (len(N_layer_groups)):       # each layer group
            model.add(keras.layers.Conv2D(filtersizes[i][0],kernelsizes[i],kernelstrides[i],
                                         activation='relu',name='C'+str(i+1)+'A'))
            model.add(keras.layers.Conv2D(filtersizes[i][1],kernelsizes[i],kernelstrides[i],
                                         activation='relu',name='C'+str(i+1)+'B'))
            model.add(keras.layers.MaxPool2D(poolsizes[i],name='P'+str(I+1)))

        # Prepare Dense layers
        model.add(keras.layers.Flatten(name='F1'))
        for i,nodes in enumerate(layerunits):       # each dense layer
            model.add(keras.layers.Dense(units=nodes,activation='relu',name='D'+str(i+1)))

        model.add(keras.layers.Dense(units=self.n_classes,activation='softmax',name='Output'))
        
        # Compile, Summary & Return
        model.compile(optimizier=keras.optimizers.Adam(learning_rate=0.01),
                      loss=keras.losses.CategoricalCrossentropy(),
                      metrics=metrics)
        print(model.summary())
        return model 

    def Conv3D_Network (self,name,inputshape,filtersizes=[32,32],kernelsizes=[(3,3),(3,3)],
                        kernelstrides=[(1,1),(1,1)],poolsizes=[(2,2),(2,2)],
                        layerunits=[128],metrics=['Precision','Recall']):
        """
        Create Tensorflow Convolutional Neural Network Model
        --------------------------------
        name (str) : Name to attatch to Network Model
        inputshape (iter) : List-like of ints, indicating dimensionality of input figures
        filtersizes (iter) : List-like of ints.
            i-th element is number of filters in i-th layer group
        kernelsizes (iter) : List-like of tups of ints.
            i-th element is shape of kernel in i-th layer group
         kernelsizes (iter) : List-like of tups of ints.
            i-th element is shape of kernel in i-th layer group
        layerunits (iter) : List-like of ints. I-th element is nodes in I-th hiddenlayer
        metrics (iter) : Array-like of strs contraining metrics to track
        --------------------------------
        """
        pass
