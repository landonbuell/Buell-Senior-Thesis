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
phasespace_shape = (2,4096,1)

input_shapes = [perceptron_shape,spectrogram_shape,phasespace_shape]

class Network_Container:
    """
    Object that creates and contains Neural Network Model objects
    --------------------------------
    model_names (iter) : list-like of 3 strings indicating names for models
    input_shapes (iter) : list-like of 3 tuples indicating input shape of modles
    n_classes (int) : number of discrete classes for models
    path (str) : Local parent path object where models are stored
    new (bool) : If true, new Neural Networks are overwritten
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

        self.parent_path = path         # set parent path for models
        self.n_classes = n_classes      # number of output classes
        self.new_models = new           # create new models

        if self.new_models == True:             # create new networks?
            networks = self.__createmodels__()  # create 3 models
            for network in networks:            # for each model
                self.__savemodel__(network)     # save them locally, and erase from ram
        else:                                   # load exisitng networks
            networks = self.__getmodelnames__()
            for network in networks:                # each model
                model = self.__loadmodel__(network) # load it into RAM
                self.__savemodel__(model)           # save Locally

    def __createmodels__(self):
        """ Create & name Neural Network Models """
        Model_Class = Neural_Network_Models(self.n_classes) # Instantiate class

        # Create multilayer Perceptron, and set local savepath
        MLP_Classifier = Model_Class.Multilayer_Perceptron(self.MLP_name,input_shapes[0])
        setattr(MLP_Classifier,'savepath',self.MLP_savepath)

        # Create Spectrogram 2D Conv Network and set local save path
        SXX_Classifier = Model_Class.Conv2D_Network(self.SXX_name,input_shapes[1])
        setattr(SXX_Classifier,'savepath',self.SXX_savepath)

        # Create Phase Space 1D Conv Network and set loca save path
        PSC_Classifier = Model_Class.Conv1D_Network(self.PSC_name,input_shapes[2])
        setattr(PSC_Classifier,'savepath',self.PSC_savepath)

        # NOTE: 'input_shape' (list) is a global variable in this namespace
        return [MLP_Classifier,SXX_Classifier,PSC_Classifier]

    @property
    def __getlocalpaths__(self):
        """ Get local paths where network models are stored """
        return [self.MLP_savepath,self.SXX_savepath,self.PSC_savepath]

    @property
    def __getmodelnames__(self):
        """ Return list of models names as strings """
        return [self.MLP_name,self.SXX_name,self.PSC_name]

    def __loadmodel__(self,path):
        """ Load Local model Parameter into RAM """
        if os.path.isdir(path) == False:     # is a path ?
            path = os.path.join(self.parent_path,path)  # try joining
        try:                        # try this:              
            model = tf.keras.models.load_model(path)    # load local model
        except:                     # failure ...
            print("\n\tERROR! - Could not load model from path\n\t",path)
            model = None
        return model             # return none-type

    def __savemodel__(self,model):
        """ Save model to Local Disk """
        assert type(model) == keras.models.Sequential   # must be a keras model
        try:                                            # attempt
            model.save(model.savepath,overwrite=True)   # save locally
        except:                                         # if failure ...
            model.save(os.path.join(self.parent_path,
                        model.name,overwrite=True))     # save locally
        del(model)                                  # delete from RAM
        return self                                 # return itself!


class Neural_Network_Models:
    """
    Container for Neural network model architectures
    --------------------------------
    * no args
    --------------------------------
    High-level wrapper for Tensorflow Sequential models
    """
    def __init__(self,n_classes):
        """ Initialize Class Object Instance """
        self.n_classes = n_classes

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
        model.add(keras.layers.InputLayer(input_shape=n_features,name='Input')) # input layer
        
        # Add Hidden Dense Layers
        for i,nodes in enumerate(layerunits):           # Each hidden layer
            model.add(keras.layers.Dense(units=nodes,activation='relu',name='D'+str(i+1)))
        # Add Output Layer
        model.add(keras.layers.Dense(units=self.n_classes,activation='softmax',name='Output'))

        # Compile, Summary & Return
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
                      loss=keras.losses.CategoricalCrossentropy(),
                      metrics=metrics)
        print(model.summary())
        return model

    def Conv2D_Network (self,name,inputshape,filtersizes=[(32,32),(32,32)],
                        kernelsizes=[(3,3),(3,3)],kernelstrides=[(2,2),(2,2)],
                        poolsizes=[(3,3),(3,3)],layerunits=[64],metrics=['Precision','Recall']):
        """
        Create Tensorflow 2D Convolutional Neural Network Model
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
        model.add(keras.layers.InputLayer(input_shape=inputshape,name='Input')) # input layer

        # Convolution Layer Groups
        n_layergroups = len(filtersizes)
        for i in range (n_layergroups):       # each layer group
            model.add(keras.layers.Conv2D(filtersizes[i][0],kernelsizes[i],kernelstrides[i],
                                         activation='relu',name='C'+str(i+1)+'A'))
            model.add(keras.layers.Conv2D(filtersizes[i][1],kernelsizes[i],kernelstrides[i],
                                         activation='relu',name='C'+str(i+1)+'B'))
            model.add(keras.layers.MaxPool2D(poolsizes[i],name='P'+str(i+1)))

        # Prepare Dense layers
        model.add(keras.layers.Flatten(name='F1'))
        for i,nodes in enumerate(layerunits):       # each dense layer
            model.add(keras.layers.Dense(units=nodes,activation='relu',name='D'+str(i+1)))
        model.add(keras.layers.Dense(units=self.n_classes,activation='softmax',name='Output'))
        
        # Compile, Summary & Return
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
                      loss=keras.losses.CategoricalCrossentropy(),
                      metrics=metrics)
        print(model.summary())
        return model 

    def Conv1D_Network (self,name,inputshape,filtersizes=[32,32],kernelsizes=[(3,3),(3,3)],
                        kernelstrides=[(1,1),(1,1)],poolsizes=[(2,2),(2,2)],
                        layerunits=[128],metrics=['Precision','Recall']):
        """
        Create Tensorflow 1D Convolutional Neural Network Model
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
        model = keras.models.Sequential(name=name)
        model.add(keras.layers.InputLayer(input_shape=inputshape,name='Input'))

        # Need to add 1D Convolution here!

        # Prepare Dense Layers
        model.add(keras.layers.Flatten(name='F1'))
        for i,nodes in enumerate(layerunits):       # each dense layer
            model.add(keras.layers.Dense(units=nodes,activation='relu',name='D'+str(i+1)))
        model.add(keras.layers.Dense(units=self.n_classes,activation='softmax',name='Output'))
        
        # Compile, Summary & Return
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
                      loss=keras.losses.CategoricalCrossentropy(),
                      metrics=metrics)
        print(model.summary())
        return model