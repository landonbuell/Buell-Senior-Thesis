"""
Landon Buell
PHYS 799
Instrument Classifier v0
10 June 2020
"""

            #### IMPORTS ####

import numpy as np
import os

import tensorflow as tf
import tensorflow.keras as keras

"""
NeuralNetworkModels.py - "Neural Network Models"
    Contains Definitions to wrap and contain creation of
    Tensorflow/Keras Sequential Neural Network Models
"""

            #### VARIABLE DECLARATIONS ####

#inputShapeMLP = (24,)
#inputShapeCNN = (1115,128,1)

inputShapeMLP = (24,)
inputShapeCNN = (558,256,1)

            #### NEURAL NETWORK OBJECTS ####

class NetworkContainer:
    """
    Object that creates and contains Neural Network Model objects
    --------------------------------
    modelName (str) : user-ID-able string to indicate model
    n_classes (int) : number of discrete classes for models
    path (str) : Local parent path object where models are stored
    inputAShape (iter) : list-like of ints giving shape of CNN branch input shape
    inputBShape (iter) : list-like of ints giving shape of MLP branch input shape   
    new (bool) : If true, new Neural Networks are overwritten
    --------------------------------
    Return Instantiated Neural Network Model Object
    """
    def __init__(self,modelName,n_classes,path,inputBShape,new=True):
        """ Instantiate Class Object """
        self.name = modelName
        self.shapeB = inputBShape   # MLP shape

        self.n_classes = n_classes  # number of output classes
        self.parentPath = os.path.join(path,self.name)  # Set parent path for models      
        self.newModel = new         # create new models?
        
        if self.newModel == True:               # create new networks?
            self.MODEL = self.CreateNewModel()  # create new
            self.SaveModel()                    # save locally
        else:                                   # load exisitng networks
           self.MODEL = self.LoadExistingModel()# load model
           self.n_classes = self.MODEL.get_layer(index=-1).output_shape[-1]
           print("\t\tUPDATE: Found",self.n_classes,"classes to sort")
        assert self.n_classes is not None       # make sure we know how many classes

    def __repr__(self):
        """ Return string representation of NetworkContainer Class """
        return "Neural Network Contained Object Instance\n" + \
                "\tContains Mutli-Modal Network, Wrappers to Save and Load Parameters"

    @property
    def GetInputShapes(self):
        """ Get Input Layer Shape of Neural network """
        return [self.shapeB]

    @property
    def GetOutputShapes(self):
        """ Get Output Layer Shape of Neural network """
        return (self.n_classes,)

    def SetDecoder (self,decoder):
        """ Add Decoder Dictionary, maps Int -> Str """
        self.classDecoder = decoder
        return self
        
    def CreateNewModel (self):
        """
        Create New Neural network to specified Parameters
        --------------------------------
        * no args
        --------------------------------
        Return Instantiated Neural Network Model
        """
        return NeuralNetworkModels.MultilayerPerceptron(self.name,
                            self.shapeB,self.n_classes)
        
    def LoadExistingModel (self):
        """
        Load exisitng Neural network to specified Parameters
        --------------------------------
        * no args
        --------------------------------
        Return Instantiated Neural Network Model
        """
        try:
            model = keras.models.load_model(self.parentPath)   # load stored Model
            print("\n\tSuccessfully loaded model:",self.name,"from\n\t\t",self.parentPath)
        except:
            print("\n\tERROR! - Could not find model:",self.name,"at:\n\t\t",self.parentPath)
            raise FileNotFoundError()
        return model

    def SaveModel(self):
        """ 
        Store Current Model to local disk for future Use
        --------------------------------
        * no args
        --------------------------------
        Return Instantiated Neural Network Model
        """
        print("\tSaving Model",self.MODEL.name,"at:\n\t\t",self.parentPath)
        try:
            keras.models.save_model(self.MODEL,self.parentPath,overwrite=True)
            print("\n\tSuccessfully saved model:",self.name,"to\n\t\t",self.parentPath)
        except:
            print("\n\tERROR! - Could not save model:",self.name,"at:\n\t\t",self.parentPath)
            raise FileNotFoundError()
        return self

class IdentityLayer (keras.layers.Layer):
    """
    Identity Layer for Neural network, does nothing
    """
    def __init__(self,name):
        """ Initialize Class Object Instance """
        super().__init__(trainable=False,name=name)

    def __repr__(self):
        """ Return string representation of NullLayer Class """
        return "IndentityLayer Class returns unmodifed input 'X' "

    def Call (self,X):
        """ Call Null Layer Object """
        return X

class NeuralNetworkModels:
    """
    Container for Neural network model architectures
    --------------------------------
    * no args
    --------------------------------
    High-level wrapper for Tensorflow Sequential models
    """

    @staticmethod
    def MultilayerPerceptron(modelName,inputShape,outputShape,neurons=[64,64]):
        """
        Create MultiLayer Perceptron branch for Neural Network
        --------------------------------
        inputShape (iter) : Iterable of ints indicating shape input array
        outputShape (int) : Number of unique output classes
        neurons (iter) : Iterable of ints. i-th elem is number of nodes in i-th dense layer
        --------------------------------
        return un-complied MLP model
        """
        networkInput = keras.layers.Input(shape=inputShape,name='inputMLP')
        x = IdentityLayer(name='N1_')(networkInput)
        for i,nodes in enumerate(neurons):
            x = keras.layers.Dense(units=nodes,activation='relu', name='D'+str(i+1))(x)
        x = keras.layers.Dense(units=outputShape,activation='softmax',name='Output')(x)
        modelMain = keras.Model(inputs=networkInput,outputs=x,name=modelName)
        modelMain.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001,
                                        beta_1=0.9,beta_2=0.999,epsilon=1e-07),
                            loss=keras.losses.CategoricalCrossentropy(),
                            metrics=[keras.metrics.Accuracy(),keras.metrics.Precision(),keras.metrics.Recall()])
        return modelMain

   

