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

inputShapeMLP = (20,)
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
    def __init__(self,modelName,n_classes,path,inputAShape,inputBShape,new=True):
        """ Instantiate Class Object """
        self.name = modelName
        self.shapeA = inputAShape   # CNN shape
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
        return [self.shapeA,self.shapeB]

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
        return NeuralNetworkModels.MultiInputNetwork(self.name,
                            self.shapeA,self.shapeB,self.n_classes)
        
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
    def MultilayerPerceptron(inputShape,outputShape,neurons=[64,64]):
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
            x = keras.layers.Dense(units=nodes,activation='relu',
                                   name='D'+str(i+1))(x)
        x = keras.Model(inputs=networkInput,outputs=x,name="MultilayerPerceptron")
        return x

    @staticmethod
    def ConvolutionalNeuralNetwork2D (inputShape,outputShape,filterSizes=[32,32],
                    kernelSizes=[(3,3),(3,3)],poolSizes=[(3,3),(4,4)],neurons=[128]):
        """
        Create 2d Convolutional Neural Network branch for Neural Network
        --------------------------------
        inputShape (iter) : Iterable of ints indicating shape input array
        outputShape (int) : Number of unique output classes
        filtersSizes (iter) : Iterable of ints. i-th elem is n_filters in i-th conv layer group
        kernelSizes (iter) : Iterable of ints. i-th elem is kernel size in i-th layer group  
        neurons (iter) : Iterable of ints. i-th elem is number of nodes in i-th dense layer
        --------------------------------
        return un-complied CNN model
        """
        assert len(filterSizes) == len(kernelSizes)
        networkInput = keras.layers.Input(shape=inputShape,name="inputCNN")
        x = IdentityLayer(name='N1')(networkInput)
        for i,(filters,kernel,pool) in enumerate(zip(filterSizes,kernelSizes,poolSizes)):
            x = keras.layers.Conv2D(filters=filters,kernel_size=(kernel),activation='relu',name='C'+str(i+1)+'A')(x)
            x = keras.layers.Conv2D(filters=filters,kernel_size=(kernel),activation='relu',name='C'+str(i+1)+'B')(x)
            x = keras.layers.MaxPool2D(pool_size=pool,name='P'+str(i+1))(x)

        x = keras.layers.Flatten(name="F1")(x)          
        for i,nodes in enumerate(neurons):
            x = keras.layers.Dense(units=nodes,activation='relu',name='CD'+str(i+1))(x)
        x = keras.Model(inputs=networkInput,outputs=x,name="ConvolutionalNetwork2D")
        return x


    @staticmethod
    def MultiInputNetwork (name,inputA,inputB,n_classes):
        """
        Create Multi-Input layer neural Network
        --------------------------------
        name (str) : Name to use for neural network
        inputA (iter) : list-like of ints indicating input shape of CNN branch
        inputA (iter) : list-like of ints indicating input shape of MLP branch
        n_classses (int) : Number of unique output classes
        --------------------------------
        Return complied tf.keras model
        """
        modelCNN = NeuralNetworkModels.ConvolutionalNeuralNetwork2D(inputA,n_classes,
                        filterSizes=[32,32,32],kernelSizes=[(3,3),(3,3),(3,3)],
                        poolSizes=[(3,3),(3,3),(3,3)],neurons=[64,64])
        modelMLP = NeuralNetworkModels.MultilayerPerceptron(inputB,n_classes)

        x = keras.layers.concatenate([modelCNN.output,modelMLP.output])

        x = keras.layers.Dense(units=n_classes,activation='softmax',name='Output')(x)
        modelMain = keras.Model(name=name,inputs=[modelCNN.input,modelMLP.input],outputs=x)

        modelMain.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001,
                                        beta_1=0.9,beta_2=0.999,epsilon=1e-07),
                            loss=keras.losses.CategoricalCrossentropy(),
                            metrics=[keras.metrics.Accuracy(),keras.metrics.Precision(),keras.metrics.Recall()])
        return modelMain

