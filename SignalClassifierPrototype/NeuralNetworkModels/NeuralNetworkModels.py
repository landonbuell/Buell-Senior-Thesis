"""
Repository:     Buell-Senior-Thesis
Solution:       SignalClassifierPrototype
Project:        NeuralNetworkModels
File:           __main__.py
 
Author:         Landon Buell
Date:           January 2022
"""

        #### IMPORTS ####

import os
import sys

import numpy as np
import tensorflow as tf


        #### CLASS DEFINTIONS ####

class ModelConfiguration:
    """ Class To Hold All Model Configuration Data """

    def __init__(self):
        """ Constructor for ModelConfiguration Instance """
        self._configSubModelA = None
        self._configSubModelB = None
        self._configSubModelC = None

    def __del__(self):
        """ Destructor for ModelConfiguration Instance """
        self._configSubModelA = None
        self._configSubModelB = None
        self._configSubModelC = None

class IdentityLayer(tf.keras.layers.Layer):
    """ Identity Layer Returns Inputs Exactly """

    def __init__(self,name):
        """ Constructor for IdentityLayer Instance """
        super().__init__(trainable=False,name=name,dtype=tf.float32)

    def __del__(self):
        """ Destructor for IdentityLayer Instance """
        super().__del__()

    def call(self,inputs):
        """ Invoke Layer w/ Inputs """
        return inputs

class TensorflowMultilayerPerceptron:
    """ Construct Tensorflow Multilayer Perceptron Instance """

    def __init__(self,
                 numFeatures,
                 numClasses=None,
                 neurons=[96,128,128,96,96,64],
                 activationFunction="relu",
                 name="defaultMLP"):
        """ Construct TensorflowMultilayerPerceptron Instance """
        self._numFeatures = numFeatures
        self._numClasses = numClasses
        self._layerWidths = neurons
        self._activationFunction = activationFunction
        self._name = name
        self._model = None

    def __del__(self):
        """ Destruct TensorflowMultilayerPerceptron Instance """
        self._model = None

    def getModel(self):
        """ Return the Tensorflow Model Instance """
        return self._model

    def assembleModel(self):
        """ Construct the tensorflow model """
        modelInput = tf.keras.Input(
            shape=(self._numFeatures,),
            dtype=tf.float32,
            name="inputMLP")
        x = IdentityLayer("identityA")(modelInput)

        # Add Dense Layers
        for i,nodes in enumerate(self._layerWidths):
            x = tf.keras.layers.Dense(
                units=nodes,
                activation=self._activationFunction,
                name="denseMlp" + str(i))(x)

        if (self._numClasses is not None):
            x = tf.keras.layers.Dense(
                units=self._numClasses,
                activation='softmax',
                name="outputMlp")(x)

        # Assemble the Model
        self._model = tf.keras.Model(inputs=modelInput,outputs=x,name=self._name)
        return self

class TensorflowConvolutionNeuralNetwork:
    """ Construct Tensorflow Convolutional Neural Network """

    def __init__(self,
                 inputShape,
                 numClasses=None,
                 filterSizes=[32,32],
                 kernelSizes=[(3,3),(3,3)],
                 poolSizes=[(3,3),(3,3)],
                 neurons=[64,64],
                 activationFunction='relu',
                 name="defaultCnn"):
        """ Constructor for TensorflowConvolutionNeuralNetwork Instance """
        self._inputShape = inputShape
        self._numClasses = numClasses
        self._filterSizes = filterSizes
        self._kernelSizes = kernelSizes
        self._poolSizes = poolSizes
        self._layerWidths = neurons
        self._activationFunction = activationFunction
        self._name = name
        self._model = None

    def __del__(self):
        """ Destructor for TensorflowConvolutionNeuralNetwork Instance """
        self._model = None

    def getModel(self):
        """ Return the Tensorflow Model Instance """
        return self._model

    def assembleModel(self):
        """ Construct the Tensorflow Model """
        modelInput = tf.keras.Input(
            shape=self._inputShape,
            dtype=tf.float32,
            name="inputCNN")
        x = IdentityLayer(name='identityCNN')(modelInput)

        # Add in filter layer groups
        for i,(filters,kernel,pool) in enumerate(zip(self._filterSizes,self._kernelSizes,self._poolSizes)):
            x = tf.keras.layers.Conv2D(filters=filters,kernel_size=(kernel),activation='relu',name='C'+str(i+1)+'A')(x)
            x = tf.keras.layers.Conv2D(filters=filters,kernel_size=(kernel),activation='relu',name='C'+str(i+1)+'B')(x)
            x = tf.keras.layers.MaxPool2D(pool_size=pool,name='P'+str(i+1))(x)

        # Flatter + Add Dense Nodes
        x = tf.keras.layers.Flatten(name="F1")(x)          
        for i,nodes in enumerate(self._layerWidths):
            x = tf.keras.layers.Dense(
                units=nodes,
                activation='relu',
                name='denseCnn'+str(i))(x)
        
        if (self._numClasses is not None):
            x = tf.keras.layers.Dense(
                units=self._numClasses,
                activation='softmax',
                name="outputCnn")(x)    
            
        self._model = tf.keras.Model(inputs=modelInput,outputs=x,name="ConvolutionalNetwork2D")
        return self

class HybridNeuralNetwork:
    """ Construct Tensorflow Hybrid Neural Network """

    def __init__(self,
                 numClasses,
                 tfMLP,
                 tfCNN,
                 neurons=[64,64]):
        """ Constructor for HybridNeuralNetwork Instance """
        self._numClasses = numClasses
        self._tfMLP = tfMLP
        self._tfCNN = tfCNN
        self._layerWidths = neurons
        self._model = None

    def __del__(self):
        """ Destructor for HybridNeuralNetwork Instance """
        self._tfMLP = None
        self._tfCNN = None
        self._model = None

    def getModel(self):
        """ Return the Tensorflow Model Instance """
        return self._model
    
    def assembleModel(self):
        """ Create the two sub-models + join then """
        self._tfMLP.assembleModel()
        self._tfCNN.assembleModel()
        modelMLP = self._tfMLP.getModel()
        modelCNN = self._tfCNN.getModel()

        # Join the Models
        x = tf.keras.layers.concatenate(
            inputs=[modelMLP.output,modelCNN.output],
            axis=-1,
            name='concat')

        # Add Dense Layers
        for i,nodes in enumerate(self._layerWidths):
            x = tf.keras.layers.Dense(
                units=nodes,
                activation='relu',
                name='denseHnn'+str(i))(x)

        # Create the output layer
        x = tf.keras.layers.Dense(
            units=self._numClasses,
            activation='softmax',
            name="outputHybrid")(x)  

        self._model = tf.keras.Model(
            inputs=[modelMLP.input,modelCNN.input],
            outputs=x,
            name="ConvolutionalNetwork2D")

        return self
    