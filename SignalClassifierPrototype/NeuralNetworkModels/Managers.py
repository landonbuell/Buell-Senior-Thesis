"""
Repository:     Buell-Senior-Thesis
Solution:       SignalClassifierPrototype
Project:        NeuralNetworkModels
File:           Managers.py
 
Author:         Landon Buell
Date:           January 2022
"""

    #### IMPORTS ####

import os
import sys

import numpy as np
import tensorflow as tf

import NeuralNetworkModels
import CommonStructures

    #### CLASS DEFINITIONS ####

class Settings:
    """ Hold all RunTime Constants in one Spot (STATIC) """

    inputPath = "C:\\Users\\lando\\Documents\\audioFeatures\\devTestV1"
    outputPath = "C:\\Users\\lando\\Documents\\audioFeatures\\modelTestv1"

    numSamples = 17599
    batches = np.arange(0,275,1)
    batchSize = 64

    shapeFeaturesA = [76]
    shapeFeaturesB = [1487,256,1]

    numClasses = 34
    mplDenseLayerWidths = [96,96,96,64]
    cnnDenseLayerWidths = [96,96,64,64]
    hnnDenseLayerWidths = [96,96,64,64]

    optimizer = tf.keras.optimizers.Adam()
    objective = tf.keras.losses.CategoricalCrossentropy()
    metrics = [ tf.keras.metrics.Accuracy(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()]

    @staticmethod
    def getShapeA():
        """ Get the Shape of Design Matrix A """
        return [Settings.batchSize] + Settings.shapeFeaturesA

    @staticmethod
    def getShapeB():
        """ Get the Shape of Design Matrix A """
        return [Settings.batchSize] + Settings.shapeFeaturesB

class TensorflowModelManager:
    """ Build and Organize Tensorflow Models """

    def __init__(self):
        """ Constructor for TensorflowModelManager Instance """
        self._model = None
        self._trainingHistory = []

    def __del__(self):
        """ Destructor for TensorflowModelManager Instance """
        pass

    # Getters and Setters

    def getTensorflowModel(self):
        """ Return the Tensorflow Model """
        return self._model

    # Public Interface

    def generateModel(self,compileModel=True):
        """ Run this Instance """
         # Create a Multilayer Perceptron
        mlpBuilder = NeuralNetworkModels.TensorflowMultilayerPerceptron(
            numFeatures=Settings.shapeFeaturesA[0],
            numClasses=None,
            neurons=Settings.mplDenseLayerWidths)

        # Create a Convolution Neural Network
        cnnBuilder = NeuralNetworkModels.TensorflowConvolutionNeuralNetwork(
            inputShape=Settings.shapeFeaturesB,
            numClasses=None,
            filterSizes=[32,32,32,32],
            kernelSizes=[(3,3),(3,3),(3,3),(3,3)],
            poolSizes=[(3,3),(3,3),(3,3),(3,3)],
            neurons=Settings.cnnDenseLayerWidths)

        # Create the Hybrid Model
        hybridBuilder = NeuralNetworkModels.HybridNeuralNetwork(
            numClasses=Settings.numClasses,
            tfMLP=mlpBuilder,
            tfCNN=cnnBuilder,
            neurons=Settings.hnnDenseLayerWidths)

        # Assamble + Compile
        hybridBuilder.assembleModel()

        self._model = hybridBuilder.getModel()
        if (compileModel == True):
            self._model.compile(
                optimizer=Settings.optimizer,
                loss=Settings.objective,
                metrics=Settings.metrics)
        return self._model




class DatasetManager:
    """ Load + Maintain Dataset for Duration of Program """

    def __init__(self):
        """ Constructor for DatasetManager Instance """
        self._inputPath = Settings.inputPath
        self._designMatrixA = None
        self._designMatrixB = None

    def __del__(self):
        """ Destructor for DatasetManager Instance """
        self._designMatrixA = None
        self._designMatrixB = None

    # Public Interface

    def fetchFullDataset(self):
        """ Load In Full dataset """


    def loadBatch(self,batchIndex):
        """ Load a Batch of Data """
        fileY = "batch" + str(batchIndex) + + "Y.bin"
        fileA = "batch" + str(batchIndex) + "_Xa.bin"
        fileB = "batch" + str(batchIndex) + "_Xb.bin"

        pathY = os.path.join(self._inputPath,fileY)
        pathA = os.path.join(self._inputPath,fileA)
        pathB = os.path.join(self._inputPath,fileB)

        #Get Design Matricies
        print("\tLoading Batch {0:<4}...".format(batchIndex))
        matrixA = CommonStructures.DesignMatrix.deserialize(
            pathA,pathY,Settings.batchSize,Settings.getShapeA() )
        matrixB = CommonStructures.DesignMatrix.deserialize(
            pathB,pathY,Settings.batchSize,Settings.getShapeB() )
        return [matrixA,matrixB]