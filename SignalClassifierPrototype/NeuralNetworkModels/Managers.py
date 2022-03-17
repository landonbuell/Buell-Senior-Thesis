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
from re import fullmatch
import sys

import numpy as np
import sklearn.preprocessing as prep
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

    shapeFeaturesA = [75]
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

    def __init__(self,batchesToProcess):
        """ Constructor for DatasetManager Instance """
        self._inputPath = Settings.inputPath
        self._batchesToProcess = batchesToProcess
        self._standardScalerA = prep.StandardScaler(copy=False)
        self._standardScalerB = prep.StandardScaler(copy=False)
        self._scalerParamsA = None
        self._scalerParamsB = None

    def __del__(self):
        """ Destructor for DatasetManager Instance """
        pass

    # Getters and Setters

    def getNumBatches(self):
        """ Get the Number of Batches Collected """
        return self._batchesToProcess.shape[0]

    def getScalerA(self):
        """ Return Scaler A """
        return self._standardScalerA

    def getScalerB(self):
        """ Return Scaler B """
        return self._standardScalerB

    # Public Interface

    def loadBatches(self,batches):
        """ Load in an Collection of batches """
        numSamples = len(batches) * Settings.batchSize
        matrixA = CommonStructures.DesignMatrix(numSamples, Settings.shapeFeaturesA)
        matrixB = CommonStructures.DesignMatrix(numSamples, Settings.shapeFeaturesB)
        sampleCounter = 0
        for idx in batches:
            subMatrices = self.loadBatch(idx)
            for sample in range( subMatrices[0].getNumSamples() ):
                matrixA._data[sampleCounter] = subMatrices[0]._data[sample]
                matrixA._tgts[sampleCounter] = subMatrices[0]._tgts[sample]
                matrixB._data[sampleCounter] = subMatrices[1]._data[sample]
                matrixB._tgts[sampleCounter] = subMatrices[1]._tgts[sample]
                sampleCounter += 1
        # Loaded all batches - Return them
        return [matrixA,matrixB]

    def loadBatch(self,batchIndex):
        """ Load a Batch of Data """
        fileY = "batch" + str(batchIndex) + "_Y.bin"
        fileA = "batch" + str(batchIndex) + "_Xa.bin"
        fileB = "batch" + str(batchIndex) + "_Xb.bin"

        pathY = os.path.join(self._inputPath,fileY)
        pathA = os.path.join(self._inputPath,fileA)
        pathB = os.path.join(self._inputPath,fileB)

        #Get Design Matricies
        print("\tLoading Batch {0:<4}...".format(batchIndex))
        matrixA = CommonStructures.DesignMatrix.deserialize(
            pathA,pathY,Settings.batchSize,Settings.shapeFeaturesA )
        matrixB = CommonStructures.DesignMatrix.deserialize(
            pathB,pathY,Settings.batchSize,Settings.shapeFeaturesB )
        return [matrixA,matrixB]

    def preprocessSamples(self,numSubsets=4):
        """ Fit All batches on the Preprocessor """
       
        subsetSize = self.getNumBatches() // numSubsets
        newShapeXb = DatasetManager.productOfCollection(Settings.shapeFeaturesB)
        # Get the Batches for each subset
        for subsetIndex in range(numSubsets):
            # Fetch the batches
            batchesInSubset = np.array([x for x in range(subsetIndex,self.getNumBatches(),numSubsets)])
            data = self.loadBatches(batchesInSubset)
            # Remove Samples w/ NaN + Inf Values
            mask = data[0].getMaskForNaNsAndInfs()
            remainingSamples = np.sum(mask)
            data[0].applyMask(mask)
            data[1].applyMask(mask)
            # Extract Remaining Samples       
            dataXa = data[0].getFeatures()
            dataXb = data[1].getFeatures().reshape((remainingSamples,newShapeXb))
            data.clear()
            # Fit the Scaler
            self._standardScalerA.fit(dataXa)
            self._standardScalerB.fit(dataXb)
        # Done scaling - Export Parameters
        self._scalerParamsA = DatasetManager.ScalingParameters("A",self._standardScalerA)
        self._scalerParamsB = DatasetManager.ScalingParameters("B",self._standardScalerB)
        self._scalerParamsA.export(os.path.join(self._inputPath,"scalerParamsA.txt"))
        self._scalerParamsB.export(os.path.join(self._inputPath,"scalerParamsB.txt"))
        return self

    # Private Interface

    @staticmethod
    def productOfCollection(collection):
        """ Compute the product of all elements in a sequence """
        result = 1;
        for item in collection:
            result *= item;
        return result

    class ScalingParameters:
        """ Class To Hold Scaling Params for Future Use """
        
        def __init__(self,identifier,scaler):
            """ Constructor for Scaling Parameters Instance """
            self._id = identifier
            self._numFeatures = scaler.n_features_in_
            self._means = scaler.mean_
            self._varis = scaler.var_

        def export(self,fileName):
            """ Export These Parameters to Disk """
            with open(fileName,"w") as outputFile:
                # Write Means
                for i in range(self._numFeatures):
                    outputFile.write(str(self._means[i]) + "\t")
                outputFile.write("\n")
                # Write Variances
                for i in range(self._numFeatures):
                    outputFile.write(str(self._varis[i]) + "\t")
            return None

class TrainingManager:
    """ Class to Manage the Model's Training Process """

    def __init__(self,model):
        """ Constructor for TrainingManager Instance """
        self._model = model
        self._numTrainingSamples = 0
        self._numTestingSamples = 0