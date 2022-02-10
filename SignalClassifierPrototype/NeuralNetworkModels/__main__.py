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

import tensorflow as tf

import NeuralNetworkModels

    #### MAIN EXECUTABLE ####

if __name__ == "__main__":


    # Create a Multilayer Perceptron
    mlpBuilder = NeuralNetworkModels.TensorflowMultilayerPerceptron(
        numFeatures=76,
        numClasses=None,
        neurons=[96,96,96,64])

    # Create a Convolution Neural Network
    cnnBuilder = NeuralNetworkModels.TensorflowConvolutionNeuralNetwork(
        inputShape=(1114,256,1),
        numClasses=None,
        filterSizes=[32,32,32,32],
        kernelSizes=[(3,3),(3,3),(3,3),(3,3)],
        poolSizes=[(3,3),(3,3),(3,3),(3,3)],
        neurons=[96,96,64,64])

    # Create the Hybrid Model
    hybridBuilder = NeuralNetworkModels.HybridNeuralNetwork(
        numClasses=34,
        tfMLP=mlpBuilder,
        tfCNN=cnnBuilder,
        neurons=[96,96,64,64])
    hybridBuilder.assembleModel()


    hybridModel = hybridBuilder.getModel()
    hybridModel.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.Accuracy(),
                 tf.keras.metrics.Precision(),
                 tf.keras.metrics.Recall()])

    hybridModel.summary()

    sys.exit(0)