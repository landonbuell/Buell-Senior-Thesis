"""
Landon Buell
PHYS 799.32
Classifier Analysis Main
28 July 2020
"""

        #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

import tensorflow as tf
import tensorflow.keras as keras

        #### OBJECT DEFINITIONS ####

class AnalyzeModels:
    """
    Class Object to Analyze performance of model outputs using metrics
    --------------------------------
    model_names (iter) : list-like of 3 strings indicating names for models
    datapath (str) : Local Directory path where input file is held
    infile (str) : File name w/ ext indicating file to read
    n_classes (int) : number of discrete classes for models
    --------------------------------
    Return Instante of class
    """

    def __init__(self,modelName,datapath,infile,n_classes):
        """ Initialize Class Object Instance """
        self.modelName = modelName
        self.dataPath = datapath
        self.inFile = infile
        self.n_classes = n_classes

        self.outFile = infile.replace("PREDICTIONS","ANALYSIS")
        self.fullInpath = os.path.join(self.dataPath,self.inFile)
        self.fullOutpath = os.path.join(self.dataPath,self.outFile)

    def ReadData(self):
        """ Read raw prediction data from local file """
        self.frame = pd.read_csv(self.fullInpath)
        return self

    def __Call__(self):
        """ Call Program Mode """
        
        self.ReadData()             # get   
        labels = self.frame['Int Label']
        predictions = self.frame['Int Prediction']
        confidence = self.frame['Confidence']

        self.weightedConfusion = ConfusionMatrix.WeightedConfusion(self.n_classes,
                                                labels,predictions,confidence)
        self.standardConfusion = ConfusionMatrix.WeightedConfusion(self.n_classes,
                                                labels,predictions)
        return self

class ConfusionMatrix :
    """
    Create Confusion Matricies 
    --------------------------------
    The confusion matrix, 'C' for a k-classes classifier is a k x k array:
    C[i,j] = number of samples that belong to class i, 
        and were predicted to be in class j
    --------------------------------
    """

    @staticmethod
    def WeightedConfusion(n_classes,labels,preds,scores=None):
        """ Create a confusion matric weighted by confidence """
        matrix = np.zeros(shape=(n_classes,n_classes))      # empty conf-mat
        if scores is None:                                  # no weights given, use 1's
            scores = np.ones(shape=(len(labels)))      # weight with 1's
        for x,y,z in zip(labels,preds,scores):              # iterate through labs,preds,& scores
            matrix[x,y] += z                                # weight by confidence
        return matrix

    @staticmethod
    def PlotConfusion(n_classes,X,title="",show=True):
        """ Visualize Confusion with ColorMap """
        plt.title(title,fontsize=20,weight='bold')
        plt.imshow(X,cmap=plt.cm.binary)
        plt.xticks(np.arange(0,n_classes,1))
        plt.yticks(np.arange(0,n_classes,1))
        if show == True:
            plt.show()
        return None

    @staticmethod
    def ExportConfusion(X,fileName,filePath):
        """ Export Confusion Matrix to .csv file """
        X = pd.DataFrame(data=X)
        fullpath = os.path.join(filePath,fileName+".csv")
        X.to_csv(fullpath,header=None,index=None,mode='w')
        return None
        

class EncoderMap :
    """
    Get the Encder/Decoder Map for this classifier
    """

    def __init__(self):
        """ Initialize EncoderMap Instance """

