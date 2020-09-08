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
        self.datapath = datapath
        self.infile = infile
        self.n_classes = n_classes

        self.outfile = infile.replace("PREDICTIONS","ANALYSIS")
        self.full_inpath = os.path.join(self.datapath,self.infile)
        self.full_outpath = os.path.join(self.datapath,self.outfile)

    def AssignMetrics (self,metrics_list=[]):
        """ Assign metrics objects to list to self """
        self.metrics = metrics_list
        return self

    def ReadData(self):
        """ Read raw prediction data from local file """
        self.indata = pd.read_csv(self.full_inpath,header=0,index_col=0)
        self.truth = self.indata['Label'].to_numpy(dtype=np.int32)
        self.predictions = self.indata['Prediction'].to_numpy(dtype=np.int32)
        return self

    def InitOutput(self):
        """ Initialize Output File data """
        self.scores = {'Metric':[]}          # column to hold metrics
        for metric in self.metrics:     # each metric:
            self.scores['Metric'].append(metric.name)    # add metric to col
        for model in self.model_names:   # each model:
            self.scores.update({model:np.array([])})
        return self

    def WriteOutput (self):
        """ Write Metric scores to local file """
        raise NotImplementedError()

    def __Call__(self):
        """ Call Program Mode """
        
        self.ReadData()      
        confusion = tf.math.confusion_matrix(self.truth,self.predictions,self.n_classes)
        self.PlotConfusion(confusion,self.modelName)
        return self

        #### FUNCTIONS DEFINITIONS ####

    def PlotConfusion(self,X,title,show=True):
        """ Plot Confusion Matrix """
        plt.title(title,fontsize=40,weight='bold')
        plt.imshow(X,cmap=plt.cm.binary)
        plt.xticks(np.arange(0,self.n_classes,1))
        plt.yticks(np.arange(0,self.n_classes,1))
        if show == True:
            plt.show()

