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

class Analyze_Models:
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

    def __init__(self,model_names,datapath,infile,n_classes):
        """ Initialize Class Object Instance """
        self.model_names = model_names
        self.datapath = datapath
        self.infile = infile
        self.n_classes = n_classes

        self.outfile = 'ANALYSIS@'+self.infile.split('@')[-1]
        self.full_inpath = os.path.join(self.datapath,self.infile)
        self.full_outpath = os.path.join(self.datapath,self.outfile)

    def assign_metrics (self,metrics_list=[]):
        """ Assign metrics objects to list to self """
        self.metrics = metrics_list
        return self

    def read_data(self):
        """ Read raw prediction data from local file """
        self.indata = pd.read_csv(self.full_inpath,header=0,index_col=0)
        self.truth = self.indata['Label'].to_numpy(dtype=np.int32)
        return self

    def init_output(self):
        """ Initialize Output File data """
        self.scores = {'Metric':[]}          # column to hold metrics
        for metric in self.metrics:     # each metric:
            self.scores['Metric'].append(metric.name)    # add metric to col
        for model in self.model_names:   # each model:
            self.scores.update({model:np.array([])})
        return self


    def write_output (self):
        """ Write Metric scores to local file """
        pass

    def __call__(self):
        """ Call Program Mode """
        self.init_output()
        for model in self.model_names:      # each model
            model_pred = self.indata[model].to_numpy(dtype=np.int32)
            confusion = tf.math.confusion_matrix(self.truth,model_pred,
                                                 self.n_classes)
            self.Plot_Confusion(confusion,model)

        return self

        #### FUNCTIONS DEFINITIONS ####

    def Plot_Confusion(self,X,title,show=True):
        """ Plot Confusion Matrix """
        plt.title(title,fontsize=40,weight='bold')
        plt.imshow(X,cmap=plt.cm.binary)
        plt.xticks(np.arange(0,self.n_classes,1))
        plt.yticks(np.arange(0,self.n_classes,1))
        if show == True:
            plt.show()

