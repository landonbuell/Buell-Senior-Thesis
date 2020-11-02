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

class CompareCrossValidations :
    """
    Compare Multiple models that have had K-Folds Cross Validation applied
    --------------------------------
    modelNames (iter) : List of of string giving model names
    --------------------------------
    Return New Instance of Class
    """

    def __init__(self,modelNames,parentPath,n_classes):
        """ Initialize CompareCrossValidations Instance """
        self.modelNames
        self.parentPath

    def __Call__(self):
        """ Run Main Execution of this Class """

        for model in self.modelNames:                       # each model
            modelPath = os.path.join(self.parentPath,model) # get the path
            predcitionFiles = self.GetPredictions()

    def GetPredictions (self,path):
        """ Get Prediction Files """
        keyword = "@PREDICTIONS@"
        files = []
        for file in os.listdir(path):   # in this path
            if keyword in file:
                files.append(file)      # add to list
        return files                    # return the list

class AnalyzeModels:
    """
    Class Object to Analyze performance of model outputs using metrics
    --------------------------------
    modelName (iter) : list od strings indicating names for models
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
        
        print("File:",self.inFile)
        self.ReadData()             # get   
        labels = self.frame['Int Label']
        predictions = self.frame['Int Prediction']
        confidence = self.frame['Confidence']

        # Create Confusion matricies
        Metrics = ClassifierMetrics(self.n_classes,labels,predictions,confidence)
        self.weightedConfusion = Metrics.WeightedConfusion()
        self.standardConfusion = Metrics.StandardConfusion()

        # Compute Metrics
        print("\tLoss:",Metrics.LossScore())
        print("\tPrecision:",Metrics.PrecisionScore())
        print("\tRecall:",Metrics.RecallScore())

        return self

class ClassifierMetrics :
    """
    Create Confusion Matricies 
    --------------------------------
    The confusion matrix, 'C' for a k-classes classifier is a k x k array:
    C[i,j] = number of samples that belong to class i, 
        and were predicted to be in class j
    --------------------------------
    """

    def __init__(self,n_classes,labels,predictions,scores):
        """ Initialize ClassifierMetric Instance """
        self.n_classes = n_classes
        self.x = labels
        self.y = predictions
        self.z = scores
     
    def WeightedConfusion(self):
        """ Create a confusion matric weighted by confidence & occurance """
        weightedMatrix = np.zeros(shape=(self.n_classes,self.n_classes),dtype=float)  # empty conf-mat
        standardMatrix = self.StandardConfusion()               # standard conf-mat
        for x,y,z in zip(self.x,self.y,self.z):                 # labels,predictions,scores
            weightedMatrix[x,y] += z                            # add confidence
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                if (standardMatrix[i,j] != 0.0):                                            # attempt to divide
                    weightedMatrix[i,j] /= standardMatrix[i,j]  # weight by occ.
                else:                                # zero dividion error
                    weightedMatrix[i,j] = 0.0       # set to zero
        return weightedMatrix                       # return the weighted matrix

    def StandardConfusion(self):
        """ Create a confusion matric weighted by confidence & occurance """
        standardMatrix = np.zeros(shape=(self.n_classes,self.n_classes),dtype=float)  # empty conf-mat
        for x,y in zip(self.x,self.y):                  # labels,predictions,scores
            standardMatrix[x,y] += 1.                   # add counter
        return standardMatrix

    def PrecisionScore(self):
        """ Compute Precision Score of Data """
        _labs = keras.utils.to_categorical(self.x,self.n_classes)
        _prds = keras.utils.to_categorical(self.y,self.n_classes)
        Prec = keras.metrics.Precision()
        Prec.update_state(_labs,_prds)
        return Prec.result().numpy()

    def RecallScore (self):
        """ Compute Recall Score of Data """
        _labs = keras.utils.to_categorical(self.x,self.n_classes)
        _prds = keras.utils.to_categorical(self.y,self.n_classes)
        Recl = keras.metrics.Recall()
        Recl.update_state(_labs,_prds)
        return Recl.result().numpy()

    def LossScore (self):
        """ Compute Loss Score of Data """
        _labs = keras.utils.to_categorical(self.x,self.n_classes)
        _prds = keras.utils.to_categorical(self.y,self.n_classes)
        _loss = keras.losses.categorical_crossentropy(_labs,_prds).numpy()
        return np.mean(_loss)

    @staticmethod
    def PlotConfusion(X,n_classes,title="",show=True,save=True):
        """ Visualize Confusion with ColorMap """
        plt.title(title,fontsize=20,weight='bold')
        plt.imshow(X,cmap=plt.cm.jet)
        plt.xticks(np.arange(0,n_classes,1))
        plt.yticks(np.arange(0,n_classes,1))      
        if save == True:
            plt.savefig(title.replace(" ","_")+".png")
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
        
