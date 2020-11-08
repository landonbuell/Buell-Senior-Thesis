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
import datetime

import tensorflow as tf
import tensorflow.keras as keras

        #### OBJECT DEFINITIONS ####


class AnalyzeModels:
    """
    Class Object to Analyze performance of model outputs using metrics
    --------------------------------
    modelName (str) : String indicating names for model
    datapath (str) : Local Directory path where input file is held
    n_classes (int) : number of discrete classes for models
    --------------------------------
    Return Instante of class
    """

    def __init__(self,modelName,datapath,n_classes):
        """ Initialize Class Object Instance """
        # Set time stamp
        dt_obj = datetime.datetime.now()
        startTime = dt_obj.isoformat(sep='.',timespec='auto')
        self.startTime = startTime.replace(':','.').replace('-','.')
        print("Time Stamp:",self.startTime)

        # Set Attributes
        self.modelName = modelName
        self.dataPath = datapath  
        self.n_classes = n_classes
        self.outputFile = self.modelName+"@ANALYSIS@"+self.startTime+".csv"

    def GetKeywordFiles(self,keyword):
        """ Find files in path w/ Keyword in name """
        pathList = []
        for item in os.listdir(self.dataPath):
            fullPath = os.path.join(self.dataPath,item)
            if (self.modelName in item) and (keyword in item):
                # matches criteria
                pathList.append(fullPath)
        return pathList

    def ExportMetrics(self,outputPath):
        """ Export Metrics Array to Local path """
        colNames = ["Accuracy","Precision","Recall","Loss"]
        outputFrame = pd.DataFrame(data=None,index=None)
        outputFrame["CLF"] = [x.split("\\")[-1] for x in self.predictionFiles]
        for i in range(len(colNames)):
            outputFrame[colNames[i]] = self.metricsArray[:,i]
        outputPath = os.path.join(outputPath,self.outputFile)
        outputFrame.to_csv(outputPath,index=False)

    def __Call__(self):
        """ Run Main program instance """
        # Get needed Files
        self.trainingFiles = self.GetKeywordFiles("@TRAINING-HISTORY@")
        self.predictionFiles = self.GetKeywordFiles("@PREDICTIONS@")
        self.classDictionary = self.GetKeywordFiles("Categories")[-1]

        nRows = len(self.predictionFiles)
        nCols = 4
        self.metricsArray = np.empty(shape=(nRows,nCols))

        # Iterate through predictions
        for i,predFile in enumerate(self.predictionFiles):

            # Gather all metrics for each Classfier
            data = pd.read_csv(predFile)
            labels = data["Int Label"].to_numpy()
            predns = data["Int Prediction"].to_numpy()
            scores = data["Confidence"].to_numpy()

            ComputeMetrics = ClassifierMetrics(self.n_classes,labels,predns,scores) 
            metrics = ComputeMetrics.MetricScores
            self.metricsArray[i] = metrics
       
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

        self.xOneHot = keras.utils.to_categorical(self.x,self.n_classes)
        self.yOneHot = keras.utils.to_categorical(self.y,self.n_classes)
     
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

    """
    Check Documentation for TensorFlow's Metrics
        These four Methods require updates!
    """

    def AccuracyScore (self):
        """ Compute Recall Score of Data """    
        Accr = keras.metrics.Accuracy()
        Accr.update_state(self.x,self.y)
        return Accr.result().numpy()

    def PrecisionScore(self):
        """ Compute Precision Score of Data """
        Prec = keras.metrics.Precision()
        Prec.update_state(self.x,self.y)
        return Prec.result().numpy()

    def RecallScore (self):
        """ Compute Recall Score of Data """
        Recl = keras.metrics.Recall()
        Recl.update_state(self.x,self.y)
        return Recl.result().numpy()

    def LossScore (self):
        """ Compute Loss Score of Data """
        Loss = keras.losses.categorical_crossentropy(self.xOneHot,self.yOneHot)
        return np.mean(Loss.numpy())

    @property
    def MetricScores(self):
        """ Collect All metric Scores in one arrays """
        _accr = self.AccuracyScore()
        _prec = self.PrecisionScore()
        _recl = self.RecallScore()
        _loss = self.LossScore()
        return np.array([_accr,_prec,_recl,_loss])

    ### Include prediction accuracy
    ### Include prediction threshold

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
        
