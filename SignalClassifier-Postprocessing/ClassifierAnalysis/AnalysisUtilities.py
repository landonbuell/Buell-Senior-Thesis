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
        self.outputFile = self.modelName+"@"+self.startTime+"@ANALYSIS"+".csv"

    def MakeDecodeDictionary(self):
        """ Make Decoder Dictionary From Local Path """
        decodeFileName = self.modelName + "Categories.csv"
        decodeFullPath = os.path.join(self.dataPath,decodeFileName)
        decoder = {}
        encoder = {}
        rawData = pd.read_csv(decodeFullPath)
        Ints,Strs = rawData.iloc[:,0],rawData.iloc[:,1]   
        self.counts = rawData.iloc[:,2]
        for Int,Str in zip(Ints,Strs):          # iterate by each
            encoder.update({str(Str):int(Int)})
            decoder.update({int(Int):str(Str)})
        return encoder,decoder

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
        colNames = ["Accuracy","Precision","Recall"]
        outputFrame = pd.DataFrame(data=None,index=None)
        outputFrame["CLF"] = [x.split("\\")[-1] for x in self.predictionFiles]
        for i in range(len(colNames)):
            outputFrame[colNames[i]] = self.metricsArray[:,i]
        outputPath = os.path.join(outputPath,self.outputFile)
        outputFrame.to_csv(outputPath,index=False)

    def __Call__(self,exptPath):
        """ Run Main program instance """
        # Get needed Files
        self.trainingFiles = self.GetKeywordFiles("@TRAINING-HISTORY")
        self.predictionFiles = self.GetKeywordFiles("@PREDICTIONS")
        self.classDictionary = self.GetKeywordFiles("Categories")[-1]
        homePath = os.getcwd()

        nRows = len(self.predictionFiles)
        nCols = 3
        self.metricsArray = np.empty(shape=(nRows,nCols))

        # Initialize 3 Average Confusion matrices
        avgStandardConfMat = np.zeros(shape=(self.n_classes,self.n_classes))
        avgHitsWeightedConfMat = np.zeros(shape=(self.n_classes,self.n_classes))
        avgScrsWeightedConfMat = np.zeros(shape=(self.n_classes,self.n_classes))

        # Iterate through predictions
        for i,predFile in enumerate(self.predictionFiles):

            # Gather all metrics for each Classfier
            data = pd.read_csv(predFile)
            labels = data["Int Label"].to_numpy()
            predns = data["Int Prediction"].to_numpy()
            scores = data["Confidence"].to_numpy()
            ComputeMetrics = ClassifierMetrics(self.n_classes,labels,predns,scores) 

            # Create & Names for Confusion matrices
            stdMatName =  self.modelName + str(i) + " Standard Confusion"
            hitsMatName =  self.modelName + str(i) + " Hits Weighted Confusion"
            scrsMatName =  self.modelName + str(i) + " Score Weighted Confusion"

            # Create the matricies
            standardConfMat = ComputeMetrics.StandardConfusion()
            hitsWghtConfMat = ComputeMetrics.HitsWeightedConfusion()
            scrsWghtConfMat = ComputeMetrics.ScoreWeightedConfusion()

            # Compute Metrics for this split
            
            metrics = ComputeMetrics.MetricScores(standardConfMat)
            metrics = np.mean(metrics,axis=1)
            self.metricsArray[i] = metrics

            # Add To average Conf Mats
            avgStandardConfMat += standardConfMat
            avgHitsWeightedConfMat += hitsWghtConfMat
            avgScrsWeightedConfMat += scrsWghtConfMat

        # scale Avg Conf Mats
        avgStandardConfMat /= nRows
        avgHitsWeightedConfMat /= nRows
        avgScrsWeightedConfMat /= nRows

        avgStdMatName =  self.modelName + " Avg Standard Confusion"
        avgHitMatName =  self.modelName + " Avg Hits Weighted Confusion"
        avgScrMatName =  self.modelName + " Avg Score Weighted Confusion"
        
        """
        ClassifierMetrics.ExportConfusion(avgStandardConfMat,avgStdMatName,exptPath)
        ClassifierMetrics.ExportConfusion(avgHitsWeightedConfMat,avgHitMatName,exptPath)
        ClassifierMetrics.ExportConfusion(avgScrsWeightedConfMat,avgScrMatName,exptPath)

        os.chdir(exptPath)
        ClassifierMetrics.PlotConfusion(avgStandardConfMat,self.n_classes,avgStdMatName)
        ClassifierMetrics.PlotConfusion(avgHitsWeightedConfMat,self.n_classes,avgHitMatName)
        ClassifierMetrics.PlotConfusion(avgScrsWeightedConfMat,self.n_classes,avgScrMatName)
        os.chdir(homePath)
        """

        # Compute metrics avg. across all classes
        self.ConfusionMatrix = avgStandardConfMat
       
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
     
    def HitsWeightedConfusion(self):
        """ Create a confusion matric weighted by occurance in each row """
        weightedMatrix = np.zeros(shape=(self.n_classes,self.n_classes),dtype=float)  # empty conf-mat
        standardMatrix = self.StandardConfusion()       # standard conf-mat
        sumRow = np.sum(standardMatrix,axis=1)          # sum by each row
        for i in range(self.n_classes):     # each row:
            if sumRow[i] != 0:               # not zero
                weightedMatrix[i] = standardMatrix[i] / sumRow[i]              
        return weightedMatrix                       # return the weighted matrix

    def ScoreWeightedConfusion(self):
        """ Create a confusion matric weighted by confidence in predictions """
        weightedMatrix = np.zeros(shape=(self.n_classes,self.n_classes),dtype=float)  # empty conf-mat
        standardMatrix = self.StandardConfusion()               # standard conf-mat
        sumRow = np.sum(standardMatrix,axis=1)                  # sum by each row
        for x,y,z in zip(self.x,self.y,self.z):                 # labels,predictions,scores
            weightedMatrix[x,y] += z                            # add confidence
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                if (standardMatrix[i,j] != 0.0):                                            # attempt to divide
                    weightedMatrix[i,j] /= standardMatrix[i,j]  # weight by occ.
                else:                               # zero dividion error
                    weightedMatrix[i,j] = 0.0       # set to zero
        return weightedMatrix                       # return the weighted matrix

    def StandardConfusion(self):
        """ Create a confusion matric weighted by confidence & occurance """
        standardMatrix = np.zeros(shape=(self.n_classes,self.n_classes),dtype=int)  # empty conf-mat
        for x,y in zip(self.x,self.y):                  # labels,predictions,scores
            standardMatrix[x,y] += 1.                   # add counter
        return standardMatrix

    def AccuracyScore (self,confMat):
        """ Compute Recall Score of Data """    
        accy = np.zeros(shape=self.n_classes)   # store accuracy scores
        for i in range(self.n_classes):         # number of classes
            accy[i] = confMat[i,i]              # True pos.
        _sum = np.sum(confMat)
        accy /= _sum                            # divide by all predictions
        return accy                             # return accuracy

    def PrecisionScore(self,confMat):
        """ Compute Precision Score of Data """
        sumRow = np.sum(confMat,axis=1)         # sum across rows
        prec = np.zeros(shape=self.n_classes)   # store precision scores
        for i in range(self.n_classes):         # each class
            if sumRow[i] != 0:                  # if not zero
                p = confMat[i,i]/sumRow[i]      # precision of class i
            else:                               # div by zero
                p = 0                           # zero
            prec[i] += p                        # add to total
        return prec                             # return precision
       
    def RecallScore (self,confMat):
        """ Compute Recall Score of Data """
        sumCol = np.sum(confMat,axis=0)         # sum across cols
        recl = np.zeros(shape=self.n_classes)   # store precision scores
        for i in range(self.n_classes):         # each class
            if sumCol[i] != 0:                  # not zero
                r = confMat[i,i]/sumCol[i]      # recall of class i
            else:                               # div by zero?
                r = 0                           # set to 0
            recl[i]+= r                         # add to total
        return recl                             # avg over classes

    def MetricScores(self,confMat):
        """ Collect All metric Scores in one arrays """
        scoreMatrix = np.zeros(shape=(3,self.n_classes))
        scoreMatrix[0] = self.AccuracyScore(confMat)
        scoreMatrix[1] = self.PrecisionScore(confMat)
        scoreMatrix[2] = self.RecallScore(confMat)
        return scoreMatrix

    ### Include prediction threshold???

    @staticmethod
    def PlotConfusion(X,n_classes,title="",show=True,save=True):
        """ Visualize Confusion with ColorMap """
        plt.figure(figsize=(16,16))
        plt.title(title,fontsize=20,weight='bold')
        plt.imshow(X,cmap=plt.cm.jet)
        plt.xticks(np.arange(0,n_classes,1),weight='bold')
        plt.yticks(np.arange(0,n_classes,1),weight='bold')
        plt.tight_layout()
        if save == True:
            plt.savefig(title.replace(" ","_")+".png")
        if show == True:
            plt.show()
        plt.close()
        return None


    @staticmethod
    def PlotMetricsForClass(X,namesave="",show=True,save=True):
        """ Plot Accuracy, Precision, Recall for A Given Class """
        plt.figure(figsize=(8,6))
        plt.bar(["Accuracy","Precision","Recall"],height=X,
                width=0.4,color=['r','g','b'])

        plt.yticks(np.arange(0,1.1,0.1),size=20,weight='bold')
        plt.xticks(size=20,weight='bold')

        plt.grid()
        plt.tight_layout()
        if save == True:
            plt.savefig(namesave.replace(" ","_")+".png")
        if show == True:
            plt.show()
        plt.close()
        return None

    @staticmethod
    def ExportConfusion(X,fileName,filePath):
        """ Export Confusion Matrix to .csv file """
        X = pd.DataFrame(data=X)
        fullpath = os.path.join(filePath,fileName+".csv")
        X.to_csv(fullpath,header=None,index=None,mode='w')
        return None
        
