"""
Landon Buell
PHYS 799
X-Validation Utilities
5 October 2020
"""

            #### IMPORTS ####

import os
import sys
import numpy as np
import pandas as pd

            #### CLASS OBJECT DEFINITIONS ####

class CrossValidationSplit:
    """ Apply Cross-Validation Split to Full data set """
    
    def __init__(self,fileObjs,filePath,nSplits=10):
        """ Initialize CrossValidationSplit Instance """
        self.fileObjects = fileObjs     # Add objects
        self.filePath = filePath        # full data is stored
        self.splitsPath = os.path.join(self.filePath,"ValidationSplits")
        self.K = nSplits                # number of splits
        self.ApplySplit()               # Dew it

    def ApplySplit(self):
        """ Split Data into 'K' equally sized groups """
        permuted = np.random.permutation(self.fileObjects)  # permute
        self.folds = [[] for i in range(self.K)]            # list to hold list of file objs
        for i in range(len(self.fileObjects)):      # each file
            _bin = i % self.K                       # get the bin for this sample
            self.folds[_bin].append(permuted[i])    # add sample to that bin
        return self         

    def GetTrainTestData (self,i):  
        """ Get Train / Test Data for i-th fold (i <= K) """
        Xtest = self.folds[i]
        Xtrain = []
        for j in range(self.K):     # each subset
            if i == j:              # this is test data
                pass                # skip it
            else:                   # otherwise
                Xtrain += self.folds[j] # add to train data
        return Xtrain,Xtest

    def ExportSplits(self,sampleList,outPath):
        """ Export Split Train/Test Data """
        cols = ["Fullpath","Target Int","Target Str"]
        data = {"Fullpath":[x.fullpath for x in sampleList],
                "Target Int":[x.targetInt for x in sampleList],
                "Target Str":[x.targetStr for x in sampleList]}
        # Make & Export Frame
        os.makedirs(outPath,exist_ok=True)      # make the outpath
        outPath = os.path.join(outPath,"X.csv")
        frame = pd.DataFrame(data=data,columns=cols)
        frame.to_csv(outPath,header=True,index=False,mode='w')
        return self

    def __Call__(self):
        """ Run Splitting Functions """
        for i in range(self.K):         # for each split
            Xtrain,Xtest = self.GetTrainTestData(i)
            # Export training Data
            pathName = os.path.join(self.splitsPath,"split"+str(i)+"train")
            self.ExportSplits(Xtrain,pathName)
            # Export testing Data
            pathName = os.path.join(self.splitsPath,"split"+str(i)+"test")
            self.ExportSplits(Xtest,pathName)
        return self

class CrossValidator :
    """ Run K- Folds Cross Validation on Data """

    def __init__(self,modelName,nSplits,scriptData,pathList):
        """ Initialize CrossValidator """
        self.modelName = modelName      # parent model name
        self.K = nSplits                # number of splits
        self.scriptPath = scriptData[0]    # where the Classifier scrip is located
        self.scriptName = scriptData[1]    # what the cript is called
        # Establish Paths
        self.dataPath = pathList[0]     # Classifiers gets data from here
        self.exportPath = pathList[1]   # Classifiers Exports data to here
        self.modelPath =  pathList[2]   # Classifiers are stored here
        
    def GetTrainArguments (self,iter):
        """ Get Command Line Arguments to Pass to Classifier Program for Training """
        readPath = os.path.join(self.dataPath,"Split"+str(iter)+"train")
        _sysArgs = ["python",self.scriptName]
        pathArgs = [readPath,self.exportPath,self.modelPath]
        progArgs = ['train',self.modelName+str(iter),"True"]
        _allArgs = _sysArgs + pathArgs + progArgs   # combine all agrs into single list
        _argsStr = " ".join(_allArgs)               # concat list args into str
        return _argsStr                      

    def GetTestArgumnets (self,iter):
        """ Get Command Line Arguments to Pass to Classifier Program """
        readPath = os.path.join(self.dataPath,"Split"+str(iter)+"test")
        _sysArgs = ["python",self.scriptName]
        pathArgs = [readPath,self.exportPath,self.modelPath]
        progArgs = ['predict',self.modelName+str(iter),"False"]
        _allArgs = _sysArgs + pathArgs + progArgs   # combine all agrs into single list
        _argsStr = " ".join(_allArgs)               # concat list args into str
        return _argsStr                  

    def __Call__(self,XValSplit,homePath):
        """ Run k-Folds Cross Validation """
        os.chdir(self.scriptPath)       # changed to script path
        startK = 0                          # whaere to start (incase error!)
        for i in range(startK,self.K,1):    # for each split
            # Create a New Model & Train
            print("")
            commandLineArgs = self.GetTrainArguments(i)
            #print(commandLineArgs)
            os.system(commandLineArgs)

            # Load Existing Model and & Test
            print("")
            commandLineArgs = self.GetTestArgumnets(i)
            print(commandLineArgs)
            os.system(commandLineArgs)

        os.chdir(homePath)
        return self




            
