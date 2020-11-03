"""
Landon Buell
PHYS 799
Instrument Classifier v0
12 June 2020
"""

            #### IMPORTS ####

import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import SystemUtilities as sys_utils
import FeatureUtilities as feat_utils
import PlottingUtilities as plot_utils
import MachineLearningUtilities as ML_utils
import NeuralNetworkUtilities as NN_utils


"""
ModeUtilities.py - 'Program Mode Utilities'
    Contains Definitions that are only called directly from MAIN script
    Functions are large & perform Groups of important operations
"""

            #### FUNCTION DEFINITIONS ####  

class ProgramMode:
    """
    Base Program mode object from which all programs objects inherit from
    --------------------------------
    FILEOBJS (iter) : List-like of File_object instances
    modelName (str) : user-ID-able string to indicate model
    n_classes (int) : number of discrete classes for models
    timestamp (str) : String representing time of program start
    exportpath (str) : Local Path to output filedata  
    groupSize (int) : number of file samples in each design matrix
    --------------------------------
    Execute MAIN program in perscribed mode
    """
    def __init__(self,FILEOBJS,modelName,n_classes,timestamp,exportpath=None,
                 groupSize=256):
        """ Inititialize Class Object Instance """
        self.FILEOBJS = FILEOBJS            # file objects in use
        self.modelName = modelName          # name of Neural network
        self.n_classes = n_classes          # number of classes
        self.timestamp = timestamp          # time when program began
        self.exportpath = exportpath        # path to push results to       
        self.groupSize = groupSize          # giles to use in each mega-batch
        self.n_files = len(self.FILEOBJS)   # number of file objects
        self.groupCounter = 0

        self.Scaler = StandardScaler()      # design matrix scaler
    
    def LoopCounter (self,cntr,max,text):
        """ Print Loop Counter for User """
        print('\t\t('+str(cntr)+'/'+str(max)+')',text)
        return None

    def ScaleData (self,matrixObj):
        """ Scale Design Matrix 'X' for processing """
        X = matrixObj.__Get_X__()           # get raw data
        self.Scaler.partial_fit(X,y=None)   # fit the matrix
        X_new = self.Scaler.transform(X)    # scale & return
        matrixObj.SetMatrixData(X_new)      # set as attrb
        return matrixObj                    # retuen updaed inst.

    def CollectFeatures (self,fileobj):
        """ Collected Features from a Given .WAV file object"""
        fileobj = fileobj.ReadFileWAV()             # read raw .wav file
        featureVector = ML_utils.FeatureArray(fileobj.targetInt)

        # Create Feature vector for MLP Branch
        timeFeatures = feat_utils.TimeSeriesFeatures(fileobj.waveform)  # collect time-domain features  
        featureVector.AddFeatures(timeFeatures.__Call__())              # and time-domain features
        freqFeatures = feat_utils.FrequencySeriesFeatures(timeFeatures.signal)
        featureVector.AddFeatures(freqFeatures.__Call__())              # add frequency-domain features
            
        # Create Spectrogram Matrix for CNN Branch
        featureMatrix = ML_utils.FeatureArray(fileobj.targetInt)        # strucutre to hold Sxx features
        featureMatrix = featureMatrix.SetFeatures(freqFeatures.spectrogram)   # set spectrogram

        del(timeFeatures)
        del(freqFeatures)
        return (featureMatrix,featureVector)    # return the feature arrays

    def ConstructDesignMatrices (self,FILES):
        """ Collect Features from a subset File Objects """

        # Intitialize Design Matricies
        X1 = ML_utils.DesignMatrix(ndim=4,n_classes=self.n_classes)  # Design matrix for Spectrogram
        X2 = ML_utils.DesignMatrix(ndim=2,n_classes=self.n_classes)  # Design matrix for MLP
        
        # Add Samples to Design Matricies
        for i,FILEOBJ in enumerate(FILES):
            self.LoopCounter(i,len(FILES),FILEOBJ.filename) # print messege
            (x1,x2) = self.CollectFeatures(FILEOBJ)         # collect features
            X1.AddSample(x1)            # Add sample to Sxx 
            X2.AddSample(x2)            # Add sample to MLP
       
        # Format Design Matricies for Input
        X1 = X1.ShapeBySample()     # shape spectrogram matrix
        X1 = X1.AddChannel()        # add channel to Matrix
        X2 = X2.ShapeBySample()     # shape design matrix
        X2 = self.ScaleData(X2)     # scale design matrix

        return (X1,X2)      # return 2 Design matricies & target matrix

class TrainMode (ProgramMode):
    """
    Run Program in 'Train Mode'
        Inherits from 'Program_Mode' parent class
    --------------------------------
    FILEOBJS (iter) : List-like of File_object instances
    modelName (str) : user-ID-able string to indicate model
    n_classes (int) : number of discrete classes for models
    timestamp (str) : String representing time of program start
    exportpath (str) : Local Path to output filedata    
    groupSize (int) : number of file samples in each design matrix
    n_iters (int) : Indicates how may iterations to do over full dataset 
    --------------------------------
    Creates Program Train Mode Object
    """
    def __init__(self,FILEOBJS,modelName,n_classes,timestamp,exportpath=None,
                 groupSize=256,n_iters=2):
        """ Instantiate Class Method """
        self.programMode = "Train"
        super().__init__(FILEOBJS=FILEOBJS,modelName=modelName,n_classes=n_classes,
                         timestamp=timestamp,exportpath=exportpath,groupSize=groupSize)

        self.outfile = self.modelName+'@TRAINING-HISTORY@'+self.timestamp+'.csv'
        self.exportpath = os.path.join(self.exportpath,self.outfile)
        self.InitOutputStructure()
        self.n_iters = n_iters
        self.n_epochs = 4
     
    def __Call__(self,Networks):
        """ Call this Instance to Execute Training and Testing """
        print("\nBegining Training process....")
        for I in range (0,self.n_iters):    # Each pass over full dataset
            print("\tIteration:",I)         # Print interation num
            self.__TRAIN__(Networks)        # Train the model
            Networks.SaveModel()            # save locally
            self.FILEOBJS = np.random.permutation(self.FILEOBJS)
        print("\tTraining Completed! =)")
        return self

    def __TRAIN__(self,Networks):
        """ Train Netorks on data from FILEOBJS """        
        
        for I in range (0,self.n_files,self.groupSize):    # In a given group
            print("\tGroup Number:",self.groupCounter)
            FILES = self.FILEOBJS[I:I+self.groupSize]      # subset of files
            (matrixSXX,matrixMLP) = self.ConstructDesignMatrices(FILES)
            X1 = matrixSXX.__Get_X__()
            X2 = matrixMLP.__Get_X__()
            Y = matrixSXX.__Get_Y__()

            modelHistory = Networks.MODEL.fit(x=[X1,X2],y=Y,
                               batch_size=32,epochs=self.n_epochs,verbose=1) 
            self.ExportHistory(modelHistory)

            del(matrixSXX,matrixMLP)    # delete Design Matrix Objs
            del(X1,X2,Y)                # delete Design Matrices
            self.groupCounter += 1      # incr coutner
        return self                     # self

    def InitOutputStructure (self):
        """ Create Output Structure for Testing/Prediction Mode """
        self.OutputData = sys_utils.OutputStructure(self.programMode,self.exportpath)
        return self

    def ExportHistory (self,historyObject):
        """ Store Keras History Object in lists """
        metricsDictionary = historyObject.history
        newKeys = self.OutputData.cols
        updateData = {}
        for key,val in zip(newKeys,metricsDictionary.values()):
            newpair = {key:val}
            updateData.update(newpair)
        self.OutputData.UpdateData(updateData)
        return self
        
class PredictMode (ProgramMode):
    """
    Run Program in 'Predict Mode'
        Inherits from 'ProgramMode' parent class
    --------------------------------
    FILEOBJS (iter) : List-like of File_object instances
    modelName (str) : user-ID-able string to indicate model
    n_classes (int) : number of discrete classes for models
    timestamp (str) : String representing time of program start
    exportpath (str) : Local Path to output filedata   
    groupSize (int) : number of file samples in each design matrix
    labels_present (bool) : If True, evaluation labels are given
    --------------------------------
    Creates Program Test Mode Object
    """
    def __init__(self,FILEOBJS,modelName,n_classes,timestamp,exportpath=None,
                 groupSize=256,labels_present=False,prediction_threshold=0.5):
        """ Initialize Class Object Instance """
        self.programMode = "Predict"
        super().__init__(FILEOBJS=FILEOBJS,modelName=modelName,
                         n_classes=n_classes,timestamp=timestamp,exportpath=exportpath,
                         groupSize=groupSize)

        outfile = self.modelName+'@PREDICTIONS@'+self.timestamp+'.csv'
        self.exportpath = os.path.join(self.exportpath,outfile)
        self.InitOutputStructure()
        self.predictionThreshold = prediction_threshold
        
    def __Call__(self,Networks):
        """ Call this Instance to Execute Training and Testing """
        print("\nBegining Testing Process...")
        self.__PREDICT__(Networks)
        print("\tTesting Completed! =)")

    def __PREDICT__(self,Networks):
        """ Test Netorks on data from FILEOBJS """
        # For Each group of files, Collect the data
        for I in range (0,self.n_files,self.groupSize):# In a given group
            print("\tGroup Number:",self.groupCounter)
            FILES = self.FILEOBJS[I:I+self.groupSize]  # subset of files
            (matrixSXX,matrixMLP) = self.ConstructDesignMatrices(FILES)
            X1 = matrixSXX.__Get_X__()
            X2 = matrixMLP.__Get_X__()

            modelPrediction = Networks.MODEL.predict(x=[X1,X2],batch_size=64,verbose=0)
            self.ExportPrediction(FILES,modelPrediction,Networks.classDecoder)
            
            del(matrixSXX,matrixMLP)        # delete Design Matrix Objs
            del(X1,X2)                      # delete Design Matricies
            self.groupCounter += 1          # incr counter
        return self                         # return self       

    def InitOutputStructure (self):
        """ Create Output Structure for Testing/Prediction Mode """
        self.OutputData = sys_utils.OutputStructure(self.programMode,self.exportpath)
        return self

    def ExportPrediction (self,fileObjects,predictionData,decoder):
        """ Export data from prediction arry to Local path """
        predictionInts = np.argmax(predictionData,axis=-1)
        predcitionStrs = [decoder[x] for x in predictionInts]
        confidences = np.max(predictionData,axis=-1)
        updateData = {  "Filepath":[x.fullpath for x in fileObjects],
                        "Int Label":[x.targetInt for x in fileObjects],
                        "Str Label":[x.targetStr for x in fileObjects],
                        "Int Prediction":predictionInts,
                        "Str Prediction":predcitionStrs,
                        "Confidence":confidences
                      }
        self.OutputData.UpdateData(updateData)
        return self
       
class TrainPredictMode (ProgramMode):
    """
    Run Program in 'Train_Mode' and 'Test Mode' sequentially
        Inherits from 'Train_Mode' and 'Test Mode' parent classes
    --------------------------------
    FILEOBJS (iter) : List-like of File_object instances
    modelName (str) : user-ID-able string to indicate model
    n_classes (int) : number of discrete classes for models
    timestamp (str) : String representing time of program start
    exportpath (str) : Local Path to output filedata
    groupSize (int) : number of file samples in each design matrix
    n_iters (int) : Indicates how may iterations to do over full dataset 
    testSize (float) : Value on interval (0,1) indicate fraction of data to test with
    --------------------------------
    Creates Program Test Mode Object
    """
    def __init__(self,FILEOBJS,modelName,n_classes,timestamp,exportpath='',
                 groupSize=256,n_iters=1,labelsPresent=True,testSize=0.1):
        """ Initialize Class Object Instance """
        super().__init__(FILEOBJS=FILEOBJS,modelName=modelName,
                         n_classes=n_classes,timestamp=timestamp,exportpath=exportpath,
                         groupSize=groupSize)
        
        self.labelsPresent = labelsPresent  # labels?
        self.n_iters = n_iters              # number of passes over data
        self.testSize = testSize            # train/test size        
        self.SplitObjs()                    # split objs
                  
    def SplitObjs (self):
        """ Split objects into training.testing subsets """
        train,test = train_test_split(self.FILEOBJS,test_size=self.testSize)
        delattr(self,'FILEOBJS')        # delete attrb
        self.TRAIN_FILEOBJS = train     # set attrb
        self.n_trainFiles = len(self.TRAIN_FILEOBJS)
        self.TEST_FILEOBJS = test       # set attrbs
        self.n_testFiles = len(self.TEST_FILEOBJS)
        return self                     # return self

    def __Call__(self,Networks):
        """ Call this Instance to Execute Training and Testing """

        # Run Training Mode
        Training = TrainMode(FILEOBJS=self.TRAIN_FILEOBJS,modelName=self.modelName ,  
                              n_classes=self.n_classes,timestamp=self.timestamp,
                              exportpath=self.exportpath,groupSize=self.groupSize,
                              n_iters=2)
        Training.__Call__(Networks)

        # Run Testing Mode
        Testing = PredictMode(FILEOBJS=self.TEST_FILEOBJS,modelName=self.modelName,
                              n_classes=self.n_classes,timestamp=self.timestamp,
                              exportpath=self.exportpath,groupSize=self.groupSize,
                              labels_present=True)
        Testing.__Call__(Networks)
            
        return self


   