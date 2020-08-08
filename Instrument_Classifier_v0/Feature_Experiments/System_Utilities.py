"""
Landon Buell
PHYS 799
Instrument Classifier v0
10 June 2020
"""

            #### IMPORTS ####

import numpy as np
import pandas as pd
import os
import sys
import datetime

import scipy.io.wavfile as sciowav
import Feature_Utilities as feat_utils


"""
Program_Utilities.py - "Program Utilities"
    Contains Variables, Classes, and Definitions for Lower program functions
    Backends, Data structure objects, os & directory organization and validations

"""

            #### DATA STRUCTURE CLASSES ####

class FileObject:
    """
    Create File 
    --------------------------------
    datarow (arr) : array of shape (1 x 3):
        has form:
        | Fullpath  | Target Int    | Target Str |
    --------------------------------
    Return instatiated file_object class
    """

    def __init__(self,datarow):
        """ Initialize Object Instance """
        self.fullpath = datarow[0]      # set full file path
        try:
            self.target = int(datarow[1])   # target as int           
        except:
            self.target = None              # no label
        dir_tree = self.fullpath.split('/')
        self.filename = dir_tree[-1]    # filename

    def AssignTarget (self,target):
        """ Assign Target value to instance """
        self.target = target    # set y
        return self             # return self

    def ReadAudio (self):
        """ Read raw data from local path """
        rate,data = sciowav.read(self.fullpath)    
        data = data.reshape(1,-1).ravel()   # flatten waveform
        self.rate = rate            # set sample rate
        self.waveform = data/np.abs(np.amax(data))
        self.n_samples = len(self.waveform)
        return self             # return self

            #### PROGRAM PROCESSING CLASSES ####

class FileIterator :
    """
    Iterate through batches of file objects and Extract Features
    --------------------------------
    FILES (list) : List of FileObject Instances
    n_classes (int) : Number of discrete classes in data
    groupSize (int) : FileObjects to use in each mega-batches
    n_iters (int) : Number of passes over the full data set
    --------------------------------
    Return Initialize FileIterator Object
    """
    
    def __init__(self,FILES,n_classes,groupSize=256,n_iters=4):
        """ Initialize Class Object Instance """
        self.FILES = FILES
        self.n_classes = n_classes
        self.groupSize=groupSize
        self.n_files = len(self.FILES)
        self.groupCounter = 0

    def LoopCounter (self,cntr,max,text):
        """ Print Loop Counter for User """
        print('\t\t('+str(cntr)+'/'+str(max)+')',text)
        return None

    def FeatureExtractor (self,file):
        """ Extract Features From single file object """
        file.ReadAudio()                # read audio, get waveform & sample rate
        featureVector = np.array([])    # array to hold all features
        
        timeSeriesFeatures = feat_utils.TimeSeriesFeatures(file.waveform)

        featureVector = np.append(featureVector,timeSeriesFeatures.TimeDomainEnvelope())
        featureVector = np.append(featureVector,timeSeriesFeatures.ZeroCrossingRate())
        featureVector = np.append(featureVector,timeSeriesFeatures.CenterOfMass())
        featureVector = np.append(featureVector,timeSeriesFeatures.WaveformDistribution())
        featureVector = np.append(featureVector,timeSeriesFeatures.AutoCorrelationCoefficients())

        return featureVector

    def __call__(self):
        """ Execute FileIterator Object """
        designMatrix = np.array([])
        for i,file in enumerate(self.FILES):  # Each fileobject          
            self.LoopCounter(i,self.n_files,file.filename)
            x = self.FeatureExtractor(file)             # get feature vector
            designMatrix = np.append(designMatrix,x)    # add samples to design matrix
        designMatrix = designMatrix.reshape(self.n_files,-1)
        self.X = designMatrix
        return self

    def ExportData (self,filename):
        """ Export Design Matrix to CSV """
        cols = ['TDE','ZXR','COM','Mean','Med','Mode','Var','ACC0','ACC1','ACC2','ACC3']
        dataframe = pd.DataFrame(data=self.X,columns=cols)
        dataframe['Class'] = [x.target for x in self.FILES]
        dataframe.to_csv(filename+'.csv')
        return self

class DataAnalyzer :
    """
    Analyze Desgin Matrix from local CSV file 

    """

    def __init__(self,filename,n_classes):
        """ Initialize Class Object Instance """
        self.filename = filename
        self.n_classes = n_classes
        self.frame = pd.read_csv(self.filename)
        self.n_rows = self.frame.shape[0]
        self.n_cols = self.frame.shape[1]

    def __call__(self):
        """ Call Data Analyzer Object """
        pass



class ProgramStart:
    """
    Object to handle all program preprocessing
    --------------------------------
    readpath (str) : Local path nagivating to where data is stored
    modelpath (str) : Local path to store data related to Nerual network Models
    exportpath (str) : Local path to export final information
    mode (str) : String indicating which mode to execute program with
    newmodels (bool): If True, create new Nueral Network Models
    --------------------------------
    Return Instantiated Program Start Class Instance
    """

    def __init__(self,readpath=None):
        """ Initialize Class Object Instance """
        dt_obj = datetime.datetime.now()
        self.starttime = dt_obj.isoformat(sep='_',timespec='auto').replace(':','.')
        self.readpath = readpath
        print("Time Stamp:",self.starttime)
                  
    @property
    def StartupMesseges (self):
        """ Print out Start up messeges to Console """
        print("Running Main Program.....")
        print("\t\tFound",self.n_files,"files to read")
        print("\t\tFound",self.n_classes,"classes to sort")
        print("\n")

    def __call__(self):
        """ Run Program Start Up Processes """        
        self.files = self.CollectCSVFiles()        # find CSV files
        fileobjects = self.CreateFileobjs()    # file all files
        self.n_files = len(fileobjects)        
        self.n_classes = self.GetNClasses(fileobjects)
        self.StartupMesseges           # Messages to User
        fileobjects = np.random.permutation(fileobjects)    # permute
        return fileobjects,self.n_classes

    def CollectCSVFiles (self,exts='.csv'):
        """ Walk through Local Path and File all files w/ extension """
        csv_files = []
        for roots,dirs,files in os.walk(self.readpath):  
            for file in files:                  
                if file.endswith(exts):       
                    csv_files.append(file)
        return csv_files

    def CreateFileobjs (self):
        """ Create list of File Objects """
        fileobjects = []                        # list of all file objects
        for file in self.files:                 # each CSV file
            fullpath = os.path.join(self.readpath,file) # make full path str
            frame = pd.read_csv(fullpath,index_col=0)   # load in CSV
            frame = frame.to_numpy()                    # make np arr   
            for row in frame:                           # each row
                # 'FileObject' class is defined above
                fileobjects.append(FileObject(row))    # add row to obj list
            del(frame)                          # del frame  
        return fileobjects                      # return list of insts

    def GetNClasses (self,fileobjects):
        """ Find Number of classes in target vector """
        y = [x.target for x in fileobjects]   # collect target from each file
        try:                        # Attempt
            n_classes = np.amax(y)  # maximum value is number of classes
            return n_classes + 1    # account for zero-index
        except Exception:           # failure?
            return None             # no classes?



