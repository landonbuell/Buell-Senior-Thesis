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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
import sklearn.feature_selection as feat_sel

import tensorflow.keras as keras
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

class DataAnalyzer :
    """
    Analyze Desgin Matrix from local CSV file 
    --------------------------------

    --------------------------------

    """

    def __init__(self,filename,n_classes):
        """ Initialize Class Object Instance """
        self.filename = filename
        self.n_classes = n_classes
        

    def LoadDataFrame (self):
        """ Read Local Data Frame-Like Object """
        frame = pd.read_csv(self.filename)
        self.X = frame.drop(['Class'],axis=1).to_numpy()
        self.Y = frame['Class'].to_numpy()       
        self.n_samples = self.Y.shape[0]
        self.n_features = self.X.shape[1]
        return self

    def ScaleDesignMatrix (self):
        """ Scale Design Matrix s.t. Cols have unit variance """
        Scaler = StandardScaler()
        Scaler.fit(self.X)
        self.X = Scaler.transform(self.X)
        return self

    def OneHotEncode (self):
        """ One-Hot-Encode Target Vector Y """
        self.Y = keras.utils.to_categorical(self.Y,self.n_classes)
        return self

    def TrainTestSplit (self,testsize=0.2):
        """ Split data into training & testing data sets """
        return train_test_split(self.X,self.Y,test_size=testsize)

    def ComputeVariance (self):
        """ Compute Variance of Each Design Matrix Column """
        self.variances = np.var(self.X,axis=0)

    def SelectKBestFeatures (self,k=8):
        """ Select K-Bets Features in Design Matric based ofn 'func' """
        self.FeatureSelector = SelectKBest(score_func=feat_sel.f_classif,k=k)
        self.X = self.FeatureSelector.fit_transform(self.X,self.Y)
        self.n_features = k
        return self

    def __call__(self,scale=True):
        """ Call Data Analyzer Object """
        self.LoadDataFrame()
        self.SelectKBestFeatures()
        self.OneHotEncode()
        if scale == True:
            self.ScaleDesignMatrix()    
        return self

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
        featureVector = np.append(featureVector,timeSeriesFeatures.WaveformDistributionData())
        featureVector = np.append(featureVector,timeSeriesFeatures.AutoCorrelationCoefficients())

        freqSeriesFeatures = feat_utils.FrequencySeriesFeatures(file.waveform,frames=timeSeriesFeatures.frames)
        featureVector = np.append(featureVector,np.avg(freqSeriesFeatures.CenterOfMass('spectrogram')))

        featureVector = np.append(featureVector,np.avg())
        return featureVector

    def __call__(self):
        """ Execute FileIterator Object """
        designMatrix = np.array([])
        for i,file in enumerate(self.FILES):  # Each fileobject          
            self.LoopCounter(i,self.n_files,file.filename)
            x = self.FeatureExtractor(file)             # get feature vector
            designMatrix = np.append(designMatrix,x)    # add samples to design matrix
            del(x)
        designMatrix = designMatrix.reshape(self.n_files,-1)
        self.X = designMatrix
        return self

    def ExportData (self,filename):
        """ Export Design Matrix to CSV """
        cols = ['TDE','ZXR','COM','Mean','Med','Var','ACC1','ACC2','ACC3','ACC4']
        dataframe = pd.DataFrame(data=self.X,columns=cols)
        dataframe['Class'] = [x.target for x in self.FILES]
        dataframe.to_csv(filename+'.csv')
        return self

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

class NeuralNetworks :
    """
    Class containing Static Methods to create various neural network models
    """

    @staticmethod
    def Multilayer_Perceptron (name,n_classes,n_features,layerunits=[40,40],
                               metrics=['Precision','Recall']):
        """
        Create Mutlilayer Perceptron and set object as attribute
        --------------------------------
        name (str) : Name to attatch to Network Model
        n_classes (int) : Number of unique output classes
        n_features (int) : Number of input features into Network
        layerunits (iter) : List-like of ints. I-th element is nodes in I-th hiddenlayer
        metrics (iter) : Array-like of strs contraining metrics to track
        --------------------------------
        Return Compiled, unfit model instance
        """
        model = keras.models.Sequential(name=name)      # create instance & attactch name
        model.add(keras.layers.InputLayer(input_shape=n_features,name='Input')) # input layer
        
        # Add Hidden Dense Layers
        for i,nodes in enumerate(layerunits):           # Each hidden layer
            model.add(keras.layers.Dense(units=nodes,activation='relu',name='D'+str(i+1)))
        # Add Output Layer
        model.add(keras.layers.Dense(units=n_classes,activation='softmax',name='Output'))

        # Compile, Summary & Return
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
                      loss=keras.losses.CategoricalCrossentropy(),
                      metrics=metrics)
        print(model.summary())
        return model
