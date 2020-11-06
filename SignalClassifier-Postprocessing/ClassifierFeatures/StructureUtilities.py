"""
Landon Buell
PHYS 799
Instrument Classifier v0
17 June 2020
"""

            #### IMPORTS ####
              
import numpy as np
import pandas as pd

import SystemUtilities as sys_utils
import FeatureUtilities as feat_utils

"""
StructureUtilities.py - "Structure Utilities"
    Contains Definitions for general mathematical functions
"""

            #### FUNCTION DEFINITIONS ####

class MathematicalUtilities :
    """
    Mathematical Utilites for feature processing
    --------------------------------
    * no args
    --------------------------------
    All methods are static
    """

    @staticmethod
    def PadZeros(X,outCols=256):
        """ Zero-pad 2D Array w/ columns """
        Xshape = X.shape                       # current array shape
        colDeficit = outCols - Xshape[-1]      # number of missing columns
        if colDeficit >= 1:                          # need to add cols
            zeroPad = np.zeros(shape=(Xshape[0],colDeficit),dtype=float)
            X = np.concatenate((X,zeroPad),axis=-1) # pad w/ zeroes
        else:                                       # otherwise
            X = X[:,:outCols]                       # remove cols
        return X                                    # return new Array

class FeatureContainer :
    """
    Hold features within a class in a large array
    """
    def __init__(self,classInt,classStr,fileObjs,nFeatures=20):
        """ Intialize FeatureContainer Instance """
        self.targetInt = classInt
        self.targetStr = classStr
        self.fileObjs = fileObjs        
        self.n_files = len(fileObjs)
        self.n_features = nFeatures
        self.matrixShape = (self.n_files,self.n_features)
        self.X = np.empty(shape=self.matrixShape,dtype=np.float64)

    def __Call__(self):
        """ Execute methods of this object """
        for i,file in enumerate(self.fileObjs):   # each file in the class
            print("\t"+file.filename)

            # Initialize
            x = np.array([])
            file.ReadFileWAV()
          
            # Collect features
            timeFeatures = feat_utils.TimeSeriesFeatures(file.waveform)
            freqFeatures = feat_utils.FrequencySeriesFeatures(file.waveform)
            x = np.append(x,timeFeatures.__Call__())
            x = np.append(x,freqFeatures.__Call__())
            self.X[i] = x

        self.InitOutput()
        return self

    @property
    def GetColumnNames(self):
        """ Geat Feature Names for DF col header """
        return ["FTR"+str(i) for i in range(self.n_features)]
   
    def InitOutput (self):
        """ Return [classInt,classStr,features] as array """
        # Prepare Frame of class Ints/Strs
        indexFrame = pd.DataFrame(data=None)
        indexFrame['Target Str'] = [str(self.targetStr) for x in range(self.n_files)]
        indexFrame['Target Int'] = [str(self.targetInt) for x in range(self.n_files)]

        # Prepare Frame of Design Matrix Data
        X = self.X
        featuresFrame = pd.DataFrame(data=X,columns=self.GetColumnNames)    # make dataFrame for features
        self.outputFrame = pd.concat([indexFrame,featuresFrame],axis=1)    # concatenate the frames
        return self

    def ExportFrame(self,exportPath):
        """ Export DataFrame """
        try:        # try to write frame
            self.outputFrame.to_csv(exportPath,index=False,header=False,mode='a')
        except:     # create frame
            raise FileNotFoundError()
        return self
            

