"""
Landon Buell
PHYS 799
Instrument Classifier 
12 June 2020
"""

            #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import os

import SystemUtilities as sys_utils
import FeatureUtilities as feat_utils
import MathUtilities as math_utils
import PlottingUtilities as plot_utils

"""
    Not Implemented!

"""

            #### MAIN EXECTUABLE ####

if __name__ == '__main__':

    homePath = "C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\SignalClassifierTools\\ClassifierFeatures"
    dataPath = "C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\SignalClassifier\\Target-Data"
    exptPath = "C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\Thesis\\Figures"
    
    ProgramSetup = sys_utils.ProgramInitializer([dataPath,homePath,exptPath])
    FILEOBJS = ProgramSetup.__Call__()
    Decoder = ProgramSetup.GetDecoder

    n_features = 20
    DesignMatrix = np.array([])
    TargetVector = np.array([])

    for fileObj in FILEOBJS[:256]:
        fileObj.ReadFileWAV()
        featureVector = np.array([])
        timeFeatures = feat_utils.TimeSeriesFeatures(fileObj.waveform)
        freqFeatures = feat_utils.FrequencySeriesFeatures(fileObj.waveform,presetFrames=timeFeatures.frames)
        featureVector = np.append(featureVector,timeFeatures.__Call__())
        featureVector = np.append(featureVector,freqFeatures.__Call__())

        DesignMatrix = np.append(DesignMatrix,featureVector)
        TargetVector = np.append(TargetVector,fileObj.targetInt)

    DesignMatrix = DesignMatrix.reshape(-1,n_features)
    DesignMatrix = math_utils.MathematicalUtilities.ScaleDesignMatrix(DesignMatrix)
    DesignMatrix = np.transpose(DesignMatrix)       # transpose for indexing:

    for i in range(len(DesignMatrix)):
        plot_utils.PlotFeatures(DesignMatrix[i],classes=TargetVector,labels=[" "," "])
    
