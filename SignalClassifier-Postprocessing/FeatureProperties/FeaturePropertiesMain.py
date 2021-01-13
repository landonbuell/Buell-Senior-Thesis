"""
Landon Buell
Kevin Short
PHYS 799
26 october 2020
"""

            #### IMPORTS ####

import numpy as np
import pandas as pd
import os

import PropertyUtilities as utils

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    # Initialize Directories
    parentPath = "C:\\Users\\lando\\Documents\\GitHub\\Buell-Senior-Thesis"
    mtrxPath = os.path.join(parentPath,"FeatureData\\Matrix2.csv")
    smplPath = os.path.join(parentPath,"FeatureData\\Matrix3.csv")
    exptPath = os.path.join(parentPath,"JMMPaper\\ClassifierTest")
    encdPath = os.path.join(parentPath,"SignalClassifier\\Model-Data")

    # Get Raw Datas
    inputFrame = pd.read_csv(mtrxPath,header=0).to_numpy()  # data from full array
    X = inputFrame[:,2:].astype(np.float64)                 # design matrix
    y = inputFrame[:,1].astype(np.int16)                    # labels

    # Get Raw Data From Unknown sample
    testFrame = pd.read_csv(smplPath,header=0).to_numpy()   # data from unknown sample
    X1 = testFrame[:,2:].astype(np.float64)                 # design matrix
    y1 = testFrame[:,1].astype(np.int16)                    # labels

    # Organize Data
    n_classes,n_features = 37,24
    Processor = utils.FeatureProcessor(X,y,n_classes,n_features)
    Processor.CreateDictionary(encdPath)
    Processor.AddUnknownSample(X1)
    Processor.__Call__(exptPath)

    print("=)")

    
