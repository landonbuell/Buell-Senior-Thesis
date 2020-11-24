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
    parentPath = "C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis"
    mtrxPath = os.path.join(parentPath,"FeatureData\\Matrix1.csv")
    exptPath = os.path.join(parentPath,"Thesis\\FiguresFeatures")
    encdPath = os.path.join(parentPath,"SignalClassifier-CrossValidation\\XValGammaCLF-Output-Data")

    # Get Raw Datas
    inputFrame = pd.read_csv(mtrxPath,header=0).to_numpy()
    X = inputFrame[1:,2:].astype(np.float64)
    y = inputFrame[1:,1].astype(np.int16)

    # Organize Data
    n_classes,n_features = 37,20
    Processor = utils.FeatureProcessor(X,y,n_classes,n_features)
    Processor.CreateDictionary(encdPath)
    Processor.__Call__(exptPath)

    print("=)")

    
