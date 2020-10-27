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

    # Get Raw Data
    mtrxPath = "C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\FeatureData\\Matrix.csv"
    exptPath = "C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\FeatureData\\Variances.csv"
    inputFrame = pd.read_csv(mtrxPath,header=0).to_numpy()
    X = inputFrame[1:,2:].astype(np.float64)
    y = inputFrame[1:,1].astype(np.int16)

    # Organize Data
    n_classes,n_features = 33,40
    Processor = utils.FeatureProcessor(X,y,n_classes,n_features)
    Processor.__Call__()


    # Export OutputFrame
    cols = ["FTR"+str(i) for i in range(n_features)]
    outputFrame = pd.DataFrame(outputFrame,columns=cols)
    outputFrame.to_csv(exptPath,index=False)

    print("=)")

    
