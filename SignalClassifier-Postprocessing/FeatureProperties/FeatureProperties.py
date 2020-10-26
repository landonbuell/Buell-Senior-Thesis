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
from sklearn.preprocessing import StandardScaler

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    # Get Raw Data
    mtrxPath = "C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\FeatureData\\Matrix.csv"
    exptPath = "C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\FeatureData\\Variances.csv"
    inputFrame = pd.read_csv(mtrxPath,header=0).to_numpy()
    X = inputFrame[1:,2:].astype(np.float64)
    y = inputFrame[1:,1].astype(np.int16)

    n_classes,n_features = 33,20

    # Scale Design Matrix and prepare output
    outputFrame = np.empty(shape=(n_classes,n_features))
    X = StandardScaler().fit_transform(X) 
    print(np.var(X,axis=0))     # variance down columns
    print(np.mean(X,axis=0))    # variance down columns

    # Each col now has unit variance. 
    # Get variance by each class
    for i in range(n_classes):          # each class
        classRows = np.where(y==i)      # where y has those classes
        classData = X[classRows]        # each row in idx
        classVar = np.var(classData,axis=0)    # variance down columns
        outputFrame[i] = classVar       # update OutputFrame

    # Export OutputFrame
    cols = ["FTR"+str(i) for i in range(n_features)]
    outputFrame = pd.DataFrame(outputFrame,columns=cols)
    outputFrame.to_csv(exptPath,index=False)

    print("=)")

    
