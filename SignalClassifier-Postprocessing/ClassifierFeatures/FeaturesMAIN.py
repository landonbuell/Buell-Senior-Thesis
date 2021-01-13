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
import StructureUtilities as struct_utils
import PlottingUtilities as plot_utils

            #### MAIN EXECTUABLE ####

if __name__ == '__main__':

    # Initialize Directories
    dataPath = "C:\\Users\\lando\\Documents\\GitHub\\Buell-Senior-Thesis\\JMMPaper\\ClassifierTest"
    exptPath = "C:\\Users\\lando\\Documents\\GitHub\\Buell-Senior-Thesis\\JMMPaper"
    mtrxPath = "C:\\Users\\lando\\Documents\\GitHub\\Buell-Senior-Thesis\\FeatureData\\Matrix3.csv"

    # Preprocessing
    n_features = 24
    ProgramSetup = sys_utils.ProgramInitializer([dataPath,exptPath])
    ProgramSetup.InitOutputMatrix(mtrxPath,n_features)
    groupedFiles = ProgramSetup.__Call__()
    Decoder = ProgramSetup.GetDecoder
    nClasses = ProgramSetup.n_classes

    for i in range(nClasses):           # each class:
        print("Class Int: "+str(i)+"\tClass Str:"+str(Decoder[i]))
        DesignMatrix = struct_utils.FeatureContainer(i,Decoder[i],groupedFiles[i],n_features)
        DesignMatrix.__Call__()
        DesignMatrix.ExportFrame(mtrxPath)
        
        print("=)")
    print("=)")
    
