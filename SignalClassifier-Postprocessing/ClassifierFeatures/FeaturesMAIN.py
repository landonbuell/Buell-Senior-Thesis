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

"""
    Not Implemented!

"""

            #### MAIN EXECTUABLE ####

if __name__ == '__main__':

    dataPath = "C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\SignalClassifier\\Target-Data"
    exptPath = "C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\Thesis\\Figures"
    
    ProgramSetup = sys_utils.ProgramInitializer([dataPath,exptPath])
    groupedFiles = ProgramSetup.__Call__()
    Decoder = ProgramSetup.GetDecoder
    nClasses = ProgramSetup.n_classes

    n_features = 20

    for i in range(nClasses):       # each class:
        DesignMatrix = struct_utils.FeatureContainer(i,Decoder[i],groupedFiles[i])
        DesignMatrix.__Call__()
    
