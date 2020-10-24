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

    dataPath = "C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\SignalClassifier\\Target-Data"
    exptPath = "C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\Thesis\\Figures"
    
    ProgramSetup = sys_utils.ProgramInitializer([dataPath,exptPath])
    groupedFiles = ProgramSetup.__Call__()
    Decoder = ProgramSetup.GetDecoder
    nClasses = ProgramSetup.n_classes

    n_features = 20

    for i in range(nClasses-1,0,-1):       # each class:
        print("Class Int: "+str(i)+"\tClass Str:"+str(Decoder[i]))
        DesignMatrix = struct_utils.FeatureContainer(i,Decoder[i],groupedFiles[i])
        DesignMatrix.__Call__()
        print(np.var(DesignMatrix.X,axis=0))
        #[plot_utils.PlotHistogram(DesignMatrix.X[:,j],50," ") for j in range(n_features)]

        print("=)")
    print("=)")
    
