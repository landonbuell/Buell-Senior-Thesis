"""
Landon Buell
Feature - Extraction
PHYS 799
6 August 2020
"""

        #### IMPORTS ####

import numpy as np
import os 
import pandas as pd

import System_Utilities as sys_utils
import Feature_Utilities as feat_utils
import Plotting_Utilities as plot_utils

        #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    path = 'C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\Instrument_Classifier_v0\\Target-Data'
    ProgramInitializer = sys_utils.ProgramStart(path)
    FILEOBJS,N_classes = ProgramInitializer.__call__()

    #Iterator = sys_utils.FileIterator(FILEOBJS,N_classes)
    #Iterator.__call__()
    #Iterator.ExportData(str(ProgramInitializer.starttime))
    
    Analyzer = sys_utils.DataAnalyzer('2020-08-07_22.59.38.562101.csv',N_classes)

    plot_utils.Plot_Features_2D(Analyzer.frame['COM'].to_numpy(),
                                np.arange(0,Analyzer.n_rows,1)+1,
                                Analyzer.frame['Class'].to_numpy(),
                                ['Center of Mass','Median'],
                                'Feature Space')


    print("=)")
