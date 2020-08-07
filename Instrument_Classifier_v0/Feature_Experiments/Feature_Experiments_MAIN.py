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

        #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    path = 'C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\Instrument_Classifier_v0\\Target-Data'
    ProgramInitializer = sys_utils.ProgramStart(path)
    FILEOBJS,N_classes = ProgramInitializer.__call__()

    Iterator = sys_utils.FileIterator(FILEOBJS,N_classes)
    Iterator.__call__()