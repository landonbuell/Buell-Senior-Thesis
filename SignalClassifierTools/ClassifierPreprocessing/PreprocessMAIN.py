"""
Landon Buell
Classifier Preprocessing Module
PHYS 799
16 August 2020
"""

        #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import os
import PreprocessUtilities as utils

        #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    # ESTABLISH DIRECTORIES
    homePath = os.getcwd()
    exptPath = "C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\Instrument_Classifier_v0\\Target-Data"
    dataPath = ["C:\\Users\\Landon\\Documents\\audioWAV2"]  
    exptFile = ["Y2.csv"]

    # Iteate through pairs of files / directories
    for path,file in zip(dataPath,exptFile):
        exportPath = os.path.join(path,file)

        Encoder = utils.TargetLabelEncoder()
        FilePaths = Encoder.ReadLocalPath(path)
        TargetStrings = Encoder.AssignTarget(FilePaths)
        TargetIntegers = Encoder.LabelEncoder(TargetStrings)

        cols = ["Fullpath","Target Int","Target String"]
        Data = {"Fullpath":FilePaths,
                "Target Int":TargetIntegers,
                "target Str":TargetStrings }
        Encoder.ConstructDataFrame(Data,exportPath)

    print('=)')