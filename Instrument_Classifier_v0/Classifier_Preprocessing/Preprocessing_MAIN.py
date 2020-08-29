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
import Preprocessing_Utils as utils

        #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    # ESTABLISH DIRECTORIES
    homePath = os.getcwd()
    dataPath = "C:\\Users\\Landon\\Documents\\audioWAV"
    exptPath = "C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\Instrument_Classifier_v0\\Target-Data"
    exptFile = "Y1.csv"
    exportPath = os.path.join(exptPath,exptFile)

    Encoder = utils.TargetLabelEncoder()
    FilePaths = Encoder.ReadLocalPath(dataPath)
    TargetStrings = Encoder.AssignTarget(FilePaths)
    TargetIntegers = Encoder.LabelEncoder(TargetStrings)

    cols = ["Fullpath","Target Int","Target String"]
    Data = {"Fullpath":FilePaths,
            "Target Int":TargetIntegers,
            "target Str":TargetStrings }
    Encoder.ConstructDataFrame(Data,exportPath)

    print('=)')