"""
Landon Buell
Classifier Preprocessing Module
PHYS 799
16 August 2020
"""

        #### IMPORTS ####

import os
import PreprocessUtilities as utils

        #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    # ESTABLISH DIRECTORIES
    homePath = os.getcwd()
    exptPath = "C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\Instrument_Classifier_v0\\Target-Data"
    dataPath = ["C:\\Users\\Landon\\Documents\\audioWAV2","C:\\Users\\Landon\\Documents\\audioWAV"]  
    exptFile = ["Y2.csv","Y3.csv"]

    # Prepare
    Encoder = utils.TargetLabelEncoder()
    FILEOBJS = []

    # Iteate through pairs of files / directories
    for path,file in zip(dataPath,exptFile):
        exportPath = os.path.join(path,file)                    # where are the files going?          
        FILEOBJS += Encoder.ReadLocalPath(path,exportPath)      # get all File objs

    print(len(FILEOBJS))
    cols = ["Fullpath","Target Int","Target String"]
    Data = {"Fullpath":FilePaths,
            "Target Int":TargetIntegers,
            "target Str":TargetStrings }
    Encoder.ConstructDataFrame(Data,exportPath)

    print('=)')