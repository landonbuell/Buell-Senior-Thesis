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
    exptPath = "C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\SignalClassifier\\Target-Data"
    dataPath = ["C:\\Users\\Landon\\Documents\\audioWAV1","C:\\Users\\Landon\\Documents\\audioWAV2",
                "C:\\Users\\Landon\\Documents\\audioNoiseWAV","C:\\Users\\Landon\\Documents\\audioSyntheticWav"]  
    exptFile = ["Y1.csv","Y2.csv","Y3.csv","Y4.csv"]

    # Iteate through pairs of files / directories
    Organizer = utils.SampleOrganizer()
    Encoder = utils.TargetLabelEncoder()
    for path,file in zip(dataPath,exptFile):
        exportPath = os.path.join(exptPath,file)    # where are the files going?          
        Organizer.ReadLocalPath(path,exportPath)    # get all File objs

    # Organize Samples
    Organizer.PermuteSamples()
    Organizer.CleanCategories()
    Organizer.GetUniqueCategories

    # Create & Run Encoder
    Encoder.SetSamples(Organizer.samples)
    Encoder.GetAcceptedClasses
    Encoder.CreateTargetEncoder()
    Organizer.EncodeSamples(Encoder.encoder)

    # Export Encoded Samples
    exportPaths = [os.path.join(exptPath,x) for x in exptFile]
    Organizer.WriteOutput(exportPaths)

    print('=)')
