"""
Landon Buell
PHYS 799
External Utilities
11 June 2020
"""

        #### IMPORTS ####

import numpy as np
import pandas as pd
import os

import Utilities

        #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    # ESTABLISH DIRECTORIES
    init_path = os.getcwd()
    data_path = 'C:/Users/Landon/Documents/wav_audio'
    expt_path = init_path

    # READ THROUGH FULL DIRECTORY
    FULLPATHS = Utilities.read_directory(data_path,'.wav')  
    FILENAMES = [x.split('/')[-1] for x in FULLPATHS]
    print("Files Found:",len(FILENAMES))

    # SET TARGET
    names = [x.split('.')[0] for x in FILENAMES]
    classes = Utilities.assign_class(names)
    y = Utilities.target_label_encoder(classes)

    # CREATE DATAFRAME
    data = np.array([FULLPATHS,y,classes]).reshape(3,-1) 
    frame = pd.DataFrame(data=data.transpose(),
        columns=['Fullpath','Target Int','Target Str'])
    print(frame.head(-10))

    frame.to_csv('TARGETS.csv')
