"""
Landon Buell
PHYS 799
Instrument Classifier v0
10 June 2020
"""

            #### IMPORTS ####

import numpy as np
import os
import sys

import Program_Utilities as prog_utils
import Plotting_Utilities as plot_utils

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':
    
    # ESTABLISH NECESSARY LOCAL PATHS
    home_path = os.getcwd()
    data_path = 'C:/Users/Landon/Documents/wav_audio'
    trgt_path = 'C:/Users/Landon/Documents/'
    itmd_path = 'C:/Users/Landon/Documents/GitHub/Buell-Senior-Thesis/Instrument_Classifer_v0'
    #data_path,trgt_path,itmd_path = prog_utils.Argument_Parser()
    prog_utils.Validate_Directories(data_path,trgt_path,itmd_path)
    