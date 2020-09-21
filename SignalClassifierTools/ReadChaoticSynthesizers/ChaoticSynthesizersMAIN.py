"""
Landon Buell
Kevin Short
PHYS 799
19 September 2020
"""

        #### IMPORTS ####

import numpy as np
import os
import sys
import ChaoticSynthesizersUtilities as utils

        #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    # Process command line args
    try:
        readPath = sys.argv[1]
    except:
        readPath = "C:\\Users\\Landon\\Documents\\audioChaoticSynthesizerTXT"

    Startup = utils.ProgramInitalizer(readPath)
    for file in Startup.csvFiles:       # each csv
        file.ReadData()

    print("=)")