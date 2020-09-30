"""
Landon Buell
Kevin Short
PHYS 799
19 September 2020
"""

        #### IMPORTS ####

import os
import sys
import ChaoticSynthesizersUtilities as utils

        #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    # Process command line args
    try:
        readPath = sys.argv[1]
    except:
        readPath = "C:\\Users\\Landon\\Documents\\audioChaoticSynthesizerWAV\\PER2TO10"
    
    # Establish Output Path
    outputName = readPath.split("\\")[-1]+".csv"
    writePath = "C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\SignalClassifier\\ChaoticSynth-Data"
    writePath = os.path.join(writePath,outputName)
    
    Program = utils.ProgramInitalizer(readPath,writePath)
    Program.__Call__()

    print("=)")