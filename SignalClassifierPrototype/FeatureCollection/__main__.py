"""
Repository:     Buell-Senior-Thesis
Solution:       SignalClassifierPrototype
Project:        FeatureCollection  
File:           __init__.py
 
Author:         Landon Buell
Date:           December 2021
"""

        #### IMPORTS ####

import os
import sys

import __init__
import Administrative

        #### MAIN EXECUTABLE ####

def developmentSettingsInstance():
    """ Return and Instance of FeatureCollection Settings for Development Purposes """
    result = Administrative.AppSettings(
        pathsInput=["..\\lib\\DemoTargetData\\Y4.csv",
                    "..\\lib\\DemoTargetData\\Y3.csv",
                    "..\\lib\\DemoTargetData\\Y2.csv",
                    "..\\lib\\DemoTargetData\\Y1.csv",],
        #pathsInput=["..\\lib\\DemoTargetData\\Y4.csv"],
        pathOutput="..\\..\\..\\..\\audioFeatures\\devTestV1",
        batchSize=64,
        batchLimit=-1,
        shuffleSeed=-1)
    return result

if __name__ == "__main__":

    # Build App Settings + App Instance
    appSettings = developmentSettingsInstance()
    appInstance = Administrative.CollectionApplicationProtoype(appSettings)

    # Construct The Component Managers
    appInstance.startup()

    # Run Feature Collection
    appInstance.execute()

    # Export All Data
    appInstance.shutdown()

    # Exit
    sys.exit(0)
