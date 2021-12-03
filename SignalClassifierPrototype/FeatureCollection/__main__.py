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

if __name__ == "__main__":

    # Build App Settings + App Instance
    appSettings = Administrative.AppSettings.developmentSettingsInstance()
    appInstance = Administrative.CollectionApplicationProtoype(appSettings)

    # Construct The Component Managers
    appInstance.buildManagers()


    # Exit
    sys.exit(0)
