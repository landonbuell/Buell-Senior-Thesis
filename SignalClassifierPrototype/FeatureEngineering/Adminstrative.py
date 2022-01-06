"""
Repository:     Buell-Senior-Thesis
Solution:       SignalClassifierPrototype
Project:        FeatureEngineering
File:           Adminstrative.py
 
Author:         Landon Buell
Date:           December 2021
"""

        #### IMPORTS ####

import os
import sys

import numpy

import CommonStructures

        #### CLASS DEFINITIONS ####

class EngineeringApplicationPrototype:
    """
    Class to Hold all functionality of the Feature Engineering Application
    """

    AppInstance = None

    def __init__(self,runPath):
        """ Constructor for EngineeringApplicationPrototype Instance """
        self._runPath = runPath
        self._runInfo = CommonStructures.RunInformation.deserialize(runPath)


    def __del__(self):
        """ Destructor for EngineeringApplicationPrototype Instance """
        pass




