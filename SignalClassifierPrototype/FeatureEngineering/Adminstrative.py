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

    def __init__(self,pathInput,pathOutput):
        """ Constructor for EngineeringApplicationPrototype Instance """
        self._pathInput     = pathInput
        self._pathOutput    = pathOutput

        self._databaseManager   = DatabaseManager(pathInput)

    def __del__(self):
        """ Destructor for EngineeringApplicationPrototype Instance """
        self._databaseManager = None



class DatabaseManager:
    """ Class To Contain all databases """

    def __init__(self,collectionRunPath):
        """ Constructor for DatabaseManager Instance """
        self._path  = collectionRunPath
        self._batches