"""
Repository:     Buell-Senior-Thesis
Solution:       SignalClassifierPrototype
Project:        FeatureCollection  
File:           CollectionMethods.py
 
Author:         Landon Buell
Date:           December 2021
"""

            #### IMPORTS ####

import os
import sys
import numpy as np

            #### CLASS DEFINIIONS ####

class CollectionMethod:
    """
    Abstract Base Class for All Collection Methods to Be Queued
    """

    def __init__(self,name,param):
        """ Constructor for CollectionMethod Base Class """
        self._methodName    = name
        self._parameter     = param

    def __del__(self):
        """ Destructor for CollectionMethod Base Class """
        pass

    # Public Interface

    def invoke(self,signal,*args):
        """ Run this Collection method """
        return np.array([0.0])
