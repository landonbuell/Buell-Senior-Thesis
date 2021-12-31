"""
Repository:     Buell-Senior-Thesis
Solution:       SignalClassifierPrototype
Project:        FeatureEngineering
File:           FeatureSpace.py
 
Author:         Landon Buell
Date:           December 2021
"""

class FeatureSpace2D:
    """
    Parent Class for Feature Analysis 
    """
    pass

class BoxPlots:
    """ Box Plots of Each Class """

    def __init__(self,numClasses):
        """ Constructor for BoxPlots Instance """
        self._numClasses = numClasses
        
    def __del__(self):
        """ Destructor for BoxPlots Instance """
        pass

    