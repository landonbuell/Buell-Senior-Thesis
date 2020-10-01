"""
Landon Buell
PHYS 799
Instrument Classifier v0
17 June 2020
"""

            #### IMPORTS ####
              
import numpy as np
import scipy.stats as stats

"""
MathUtilities.py - "Math Utilities"
    Contains Definitions for general mathematical functions
"""

            #### FUNCTION DEFINITIONS ####

class MathematicalUtilities :
    """
    Mathematical Utilites for feature processing
    --------------------------------
    * no args
    --------------------------------
    All methods are static
    """

    @staticmethod
    def DistributionData (X):
        """
        Analyze properties of an array of FP values
        --------------------------------
        X (arr) : (1 x N) Array of FP numbers to analyze as distribution
        --------------------------------
        Return array of [mean,median,mode,variance]
        """
        mean = np.mean(X,axis=-1)           # avg        
        median = np.median(X,axis=-1)       # median
        var = np.var(X,axis=-1)             # variance
        return np.array([mean,median,var])  # return data

    @staticmethod
    def ReimannSum (X,dx):
        """
        Compute Reimann Sum of 1D array X with sample spacing dx
        --------------------------------
        X (arr) : (1 x N) Array of FP numbers to compute Reimann Sum of
        dx (float) : Spacing between samples, 1 by default
        --------------------------------
        Return Reimann Sum approximation of array
        """
        return np.sum(X)*dx


