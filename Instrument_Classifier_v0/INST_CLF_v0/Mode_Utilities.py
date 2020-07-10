"""
Landon Buell
PHYS 799
Instrument Classifier v0
12 June 2020
"""

            #### IMPORTS ####

import numpy as np
import sys
import os

import System_Utilities as sys_utils
import Plotting_Utilities as plot_utils
import Machine_Learning_Utilities as ML_utils
import Neural_Network_Models 

"""
Component_Utilities.py - 'Component Utilities'
    Contains Definitions that are only called directly from MAIN script
    Functions are large & perform Groups of important operations
"""

            #### FUNCTION DEFINITIONS ####  

class Program_Mode :
    """
    Program Modes Inherit From here
    """
    def __init__(self,FILEOBJS):
        """ Inititialize Class Object Instance """
        self.FILEOBJS = FILEOBJS
        self.n_files = len(self.FILEOBJS)

    def collectfeatures (self,FILES=None):
        """ Collect Features from all given File Objects """

        return [X1,X2,X3]




class Train_Mode (Program_Mode):
    """
    Run Program in Train Mode
        Inherits from 'Base_Program_Mode'
    """
    def __init__(self,FILEOBJS):
        """ Instantiate Class Method """
        super().__init__(FILEOBJS)


    def Split_Objs (self):
        """ Split objects into training.testing subsets """
        pass

    def __call__():
        """ Call this Instance to Execute Training and Testing """


class Test_Mode (Program_Mode):
    """
    Run Program in Test Mode
        Inherits from 'Base_Program_Mode'
    """
    def __init__(self, FILEOBJS,labels=False):
        super().__init__(FILEOBJS)
        self.labels = labels        # labels for the trainign set?
       