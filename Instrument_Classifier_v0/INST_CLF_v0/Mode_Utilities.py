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
from sklearn.model_selection import train_test_split

import System_Utilities as sys_utils
import Feature_Utilities as feat_utils
import Plotting_Utilities as plot_utils
import Machine_Learning_Utilities as ML_utils
import Neural_Network_Models 


"""
Mode_Utilities.py - 'Mode Utilities'
    Contains Definitions that are only called directly from MAIN script
    Functions are large & perform Groups of important operations
"""

            #### FUNCTION DEFINITIONS ####  

class Program_Mode :
    """
    Program Modes Inherit From here
    """
    def __init__(self,FILEOBJS,group_size=256):
        """ Inititialize Class Object Instance """
        self.FILEOBJS = FILEOBJS
        self.n_files = len(self.FILEOBJS)
        self.group_size = group_size

    def loop_counter(self,cntr,max,text):
        """ Print Loop Counter for User """
        print('('+str(cntr)+'/'+str(max)+')',text)
        return None

    def collect_features (self,fileobj):
        """ Collected Features from a Given .WAV file object"""
        fileobj = fileobj.read_audio()          # read raw .wav file

        # Time series feature object
        time_domain_features = feat_utils.Time_Series_Features(fileobj.waveform)
        x1 = []

        freq_domain_features = feat_utils.Frequency_Series_Features(fileobj.waveform)
        x2 = ML_utils.Feature_Array(fileobj.target)
        x2 = x2.set_features(freq_domain_features.Sxx)

        x3 = []

        return x1,x2,x3

    def construct_design_matrices (self,FILES):
        """ Collect Features from a subset File Objects """
        X1 = ML_utils.Design_Matrix(ndim=2)     # Design matrix for MLP
        X2 = ML_utils.Design_Matrix(ndim=4)     # Design matrix for Spectrogram
        X3 = ML_utils.Design_Matrix(ndim=4)     # Design matrix for Phase-Space

        for i,FILEOBJ in enumerate(FILES):
            self.loop_counter(i,self.group_size,FILEOBJ.filename)   # print messege
            x1,x2,x3 = self.collect_features(FILEOBJ)               # collect features

            X1.add_sample(x1)   # Add sample to MLP
            X2.add_sample(x2)   # Add to 
            X3.add_sample(x3)

        X1 = X1.shape_by_sample()
        X2 = X2.pad_2D(newshape=Neural_Network_Models.spectrogram_shape)

        return [X1,X2,X3]

class Train_Mode (Program_Mode):
    """
    Run Program in Train Mode
        Inherits from 'Base_Program_Mode'
    """
    def __init__(self,FILEOBJS):
        """ Instantiate Class Method """
        super().__init__(FILEOBJS)

    def __call__(Neural_Networks):
        """ Call this Instance to Execute Training and Testing """
        Design_matrices = super().construct_design_matrices(FILEOBJS)

class Test_Mode (Program_Mode):
    """
    Run Program in Test Mode
        Inherits from 'Base_Program_Mode'
    """
    def __init__(self, FILEOBJS,labels=False):
        super().__init__(FILEOBJS)
        self.labels = labels        # labels for the trainign set?

    def __call__(Neural_Networks):
        """ Call this Instance to Execute Training and Testing """
        pass

class TrainTest_Mode (Train_Mode,Test_Mode):
    """
    Run Program in Test Mode
        Inherits from 'Base_Program_Mode'
    """
    def __init__(self, FILEOBJS,labels=False):
        super().__init__(FILEOBJS)
        self.labels = labels        # labels for the training set?
        self.testsize = 0.1
       
    def Split_Objs (self):
        """ Split objects into training.testing subsets """
        train,test = train_test_split(self.FILEOBJS,test_size=self.testsize)
        delattr(self,'FILEOBJS')      # delete attribute
        self.TRAIN_FILEOBJS = train
        self.TEST_FILEOBJS = test
        return self

    def __call__(self,Neural_Networks):
        """ Call this Instance to Execute Training and Testing """
        self.Split_Objs()

        Training = Train_Mode(self.TRAIN_FILEOBJS)
        Training.__call__(Neural_Networks)


        return self


        