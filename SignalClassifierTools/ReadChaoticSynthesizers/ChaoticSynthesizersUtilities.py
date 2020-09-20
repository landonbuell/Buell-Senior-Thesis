"""
Landon Buell
Kevin Short
PHYS 799
19 September 2020
"""

        #### IMPORTS ####

import numpy as np
import os
import sys
import pandas
import matplotlib.pyplot as plt

        #### CLASS OBJECT DECLARATIONS ####

class FileObject :
    """
    Create an object Instance to contain all data from chaotic synthesizer audio
    """

    def __init__(self,filepath):
        """ Initialize FileObject Instance """
        self.path = filepath
        self.name = self.path.split("\\")[-1]

    def __repr__(self):
        """ String Represenation of this Object """
        return "\nFileObject Located at:\n\t" + self.path

    def ReadData (self):
        """ Read raw lines """
        raise NotImplementedError()

    def PlotWaveform (self,save=False,show=True):
        """ Visualize data attribute """
        plt.figure(figsize=(16,12))
        plt.title(self.name,size=40,weight='bold')
        plt.xlabel("Time",size=20,weight='bold')
        plt.ylabel("Amplitude",size=20,weight='bold')

        plt.plot(self.data,color='blue',label='waveform')
        plt.tight_layout()
        plt.grid()

        if save == True:
            saveName = self.path
            plt.savefig(self.name+".png")
        if show == True:
            plt.show()

class ProgramInitalizer :
    """
    Initialize Program and Preprocess all data 
    """
    def __init__(self,readpath):
        """ Initilize ProgramInitializer Instance """
        self.readpath = readpath
        self.csvFiles = self.CollectFiles()

    def CollectFiles (self,exts='.txt'):
        """ Walk through Local Path and File all files w/ extension """
        fileObjs = []
        for roots,dirs,files in os.walk(self.readpath):  
            for file in files:                  
                if file.endswith(exts):  
                    fileObjs.append(FileObject(os.path.join(roots,file)))
        return fileObjs