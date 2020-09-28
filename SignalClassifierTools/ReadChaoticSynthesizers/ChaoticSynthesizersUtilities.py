"""
Landon Buell
Kevin Short
PHYS 799
19 September 2020
"""

        #### IMPORTS ####

import numpy as np
import pandas as pd
import os
import sys


        #### CLASS OBJECT DECLARATIONS ####


class ProgramInitalizer :
    """
    Initialize Program and Preprocess all data 
    """
    def __init__(self,readpath,outpath):
        """ Initilize ProgramInitializer Instance """
        self.readpath = readpath
        self.writePath = outpath

    def CollectFiles (self,exts='.wav'):
        """ Walk through Local Path and File all files w/ extension """
        self.csvFiles = []
        for roots,dirs,files in os.walk(self.readpath):  
            for file in files:                  
                if file.endswith(exts):  
                    filePath = os.path.join(roots,file)
                    self.csvFiles.append(filePath)
        return self

    def CreateFrame(self):
        """ Intialize OutputFrame """
        cols = ["Full Path"]
        self.frame = pd.DataFrame(data=self.csvFiles,columns=cols)
        self.frame.to_csv(self.writePath,index=False)
        return self

    def __Call__(self):
        """ Call This Program Instance """
        self.CollectFiles()
        self.CreateFrame()
