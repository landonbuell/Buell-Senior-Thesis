"""
Landon Buell
Classifier Preprocessing Module
PHYS 799
16 August 2020
"""

        #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

        #### CLASS DECLARATIONS ####

class TargetLabelEncoder:
    """ Encode Files By Name and Integer """

    def __init__(self):
        """ Initialize Class Object Instance """
        self.woodWinds = ['AltoFlute','AltoSax','BbClarinet','EbClarinet',
                    'Flute','Oboe','SopSax','EbClarinet','BassClarinet',
                    'BassFlute','Bassoon']
        self.strings = ['banjo','Bass','Cello','Viola','Violin']
        self.brass = ['BassTrombone','Horn','TenorTrombone','Trumpet','Tuba']
        self.percussion = ['bells','Marimba','Vibraphone','Xylophone']
        self.cymbals = ['crash','chinese','orchcrash','windgong','ride',
                    'hihat','splash','thaigong',]

    @property
    def AcceptedInstruments (self):
        """ Accepted Instrument String Titles """
        return self.woodWinds + self.strings + self.brass + self.percussion + self.cymbals

    def ReadLocalPath (self,path,ext='.wav'):
        """ Read through directory and collect all files with ext """
        FILES = []                              # hold filename
        for roots,dirs,files in os.walk(path):  # walk through tree
            for file in files:                  # each file
                if file.endswith(ext):          # .WAV file
                    fullpath = os.path.join(roots,file)
                    FILES.append(fullpath)      # add to list
        return FILES                            # return list of files

    def AssignTarget (self,filelist):
        """ Assign tagret class value based on name """
        classes = []
        for file in filelist:
            filename = file.split("\\")[-1]     # last element in list
            name = filename.split(".")[0]       # 0-th element in list
            if name in self.AcceptedInstruments:    # in valid instruments
                targetString = name.upper() # set instrument
            elif name in self.percussion:        # percussion?
                targetString = 'PERCUSSION' # set
            elif name in self.cymbals:           # cymbals?
                targetString = 'CYMBAL'     # set
            else:                           # not in lists?
                targetString = 'OTHER'      # set other
            classes.append(targetString)    # add to class list
        return classes                      # list

    def LabelEncoder (self,targetVector):
        """ Create encoding dictiory of strings to classes """
        encodingDictionary = {}                     # output dictionary
        classCounter = 0                            # class counter
        for category in np.unique(targetVector):    # unique elements
            key,val = category,classCounter
            encodingDictionary.update({key:val})    # update the dictionary
            classCounter += 1                       # incriment class counter
        targetInt = [encodingDictionary[x] for x in targetVector]
        # return the encoding/decoding dictionary and number of classes
        return targetInt

    def ConstructDataFrame(self,data,path):
        """ Construct DataFrame for Classifer Targets """    
        frame = pd.DataFrame(data=data)
        frame.to_csv(path_or_buf=path)
        return self

        
        