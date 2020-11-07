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

class FileObject:
    """ Hold file data In structure """

    def __init__(self,imptPath,exptPath):
        """ Initialize FileObject Instance """
        self.intpath = imptPath
        self.outpath = exptPath

        self.filename = self.intpath.split("\\")[-1]
        self.targetStr = self.filename.split(".")[0].upper()

    def Category (self,newCategory=None):
        """ Get or Set self.category attribute """
        if newCategory is not None:
            self.targetStr = newCategory
        return self.targetStr

    def SetTargetInt (self,Int):
        """ Set target Integer into Data """
        self.targetInt = Int
        return self

class SampleOrganizer:

    def __init__(self):
        """ Initialize Class Object Instance """
        self.samples = []

    def PermuteSamples (self):
        """ Permute Samples in Array """
        self.samples = np.random.permutation(self.samples)
        return self

    @property
    def GetUniqueCategories (self):
        """ Find Number of Unqiue Categories in Samples """
        unique = np.unique([x.targetStr for x in self.samples])
        print("\nNumber of Unqiue Categories:",len(unique))
        for k in unique:        # each class
            print("\t"+k)       # print name
        return unique

    def ReadLocalPath (self,inpath,outpath,ext='.wav'):
        """ Read through directory and collect all files with ext """
        for roots,dirs,files in os.walk(inpath):    # walk through tree
            for file in files:                      # each file
                if file.endswith(ext):              # .WAV file
                    fullpath = os.path.join(roots,file)
                    self.samples.append(FileObject(fullpath,outpath))   # add to list
        return self                           # return list of files

    def CleanCategories(self):
        """ Remove Redundant Categories """
        for sample in self.samples:     # iterate by each sample
            if sample.targetStr == "FRENCHHORN":
                sample.Category(newCategory="HORN")
            elif sample.targetStr == "DOUBLEBASS":
                sample.Category(newCategory="BASS")
            elif sample.targetStr == "CORANGLAIS":
                sample.Category(newCategory="ENGLISHHORN")
        return self

    def EncodeSamples(self,encoder):
        """ Encode Samples by Int w/ Encode Dict """
        for sample in self.samples:                  # each sample
            targetInt = encoder[sample.targetStr]    # get target int
            sample.SetTargetInt(targetInt)          # set to file instance
        return self

    def WriteOutput(self,outputPaths):
        """ Write Output Files to CSV file """
        cols = ["Fullpath","Target Int","Target Str"]
        for filepath in outputPaths:
            frame = pd.DataFrame(data=None,columns=cols)
            frame.to_csv(filepath,mode="w",index=False)
        for sample in self.samples:
            data = {cols[0]:[sample.intpath],
                    cols[1]:[sample.targetInt],
                    cols[2]:[sample.targetStr]}
            frame = pd.DataFrame(data=data)
            frame.to_csv(sample.outpath,header=False,index=False,mode="a")

        return self
    

class TargetLabelEncoder:
    """ Encode Files By Name and Integer """
    
    def __init__(self,samples=[]):
        """ Initialize TargetLabelEncoder Class Instance """
        self.samples = samples
        self.stringsBowed = ["BASS","CELLO","VIOLA","VIOLIN" ]
        self.stringsPlucked = ["BANJO","GUITAR","MANDOLIN", ]
        self.windsHigh = ["ALTOFLUTE","ALTOSAX","BBCLARINET","CLARINET",
                            "EBCLARINET","ENGLISHHORN","FLUTE","OBOE",
                            "SAXOPHONE","SOPSAX",]
        self.windsLow = ["BASSCLARINET","BASSFLUTE","BASSOON","CONTRABASSOON", ]
        self.brass = ["BASSTROMBONE","HORN","TENORTROMBONE","TROMBONE","TRUMPET","TUBA",]
        self.percussion = ["BELLS","CROTALE","HIHAT" ]
        self.mallets = ["MARIMBA","VIBRAPHONE","XYLOPHONE" ]
        self.synths = ["SawtoothWave","SineWave","SquareWave","TriangleWave"]
        self.noises = ["WhiteNoise"]

    def SetSamples(self,newSamples):
        """ Set new Samples to self """
        self.samples = newSamples
        return self

    @property
    def GetAcceptedClasses (self):
        """ Return List of accepted categories """
        classes = self.stringsBowed + self.stringsPlucked + \
                self.windsHigh + self.windsLow + \
                self.brass + self.percussion + self.mallets + \
                self.synths + self.noises
        print("\nNumber of Categories:",len(classes))
        for k in classes:        # each class
            print("\t"+k)       # print name
        return classes

    def CreateTargetEncoder (self):
        """ Encode each target class with and integer """
        classCounter = 0
        encodingDict = {}
        for category in np.unique([x.targetStr for x in self.samples]):
            if category not in encodingDict:
                val,key = classCounter,category
                encodingDict.update({key:val})
                classCounter += 1
        self.encoder = encodingDict
        return self
        
        

    

    

        
        