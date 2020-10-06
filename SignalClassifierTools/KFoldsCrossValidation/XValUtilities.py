"""
Landon Buell
PHYS 799
X-Validation Utilities
5 October 2020
"""

            #### IMPORTS ####

import numpy as np
import os
import sys

            #### CLASS OBJECT DEFINITIONS ####

class CrossValidationSplit:
    """ Apply Cross-Validation Split to Full data set """
    
    def __init__(self,fileObjs,nSplits=10):
        """ Initialize CrossValidationSplit Instance """
        self.fileObjects = fileObjs
        self.K = nSplits
        self.ApplySplit()

    def ApplySplit(self):
        """ Split Data into 'K' equally sized groups """
        permuted = np.random.permutation(self.fileObjects)  # permute
        self.folds = []                     # list to hold list of file objs
        for i in range(self.K):             # each sample:
            self.folds.append([])           # add an empty 'bin'
        for i in range(len(self.fileObjects)):      # each file
            _bin = i % self.K                       # get the bin for this sample
            self.folds[_bin].append(permuted[i])    # add sample to that bin
        return self         

    def ExportFolds(self,splitNum):
        """ Export the Splits based on the current iteration """
        self.testSubset = self.folds[splitNum]  # build test data
        self.trainSubset = []                   # init train data
        for i in range(self.k):             # of the F-fold
            if (i == splitNum):     # i = split, pass
                continue
            else:
                self.trainSubset += self.folds[i]
        return self

class TrainTestValidator :
    """ Run Model on Train then Predict Mode for a single K """
    pass



            
