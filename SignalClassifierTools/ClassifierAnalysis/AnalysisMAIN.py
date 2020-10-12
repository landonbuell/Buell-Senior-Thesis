"""
Landon Buell
PHYS 799.32
Classifier Analysis Main
28 July 2020
"""

        #### IMPORTS ####

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow.keras as keras

import AnalysisUtilities as utils

        #### MAIN EXECUTABLE ####
        
if __name__ == '__main__':

    parent = 'C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\SignalClassifier'
    export = os.path.join(parent,'Output-Data')

    infiles = ["ChaoticSynthClassifier@PREDICTIONS@2020.10.10.12.45.48.110578.csv"]
    n_classes = 33

    for file in infiles:
        modelName = file.split("@")[0]
        Program = utils.AnalyzeModels(modelName,export,file,n_classes)
        Program.__Call__()

        #utils.ConfusionMatrix.PlotConfusion(n_classes,Program.weightedConfusion)
        #utils.ConfusionMatrix.PlotConfusion(n_classes,Program.standardConfusion)

        utils.ConfusionMatrix.ExportConfusion(Program.weightedConfusion,modelName+" Weighted Confusion",export)
        utils.ConfusionMatrix.ExportConfusion(Program.standardConfusion,modelName+" Standard Confusion",export)

    print("=)")
