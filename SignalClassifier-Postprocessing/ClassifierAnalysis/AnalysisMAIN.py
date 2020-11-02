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

    parent = "C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\SignalClassifier-CrossValidation"
    dataPath = os.path.join(parent,"XVal-Output-Data")

    models = ["ChaoticXVal","FeaturesXVal"]
    n_classes = 33


    infiles = ["ChaoticXVal1@PREDICTIONS@2020.10.19.22.03.30.032221.csv",
               "FeaturesXVal1@PREDICTIONS@2020.11.01.01.31.42.020625.csv"]

    infiles = [file for file in os.listdir(dataPath) if ("@PREDICTIONS@" in file)]

    export = os.path.join(parent,"Confusions")

    for file in infiles:
        modelName = file.split("@")[0]
        Program = utils.AnalyzeModels(modelName,dataPath,file,n_classes)
        Program.__Call__()

        """
        os.chdir(export)
        utils.ClassifierMetrics.ExportConfusion(Program.weightedConfusion,modelName+" Weighted Confusion",export)
        utils.ClassifierMetrics.ExportConfusion(Program.standardConfusion,modelName+" Standard Confusion",export)

        utils.ClassifierMetrics.PlotConfusion(Program.weightedConfusion,33,modelName+" Weighted Confusion")
        utils.ClassifierMetrics.PlotConfusion(Program.standardConfusion,33,modelName+" Standard Confusion")
        """

    print("=)")
