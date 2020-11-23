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

    # Establish local Directory Paths
    parent = "C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\SignalClassifier-CrossValidation"
    modelName = "XValGammaCLF"
    dataPath = os.path.join(parent,modelName+"-Output-Data")
    exptPath = os.path.join(parent,modelName+"-Analysis")    
    n_classes = 38

    # Run Main program
    Program = utils.AnalyzeModels(modelName,dataPath,n_classes)
    encode,decode = Program.MakeDecodeDictionary()

    Program.__Call__(exptPath)
    Program.ExportMetrics(exptPath)


    # Compute & Plots Avg. metrics per class
    _ = np.arange(10)
    ComputeMetrics = utils.ClassifierMetrics(n_classes,_,_,_)
    scores = ComputeMetrics.MetricScores(Program.ConfusionMatrix)
    os.chdir("C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\Thesis\\FiguresMetrics")
    for i in range(n_classes):         # each class
        className = decode[i]
        print(className,Program.counts[i])
        utils.ClassifierMetrics.PlotMetricsForClass(scores[:,i],className,False,True)
    
    print("=)")
