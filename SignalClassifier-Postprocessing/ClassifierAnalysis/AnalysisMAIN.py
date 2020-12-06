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
import AnalysisFigs as Figs

        #### MAIN EXECUTABLE ####
        
if __name__ == '__main__':

    # Establish local Directory Paths
    parent = "C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis"
    modelName = "XValGammaCNN"
    dataPath = os.path.join(parent,"SignalClassifier-CrossValidation",modelName+"-Output-Data")
    metricsPath = os.path.join(parent,"Thesis\\FiguresMetrics")    
    figuresPath = os.path.join(parent,"Thesis\\FiguresClasses")    
    n_classes = 37

    # Run Main program
    Program = utils.AnalyzeModels(modelName,dataPath,n_classes)
    encode,decode = Program.MakeDecodeDictionary()

    Program.__Call__(metricsPath)
    #Program.ExportMetricsBySplit(metricsPath)
    #Program.ExportMetricsByClass(metricsPath)


    # Prepare Figures
    #classIntegerNames = [x for x in range(n_classes)]
    #classStringNames = [decode[x] for x in range(n_classes)]

    #Figures = Figs.MetricFigures(classIntegerNames,classStringNames,Program.metricsArray)
    #Figures.__Call__(figuresPath)

    print("=)")
