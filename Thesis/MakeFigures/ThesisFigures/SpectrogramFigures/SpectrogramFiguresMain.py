"""
Landon buell
PHYS 799
Produce Spectorgram Images
12 Nov 2020
"""

import numpy as np
import os
import matplotlib.pyplot as plt

import SystemUtilities as SystUtils
import FeatureUtilities as FeatUtils
import MathUtilities as MathUtils
import PlottingUtilities as PlotUtils

if __name__ == '__main__':

    # List of All Files that We Want to Plot
    exptPath = "C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\Thesis\\Figures"
    dataPath = "C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\SignalClassifier\\Target-Data"
    Initializer = SystUtils.ProgramInitializer(dataPath)
    FILEOBJECTS = Initializer.__Call__()

    # Create Instance of Feature Extractor
    for file in FILEOBJECTS:
        file.ReadFileWAV()             # read raw .wav file
        print("\t"+file.filename)

        FrequencyFeatures = FeatUtils.FrequencySeriesFeatures(file.waveform)
        FrequencyFeatures.__Call__()

        frequencyAxis = FrequencyFeatures.hertz
        timeAxis = FrequencyFeatures.t  
        spectrogram = FrequencyFeatures.spectrogram
       
        os.chdir(exptPath)
        PlotUtils.Plot_Spectrogram(frequencyAxis,timeAxis,spectrogram," ")

