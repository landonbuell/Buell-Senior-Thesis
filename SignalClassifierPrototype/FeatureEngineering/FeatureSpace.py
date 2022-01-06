"""
Repository:     Buell-Senior-Thesis
Solution:       SignalClassifierPrototype
Project:        FeatureEngineering
File:           FeatureSpace.py
 
Author:         Landon Buell
Date:           December 2021
"""

class FeatureSpace2D:
    """
    Parent Class for Feature Analysis 
    """
    pass

class BoxPlots:
    """ Box Plots of Each Class """

    def __init__(self,numClasses):
        """ Constructor for BoxPlots Instance """
        self._numClasses = numClasses
        
    def __del__(self):
        """ Destructor for BoxPlots Instance """
        pass

class SpectrogramVisualizer:
    """
    Class to Set Params + Visualize a Spectrogram
    """

    def __init__(self,colorMap='jet',save=False,show=True):
        """ Constructor for SpectrogramVisualizer Instance """
        self._colorMap = colorMap
        self._saveFigure = save
        self._showFigure = show
        self._data = None

    def __del__(self):
        """ Destruction for SpectrogramVisualizer Instance """
        self._data = None

    def generatePlot(self,spectrogram):
        """ Generate Plot of spectrogram """
        return None
