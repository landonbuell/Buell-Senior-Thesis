"""
Landon Buell
Kevin Short
PHYS 799
26 october 2020
"""

            #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

            #### CLASS OBJECT DEFINITIONS ####

class FeatureProcessor:
    """
    Collect and process features by class 
    """

    def __init__(self,X,y,n_classes,n_features):
        """ Initialize Feature Processor Instance """
        self.X =  StandardScaler().fit_transform(X)
        self.y = y 
        self.nClasses = n_classes
        self.nFeatures = n_features
        outputFrame = np.empty(shape=(self.nClasses,self.nFeatures))
        
    def __Call__(self):
        """ Execute Feature Processor Instance """

        # Make BoxPlot Data for the Full matrix
        for i in range(self.nFeatures):   # Each feature:
            featureData = self.X[:,i]           # get full col of matrix

            self.boxPlotArrays = [BoxPlotData(featureData,"Full Feature")]

            for j in range(self.nClasses):          # each class
                # Get data for this class
                classRows = np.where(self.y == j)   # get rows of class
                classData = featureData[classRows]  # get corresponding Rows
                self.boxPlotArrays.append(BoxPlotData(classData,"CLASS"+str(j)))

            # Plot the Data for this class
            BoxPlotFigure.BoxPlot(self.boxPlotArrays,"FTR"+str(i))

        return self

class BoxPlotData :
    """
    Contain Data for Box Plot Information
    """

    def __init__(self,data,label):
        """ Initialize QuantileData Instance """
        self.data = data
        self.label = label

    def MakeBoxPlotData(self):
        """ Compute Data Need for BoxPlot """
        _min,_max = np.min(self.data),np.max(self.data)
        _Qs = np.quantile(self.data,[0.25,0.5,0.75])
        return np.array([_min,Qs,_max]).ravel()

class BoxPlotFigure:
    """
    Contains Plotting methods
    """

    @staticmethod
    def BoxPlot(data,title,save=False,show=True):
        """
        Create Box Plot Visualization of Features
        --------------------------------
        data (list) : List of BoxPlotData instances
        title (str) : Title of Plot (Feature name)
        --------------------------------
        Return None
        """
        plt.figure(figsize=(20,8))
        plt.title(title,size=40,weight='bold')
        
        boxPlots = [x.data for x in data]
        plt.boxplot(boxPlots)

        plt.grid()
        plt.legend()
        plt.tight_layout()
        
        if save == True:
            plt.savefig(title.replace(" ","_")+".png")
        if show == True:
            plt.show()
        plt.close()
        return None

        