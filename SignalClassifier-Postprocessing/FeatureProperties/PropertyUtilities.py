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
        
    @property
    def GetFeatureNames(self):
        """ Get Names of all features as list of strs """
        names = ["Time Domain Env.","Zero X-Rate","Time Center of Mass"]                         
        names += ["Auto Correlation "+str(i+1) for i in range(4)]        
        names += ["MFCC "+str(i+1) for i in range (12)]      
        names += ["Frequency Center of Mass"]                                    
        return names

    def CreateDictionary(self,encdPath):
        """ Handle class dictionary """
        modelName = "XValCLFB"
        categories = CategoryDictionary(encdPath,modelName)
        self.classNames = categories.encoder.keys()          # get names of categories
            
    def __Call__(self,exptPath):
        """ Execute Feature Processor Instance """   
        homePath = os.getcwd()
        xTickNames = ["Full Data"] + [x for x in self.classNames] # make ticks for boxplot
        
        # Make BoxPlot Data for the Full matrix
        for i in range(self.nFeatures):   # Each feature:
           
            featureData = self.X[:,i]            # get full col of matrix
            featureName = self.GetFeatureNames[i]

            self.boxPlotArrays = [BoxPlotData(featureData,"Full Feature")]

            for j in range(self.nClasses):          # each class
                # Get data for this class
                classRows = np.where(self.y == j)   # get rows of class
                classData = featureData[classRows]  # get corresponding Rows
                self.boxPlotArrays.append(BoxPlotData(classData,str(j)))

            # Plot the Data for this class
            os.chdir(exptPath)
            BoxPlotFigure.BoxPlot(self.boxPlotArrays,featureName,xTickNames,
                                  save=True,show=False)
            os.chdir(homePath)

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

class CategoryDictionary :
    """
    Category Dictionary Maps an integer class to string class
        and vice-versa
    """

    def __init__(self,localPath,modelName):
        """ Intitialize CategoryDictionary Instance """
        self.localPath = localPath
        self.modelName = modelName
        self.fileName = modelName+"Categories.csv"
        self.filePath = os.path.join(self.localPath,self.fileName)
        self.encoder,self.decoder = self.LoadCategories()

    def LoadCategories (self):
        """ Load File to Match Int -> Str Class """
        decoder = {}
        encoder = {}
        rawData = pd.read_csv(self.filePath)
        Ints,Strs = rawData.iloc[:,0],rawData.iloc[:,1]
        for Int,Str in zip(Ints,Strs):      # iterate by each
            encoder.update({str(Str):int(Int)})
            decoder.update({int(Int):str(Str)})
        return encoder,decoder

class BoxPlotFigure:
    """
    Contains Plotting methods
    """

    @staticmethod
    def BoxPlot(data,title,xlabels,save=False,show=True):
        """
        Create Box Plot Visualization of Features
        --------------------------------
        data (list) : List of BoxPlotData instances
        title (str) : Title of Plot (Feature name)
        --------------------------------
        Return None
        """
        plt.figure(figsize=(20,12))
        plt.title(title,size=40,weight='bold')
        
        boxPlots = [x.data for x in data]
        plt.boxplot(boxPlots,showfliers=False)


        plt.grid()
        #plt.tight_layout()

        plt.ylabel("Scaled Value ($\mu = 0$, $\sigma^2 = 1$)",
                   size=12,weight='bold')
        plt.xticks(ticks=range(1,len(boxPlots)+1),labels=xlabels,
                   rotation=60,weight='bold')
        
        if save == True:
            plt.savefig(title.replace(" ","_")+".png")
        if show == True:
            plt.show()
        plt.close()
        return None

        