"""
Repository:     Buell-Senior-Thesis
Solution:       SignalClassifierPrototype
Project:        FeatureEngineering
File:           Preprocessing.py
 
Author:         Landon Buell
Date:           January 2022
"""

        #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler

EPSILON = np.array([1e-6],dtype=np.float32)

import CommonStructures

        #### CLASS DEFINTIONS ####

class PreprocessingTool:
    """ Abstract Parent Class For all Preprocessing Tools """

    def __init__(self,toolName):
        """ Constructor for PreprocessingTool Abstract Class """
        self._matrix = None
        self._toolName = toolName
        self._timesFit = 0
        
    def __del__(self):
        """ Destruction for PreprocessingTool Abstract Class """
        self._matrix = None

    # Getters and Setters

    def getToolName(self):
        """ Return the name of this tool """
        return self._toolName

    def getTimesFit(self):
        """ Get the Number of times that the tool has been fit """
        return self._timesFit

    def getIsFit(self):
        """ Return T/F if the model has been fit at all """
        return (self._timesFit > 0)

    def getSampleShape(self):
        """ Get the Shape of Each Sample """
        if (self._matrix is not None):
            return self._matrix.getSampleShape()
        else:
            return []

    def setMatrix(self,designMatrix):
        """ Set the Current Design Matrix to Operate On """
        self.checkShape(designMatrix)
        self._matrix = designMatrix
        return self

    # Public Interfacce

    def fit(self,designMatrix):
        """ Fit the Preprocessing Tool w/ a design matrix Object """
        self.setMatrix(designMatrix)
        return self

    def transform(self,designMatrix):
        """ Transform a design matrix with the fir params """
        self.setMatrix(designMatrix)
        return designMatrix

    def reset(self):
        """ Reset the state of the tool """
        self._matrix = None
        self._timesFit = 0
        return self

    # Protected Interface

    def checkShape(self,designMatrix):
        """ Check that the Design Matrix has the correct shape """
        if (self._timesFit == 0):
            # No Sample Shape Yet, just assign Matrix?
            return True

        if (designMatrix.getSampleShape() != self.getSampleShape()):
            # Shape mismatch
            errMsg = "The sample shape of the design matrix does not match that of the fit tool"
            raise RuntimeError(errMsg)
        return True


    # Magic Methods

    def __repr__(self):
        """ Return Debugger Representation of Instance """
        return str(self.__class__) + " @ " + str(hex(id(self)))

class SelectKBest(PreprocessingTool):
    """ Select K Features that score best by metric """
    
    def __init__(self,scoringCallback):
        """ Constructor for SelectKBest """
        super().__init__("SelectKBest")
        self._scoringCallback = scoringCallback

    def __del__(self):
        """ Destructor for FeatureScaler Instance """
        pass


class FeatureScaler(PreprocessingTool):
    """ Scales each Feature of a Design Matrix to Have Zero Mean and Unit Variance """

    def __init__(self):
        """ Constructor for FeatureScaler Instance """
        super().__init__("FeatureScaler")
        self._scaler = StandardScaler()


    def __del__(self):
        """ Destructor for FeatureScaler Instance """
        self._scaler = None
        
        pass

    # Getters and Setters

    

    # Public Interface

    def fit(self,designMatrix):
        """ Fit the Preprocessing Tool w/ a design matrix Object """
        self._scaler.fit(
            designMatrix.getFetures())
        self._timesFit += 1
        return self

    def transform(self,designMatrix):
        """ Transform a design matrix with the fir params """
        if (self.getIsFit() == False):
            # Tool not yet Fit
            raise RuntimeError(repr(self) + " is not yet fit")
        newFeatures = self._scaler.transform(
            designMatrix.getFeatures())
        designMatrix.setFeatures(newFeatures)
        return designMatrix

    # Private Interface

    def computeBatchMeans(self,X):
        """ Compute the Average of Each Feature in the given batch """
        return np.mean(X.getFeatures(),axis=0,dtype=np.float32)
        
    def computeBatchVaris(self,X):
        """ Compute the Variance of Each Feature in the given batch """
        return np.var(X.getFeatures(),axis=0,dtype=np.float32)
   
    def storeBatchMeans(self,x):
        """ Store the average of each feature in this batch """
        if (self._means.shape[0] == 0):
            self._sampleShape = list(x.shape)
        if (x.shape != self._sampleShape):
            raise ValueError("Shape Mismatch")
        newShape = [self._means.shape[0]] + self._sampleShape
        self._means = np.append(self._means,x)
        self._means.reshape(newShape)
        return self

    def storeBatchVaris(self,x):
        """ Store the variances of each feature in this batch """
        if (self._varis.shape[0] == 0):
            self._sampleShape = list(x.shape)
        if (x.shape != self._sampleShape):
            raise ValueError("Shape Mismatch")
        newShape = [self._varis.shape[0]] + self._sampleShape
        self._varis = np.append(self._varis,x)
        self._varis.reshape(newShape)
        return self

class MinMaxVarianceSelector(PreprocessingTool):
    """
    Select Features Based in Minimum Intra-Class Variance and Maximum Extra-Class Variance 
    """

    def __init__(self,featureNames,classNames,
                 featureMask=None,classMask=None,
                 save=False,show=True):
        """ Constructor for SelectKBest """
        super().__init__("MinMaxVarianceSelector")
        self._featureNames = featureNames
        self._classNames = classNames

        self._featureMask = None
        self._classMask = None
        
        self._save = save
        self._show = show

    def __del__(self):
        """ Destructor for FeatureScaler Instance """
        pass

    # Getters and Setters

    def getFeatureNames(self):
        """ Return List of Feature Names """
        if (self._featureNames is not None):
            if (self._featureMask is not None):
                # Need a double list comprehension here?
                result = [x for (x,y) in zip(self._featureNames,self._featureMask) if y == True]
                return result
            else:
                return self._featureNames
        else:
            return []

    def getClassNames(self):
        """ Return List of Feature Names """
        if (self._classNames is not None):
            if (self._classMask is not None):
                # Need a double list comprehension here?
                result = [x for (x,y) in zip(self._classNames,self._classMask) if y == True]
                return result
            else:
                return self._classNames
        else:
            return []

    def setFeatureMask(self,mask):
        """ Set mask of features to Use for processing """
        if (mask is None):
            self._featureMask = np.ones(
                shape=self._matrix.getSampleShape(),
                dtype=np.bool8)
        elif (mask.shape != self._matrix.getSampleShape() ):
            raise ValueError("Shape mismatch for feature mask array")
        else:
            self._featureMask = mask
        return self

    def setClassMask(self,mask):
        """ Set mask of classes to use for processing """
        if (mask is None):
            self._classMask = np.ones(
                shape=self._matrix.getUniqueClasses().shape,
                dtype=np.bool8)
        elif (mask.shape != self._matrix.getUniqueClasses() ):
            raise ValueError("Shape mismatch for class mask array")
        else:
            self.classMask = mask
        return self

    # Public Interface

    def fit(self,designMatrix,featureMask=None,classMask=None):
        """ Fit the Tool Given the Design Matrix """
        super().fit(designMatrix)
        self.setFeatureMask(featureMask)
        self.setClassMask(classMask)
        self.buildVarianceMatrix()

        # Compute The Variance for Each Feature
        self.invoke()

        if (self._save is not None or 
            self._show == True):
            # Plot and save and/or show the figure
            self.plotVarianceMatrix(title="LogVarianceMatrix")

        self.printMatrix()

        self._matrix = None
        return self._varianceMatrix

    def transform(self, designMatrix):
        """ Transform the Design Matrix w/ the Tool's Parameters """
        super().transform(designMatrix)

        return self

    def reset(self):
        """ Reset This Instance to Construction State """
        super().reset()
        self._featureMask = None
        self._classMask = None
        self._featureNames = None
        self._classNames = None
        self._save = False
        self._show = True
        return self

    # Private Interface

    def buildVarianceMatrix(self):
        """ Construct + Assign the Variance Matrix """
        if (self._matrix is None):
            raise ValueError("Must provide Design Matrix to build Variance Matrix")

        numClasses = np.sum(self._classMask)
        numFeatures = np.sum(self._featureMask)
         
        # Variance Matrix has shape (num classes x num Features)       
        varShape = [numClasses, numFeatures] # Only 2D Arrays right now?
        self._varianceMatrix = np.zeros(shape=varShape,dtype=np.float32)
        return self

    def invoke(self):
        """ Proccess All Features in One class """
        
        rowIndex = 0
        unqiueClasses = self._matrix.getUniqueClasses()
        overallVariances = self._matrix.varianceOfFeatures(self._featureMask)
        for classIndex,useClass in zip(unqiueClasses,self._classMask):
            # Skip Classes that we don't case about
            if (useClass == False):
                continue

            # Isolate all samples of class + Compute Variance
            subsetMatrix = self._matrix.samplesInClass(classIndex)
            variances = subsetMatrix.varianceOfFeatures(self._featureMask)
            
            # Copy Variances to Output Matrix and scale
            np.copyto(self._varianceMatrix[rowIndex],variances)
            self._varianceMatrix[rowIndex] /= overallVariances
            rowIndex += 1

        return self

    def plotVarianceMatrix(self,title,log=True):
        """ Create a Colormap plot of the Variance Matrix """
        plt.figure(figsize=(16,8),edgecolor='gray')
        plt.title(title,size=24,weight='bold')
        plt.xlabel("Feature Index",size=16,weight='bold')
        plt.ylabel("Class Index",size=16,weight='bold')
        
        normalizedMatrix = np.log(self._varianceMatrix + EPSILON)
        plt.matshow(normalizedMatrix,fignum=0)

        featureNames = self.getFeatureNames()
        classNames = self.getClassNames()
        plt.xticks(ticks=np.arange(len(featureNames)),
                   labels=featureNames,
                   rotation=90)
        plt.yticks(ticks=np.arange(len(classNames)),
                   labels=classNames)

        plt.grid()

        if (self._save == True):
            plt.savefig(title + ".png")
        if (self._show == True):
            plt.show()
        plt.close()
        return None

    def printMatrix(self):
        """ Print the Variance Matrix to the Console """
        for row in self._varianceMatrix:
            print(row)
        return self

    def exportVarianceMatrix(self):
        """ Export the Variance Matrix """
        frame = pd.DataFrame()

        return self



    # Magic Methods

