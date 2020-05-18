"""
Landon Buell
Instrument Classifier v0
Base Level Utility Functions
10 MAY 2020
"""

            #### IMPORTS ####

import numpy as np
import pandas as pd
import os
import sys
import argparse
import matplotlib.pyplot as plt

"""
INSTRUMENT FEATURES V0 - BASE LEVEL UTILITIES
            Script contains lowest level function and class defintions that 
            supports higher end functions 
        - Plot_Features_2D
"""

            #### PLOTTING FUNCTIONS ####

def Plot_Features_2D (X1,X2,classes,labels,title='',show=True):
    """
    Create 2D visualization Comparing features
    --------------------------------
    X1 (arr) : (1 x N) array of data to plot on x-axis
    X2 (arr) : (1 x N) array of data to plot on y-axis
    classes (arr) : (1 x N) array of labels use to color-code by class
    labels (iter) : (1 x 2) iterable containing labels for x & y axes
    title (str) : Title for plot
    --------------------------------
    return None
    """
    plt.figure(figsize=(16,12))
    plt.title(title,size=40,weight='bold')
    plt.xlabel(str(labels[0]),size=20,weight='bold')
    plt.ylabel(str(labels[1]),size=20,weight='bold')

    plt.scatter(X1,X2,c=classes)

    plt.yticks(np.arange(0,1.1,0.1))
    plt.hlines(0,0,1,color='black')
    plt.vlines(0,0,1,color='black')

    plt.grid()
    plt.tight_layout()
    if show == True:
        plt.show()

def Plot_Confusion_Matrix (model,show=True):
    """
    Visualize Confusion Matrix
    """
    plt.figure(figsize=(16,12))
    plt.title(model.name,size=40,weight='bold')
    plt.xlabel("Actual Classes",size=20,weight='bold')
    plt.ylabel("Predicted Classes",size=20,weight='bold')
    plt.imshow(model.confusion,plt.cm.binary)
    plt.tight_layout()
    plt.show()