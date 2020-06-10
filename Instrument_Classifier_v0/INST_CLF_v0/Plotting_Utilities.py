"""
Landon Buell
PHYS 799
Instrument Classifier v0
10 June 2020
"""

            #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt

            #### PLOTTING FUNCTIONS ####

def Plot_Spectrum (xdata,ydata,labels=[''],title='',
                   save=False,show=True):
    """
    Create 2D visualization Comparing features
    --------------------------------
    xdata (arr) : (1 x N) array of data to plot on x-axis
    ydata (arr) : (M x N) array of data to plot on y-axis ( can be multiple arrays)
    labels (iter) : (1 x M) iterable containing labels for y arrays
    title (str) : Title for plot
    save (bool) : If true, save MPL figure to cwd (False by Default)
    show (bool) : If true, shows current figure to User (True by Default)
    --------------------------------
    return None
    """
    plt.figure(figsize=(20,12))
    plt.title(title,size=40,weight='bold')
    plt.xlabel('Time',size=20,weight='bold')
    plt.ylabel('Amplitude',size=20,weight='bold')

    if ydata.ndim > 1:
        for I,arr in enumerate(ydata):
            plt.plot(xdata,arr,label=str(labels[I]))
        plt.legend()
    else:
        plt.plot(xdata,ydata)

    plt.grid()
    plt.tight_layout()
    if save == True:
        plt.savefig(title+'.png')
    if show == True:
        plt.show()


def Plot_Spectrogram (f,t,Sxx,title='',
                      save=False,show=True):
    """
    Create visualization of soundwave as frequency vs. time vs. power
    --------------------------------
    f (arr) : (1 x N) frequency space axis
    t (arr) : (1 x M) time space axis
    Sxx ((arr) : (N x M) matrix representing file's spectrogram
    title (str) : Title for plot
    save (bool) : If true, save MPL figure to cwd (False by Default)
    show (bool) : If true, shows current figure to User (True by Default)
    --------------------------------
    return None
    """
    plt.figure(figsize=(20,12))
    plt.title(title,size=40,weight='bold')
    plt.xlabel('Time',size=20,weight='bold')
    plt.ylabel('Frequnecy',size=20,weight='bold')

    plt.pcolormesh(t,f,Sxx,cmap=plt.cm.viridis)

    plt.grid()
    plt.tight_layout()
    if save == True:
        plt.savefig(title+'.png')
    if show == True:
        plt.show()

def Plot_Features_2D (X1,X2,classes,labels,title='',
                      save=False,show=True):
    """
    Create 2D visualization Comparing features
    --------------------------------
    X1 (arr) : (1 x N) array of data to plot on x-axis
    X2 (arr) : (1 x N) array of data to plot on y-axis
    classes (arr) : (1 x N) array of labels use to color-code by class
    labels (iter) : (2 x 1) iterable containing labels for x & y axes
    title (str) : Title for plot
    save (bool) : If true, save MPL figure to cwd (False by Default)
    show (bool) : If true, shows current figure to User (True by Default)
    --------------------------------
    return None
    """
    plt.figure(figsize=(20,12))
    plt.title(title,size=40,weight='bold')
    plt.xlabel(str(labels[0]),size=20,weight='bold')
    plt.ylabel(str(labels[1]),size=20,weight='bold')

    for X in [X1/X2]:           # Normalize Each Feature
        X = X/np.max(np.abs(X))

    plt.scatter(X1,X2,c=classes)

    plt.yticks(np.arange(0,1.1,0.1))
    plt.hlines(0,0,1,color='black')
    plt.vlines(0,0,1,color='black')

    plt.grid()
    plt.tight_layout()
    if save == True:
        plt.savefig(title+'.png')
    if show == True:
        plt.show()
