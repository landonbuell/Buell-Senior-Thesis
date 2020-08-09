"""
Landon Buell
PHYS 799
Instrument Classifier v0
10 June 2020
"""

            #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt

"""
Plotting_Utilities.py - "Plotting Utilities"
    Contains Definitions to visualize spectral data in
    time-space, frequency-space, and feature-space
"""

            #### PLOTTING FUNCTIONS ####

def Plot_Confusion (X,labels,title,save=False,show=True):
    """
    Plot 2D Confusion Matrix
    --------------------------------
    X (arr) : (N x N) array of data representing confusion matrix
    labels (arr):  (1 x N) array of data to represent labels grid
    title (str) : Title to give figure
    save (bool) : If true, save MPL figure to cwd (False by Default)
    show (bool) : If true, shows current figure to User (True by Default)
    --------------------------------
    return None
    """
    plt.imshow(X,cmap=plt.cm.binary)
    plt.xlabel("Actual Class",fontsize=20,fontweight='bold')
    plt.ylabel("Predicted Class",fontsize=20,fontweight='bold')
    plt.xticks(ticks=labels,fontsize=12)
    plt.yticks(ticks=labels,fontsize=12)

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
    plt.figure(figsize=(16,12))
    plt.title(title,size=40,weight='bold')
    plt.xlabel(str(labels[0]),size=20,weight='bold')
    plt.ylabel(str(labels[1]),size=20,weight='bold')

    X1 /= np.max(X1)
    #X2 /= np.max(X2)

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

def Plot_History (hist,model,save=False,show=True):
    """
    Visualize Data from Keras History Object Instance
    --------------------------------
    hist (inst) : Keras history object
    model (inst) : Keras Sequential model w/ name attrb
    save (bool) : If true, save MPL figure to cwd (False by Default)
    show (bool) : If true, shows current figure to User (True by Default)
    --------------------------------
    Return None
    """
    # Initialize Figure
    eps = np.array(hist.epoch)          # arr of epochs
    n_figs = len(hist.history.keys())   # needed figures

    fig,axs = plt.subplots(nrows=n_figs,ncols=1,sharex=True,figsize=(20,8))
    plt.suptitle(model.name+' History',size=50,weight='bold')
    hist_dict = hist.history
    
    for I in range (n_figs):                # over each parameter
        key = list(hist_dict)[I]
        axs[I].set_ylabel(str(key).upper(),size=20,weight='bold')
        axs[I].plot(eps,hist_dict[key])     # plot key
        axs[I].grid()                       # add grid

    plt.xlabel("Epochs",size=20,weight='bold')

    if save == True:
        plt.savefig(title.replace(' ','_')+'.png')
    if show == True:
        plt.show()

def Plot_Phase_Space (S,dS,title='',
                      save=False,show=True):
    """
    Create 2D visualization Comparing features
    --------------------------------
    S (arr) : (1 x N) array of waveform or time-frame
    dS (arr) : (1 x N) array of derivative of 'S'    
    title (str) : Title for plot
    save (bool) : If true, save MPL figure to cwd (False by Default)
    show (bool) : If true, shows current figure to User (True by Default)
    --------------------------------
    return None
    """
    plt.figure(figsize=(16,12))
    plt.title(title,size=40,weight='bold')
    plt.xlabel('Signal, $S$',size=30,weight='bold')
    plt.ylabel('Derivative $\\frac{dS}{dt}$',size=30,weight='bold')

    for X in [S,dS]:        # Center
        X -= np.mean(X)     # Subtract mean

    plt.plot(S,dS,color='blue')

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
    plt.figure(figsize=(16,12))
    plt.title(title,size=40,weight='bold')
    plt.xlabel('Time',size=30,weight='bold')
    plt.ylabel('Frequnecy',size=30,weight='bold')

    try:        # plot numpy array
        plt.pcolormesh(t,f,Sxx,cmap=plt.cm.binary)
    except:     # sparse matrix
        plt.pcolormesh(t,f,Sxx,cmap=plt.cm.binary)

    plt.grid()
    plt.tight_layout()
    if save == True:
        plt.savefig(title+'.png')
    if show == True:
        plt.show()

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
    plt.figure(figsize=(16,12))
    plt.title(title,size=40,weight='bold')
    plt.xlabel('Index',size=30,weight='bold')
    plt.ylabel('Amplitude',size=30,weight='bold')

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