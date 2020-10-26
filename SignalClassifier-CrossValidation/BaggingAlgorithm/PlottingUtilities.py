"""
Landon Buell
Kevin Short
PHYS 799
18 October 2020
"""

        #### IMPORTS ####

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

        #### Class Definitions ####

class TimePlotting:
    """ Class To Plot time-space data 
    Class of all static members """

    def PlotLoss(loss,title="",save=False,show=True):
        """ Plot Loss Score(s) at each Epoch """
        plt.figure(figsize=(16,12))
        plt.title(title,size=40,weight='bold')
        plt.xlabel("Epoch Number",size=20,weight='bold')
        plt.xlabel("Loss Score",size=20,weight='bold')

        for i,score in enumerate(loss):         # each array:
            _epochs = np.arange(len(score))     # number of epochs
            plt.plot(_epochs,score,label=str(i))

        plt.grid()
        plt.tight_layout()
        plt.legend()

        if save == True:
            plt.savefig(title+".png")
        if show == True:
            plt.show()
        plt.close()



