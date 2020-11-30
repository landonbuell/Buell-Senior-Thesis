"""
Landon Buell
PHYS 799.32
Classifier Analysis Main
28 July 2020
"""

        #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import datetime

import tensorflow as tf
import tensorflow.keras as keras

        #### OBJECT DEFINITIONS ####

class MetricFigures :
    """ Create Figures for Metrics of Each Class """

    def __init__(self,classInts,classStrs,metricsArray):
        """ Contstructor for MakeFigures Instance """
        self.classInts = classInts
        self.classStrs = classStrs
        self.totalMetricsArray = metricsArray                       # shape = (nFiles,nMetrics,n_classes)
        self.fileMetricsArray = np.mean(metricsArray,axis=0)        # shape = (nMetrics,n_classes)
        self.avgMetricsArray = np.mean(self.fileMetricsArray,axis=-1)# shape = (nMetrics)

    def __Call__(self,exportPath):
        """ Call this Class """
        print(self.avgMetricsArray)
        metricsErr = np.array([ np.min(self.totalMetricsArray,axis=0),
                                np.max(self.totalMetricsArray,axis=0)])
        homePath = os.getcwd()
        for (_int,_str) in zip(self.classInts,self.classStrs):          # class
            print("Class:",_str)
            #perClassMetrics = self.totalMetricsArray[:,:,_int]          # all metrics for this class
            #self.BoxPlotMetricsForClass(perClassMetrics,_str,save=False)
            perClassMetrics = self.fileMetricsArray[:,_int]
            os.chdir(exportPath)
            self.BarChartMetricsForClass(perClassMetrics,_str,show=False)
            os.chdir(homePath)
        return self

    def BoxPlotMetricsForClass(self,X,namesave="",show=True,save=True):
        """ Plot Accuracy, Precision, Recall for A Given Class """

        boxprops = dict(linestyle='-',linewidth=4,color='black')
        capprops = dict(linestyle='-',linewidth=3,color='green')
        whiskprops = dict(linestyle='-',linewidth=4,color='black')
        mednprops = dict(linestyle=':',linewidth=3,color='red')
        meanprops = dict(linestyle='--',linewidth=3,color='blue')

        plt.figure(figsize=(8,6))
        plt.boxplot(x=X,widths=0.6,showfliers=False,showmeans=True,meanline=True,
                    boxprops=boxprops,whiskerprops=whiskprops,capprops=capprops,
                    medianprops=mednprops,meanprops=meanprops)

        plt.yticks(np.arange(0,1.1,0.1),size=20,weight='bold')
        plt.xticks(np.arange(1,5),["Accuracy","Precision","Recall","F1"],
                   size=20,weight='bold')

        plt.grid()
        plt.tight_layout()
        if save == True:
            plt.savefig(namesave.replace(" ","_")+".png")
        if show == True:
            plt.show()
        plt.close()
        return None

    def BarChartMetricsForClass(self,X,namesave="",show=True,save=True):
        """ Plot Accuracy, Precision, Recall for A Given Class """
        plt.figure(figsize=(12,6))
        plt.bar(["Accuracy","Precision","Recall","F1"],height=X,
                width=0.5,color=['brown','red','blue','purple'])

        plt.yticks(np.arange(0,1.1,0.1),size=20,weight='bold')
        plt.xticks(size=20,weight='bold')

        plt.grid()
        plt.tight_layout()
        if save == True:
            plt.savefig(namesave.replace(" ","_")+".png")
        if show == True:
            plt.show()
        plt.close()
        return None
