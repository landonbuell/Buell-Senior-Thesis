"""
Landon Buell
Kevin Short
PHYS 799
19 September 2020
"""

        #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.fftpack as fftpack
import scipy.io.wavfile as sciowav
import tensorflow as tf
import os
import sys


        #### CLASS OBJECT DECLARATIONS ####

class FileObject :
    """
    Create an object Instance to contain all data from chaotic synthesizer audio
    """

    def __init__(self,filepath):
        """ Initialize FileObject Instance """
        self.path = filepath
        self.filename = self.path.split("\\")[-1]
        self.name = self.filename.split(".")[0]

    def __repr__(self):
        """ String Represenation of this Object """
        return "\nFileObject Located at:\n\t" + self.path

    def ReadData (self):
        """ Read raw txt file data """
        cols = ["X","Y","Z"]
        data = np.loadtxt(self.path,dtype=str,
                comments="%",delimiter=None,skiprows=2,unpack=True)
        self.t = data[0].astype(np.uint16)
        self.Xt = data[1].astype(np.float64)
        self.Yt = data[2].astype(np.float64)
        self.Zt = data[3].astype(np.float64)
        self.npts = len(self.t)
        return self

    def ExtendArray(self,niters):
        """ Extend Each Time Series array with itself """
        for i in range(0,niters):
            self.Xt = np.append(self.Xt,self.Xt)
            self.Yt = np.append(self.Yt,self.Yt)
            self.Zt = np.append(self.Zt,self.Zt)
        self.t = np.arange(len(self.Xt))
        self.npts = len(self.t)
        return self

    def FourierTransform(self):
        """ Transform Wavefrom into Frequeny Domain """
        
        self.f = fftpack.fftfreq(n=self.npts,d=1/44100)
        self.Xf = np.abs(fftpack.fft(self.Xt,n=self.npts,axis=-1))**2
        self.Yf = np.abs(fftpack.fft(self.Yt,n=self.npts,axis=-1))**2
        self.Zf = np.abs(fftpack.fft(self.Zt,n=self.npts,axis=-1))**2

    def PlotTimeSeries (self,save=False,show=True):
        """ Visualize data attribute """
        plt.figure(figsize=(12,8))
        plt.title(self.name,size=40,weight='bold')
        plt.xlabel("Time",size=20,weight='bold')
        plt.ylabel("Amplitude",size=20,weight='bold')

        plt.plot(self.t,self.Xt,color='blue',label='X')
        plt.plot(self.t,self.Yt,color='green',label='Y')
        plt.plot(self.t,self.Zt,color='purple',label='Z')

        plt.legend()
        plt.tight_layout()
        plt.grid()

        if save == True:
            saveName = self.path
            plt.savefig(self.name+".png")
        if show == True:
            plt.show()

    def PlotFreqSeries (self,save=False,show=True):
        """ Visualize data attribute """
        plt.figure(figsize=(12,8))
        plt.title(self.name,size=40,weight='bold')
        plt.xlabel("Frequency",size=20,weight='bold')
        plt.ylabel("Amplitude",size=20,weight='bold')

        plt.plot(self.f,self.Xf,color='blue',label='X')
        plt.plot(self.f,self.Yf,color='green',label='Y')
        plt.plot(self.f,self.Zf,color='purple',label='Z')

        plt.legend()
        plt.tight_layout()
        plt.grid()

        if save == True:
            saveName = self.path
            plt.savefig(self.name+".png")
        if show == True:
            plt.show()

    def WriteWAVFile(self,path):
        """ Write X,Y,Z axes as .wav files to path """
        rate = 44100
        self.Xt = self.Xt.reshape(-1,1)
        self.Yt = self.Yt.reshape(-1,1)
        self.Zt = self.Zt.reshape(-1,1)

        tf.audio.encode_wav(self.Xt,rate,name=os.path.join(path,self.name+"X.wav"))
        tf.audio.encode_wav(self.Yt,rate,name=os.path.join(path,self.name+"Y.wav"))
        tf.audio.encode_wav(self.Zt,rate,name=os.path.join(path,self.name+"Z.wav"))

        return self

class ProgramInitalizer :
    """
    Initialize Program and Preprocess all data 
    """
    def __init__(self,readpath):
        """ Initilize ProgramInitializer Instance """
        self.readpath = readpath
        self.csvFiles = self.CollectFiles()

    def CollectFiles (self,exts='.txt'):
        """ Walk through Local Path and File all files w/ extension """
        csvFiles = []
        for roots,dirs,files in os.walk(self.readpath):  
            for file in files:                  
                if file.endswith(exts):  
                    csvFiles.append(FileObject(os.path.join(roots,file)))
        return csvFiles