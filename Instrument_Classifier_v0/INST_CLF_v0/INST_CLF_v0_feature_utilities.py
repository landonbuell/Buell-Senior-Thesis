"""
Landon Buell
Instrument Classifier v0
Feature Extraction
6 April 2020
"""

            #### IMPORTS ####

import numpy as np

import INST_CLF_v0_base_utilities as base_utils
import INST_CLF_v0_time_utilities as time_utils
import INST_CLF_v0_freq_utilities as freq_utils
import INST_CLF_v0_machine_learning_utilities as ML_utils


            #### FUNCTIONS DEFINITIONS ####

def timeseries (wavfile):
    """
    Collect all training features for audio file based on
        Time spectrum data 
    --------------------------------
    wavfile (inst) : Instance of .wav file w/ waveform attribute
    --------------------------------
    Return array of time series features
    """
    features = np.array([])     # array to hold time series features
    
    # rise & decay times
    rise,decay = time_utils.rise_decay_time(wavfile.waveform)
    features = np.append(features,[rise,decay])

    # RMS Energy from frames
    energies,RMS = time_utils.Energy_Frames(wavfile.waveform,512)
    features = np.append(features,RMS)
    
    # RMS below values
    values = time_utils.RMS_Below_Val(energies,RMS,[0.1,0.25,0.5,0.75])
    features = np.append(features,values)

    features = np.ravel(features)   # flatten to 1D
    return features                 # return the feature array

def freqseries (wavfile):
    """
    Collect all training features for audio file based on
        Frequency spectrum data 
    --------------------------------
    wavfile (inst) : Instance of .wav file w/ waveform attribute
    --------------------------------
    Return array of frequency series features
    """
    features = np.array([])
    
    # SPECTROGRAM
    N_pts = 2**10                # pts in FFT
    f,t,Sxx = freq_utils.Spectrogram(wavfile.waveform,N=N_pts)
    #base_utils.Plot_Spectrogram(f,t,Sxx,str(wavfile.filename),show=True)

    # ENERGY / FREQUENCY BAND
    banks = np.array([0,32,64,128,256,512,1024,2048,6000])
    pwr_per_bank = freq_utils.Frequency_Banks(f,t,Sxx,cutoffs=banks)
    features = np.append(features,pwr_per_bank)

    return features