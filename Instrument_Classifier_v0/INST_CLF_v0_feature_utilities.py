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
    
    rise,decay = time_utils.rise_decay_time(wavfile.waveform)
    features = np.append(features,[rise,decay])


    return features             # return the feature array

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
    hann_wave = freq_utils.Hanning_Window(wavfile.waveform)         # apply hann window

    f_space,pts,f_resol = freq_utils.Frequency_Space(1024)          # create freq sp. axis
    power_spect = freq_utils.Power_Spectrum(hann_wave,pts)          # compute power spectrum

    t,f,Sxx = freq_utils.Spectrogram(wavfile.waveform,pts,N=2**10)

    base_utils.Plot_Spectrogram(t,f,Sxx,'Test Spectrogram')

    return features