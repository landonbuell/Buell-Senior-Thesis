"""
Landon Buell
Instrument Classifier v1
Main Function
3 February 2020
"""

            #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

import Instrument_CLF_v1_func as func
import Instrument_CLF_v1_features as features
import Instrument_CLF_v1_timeseries as timeseries
import Instrument_CLF_v1_freqseries as freqseries
import Instrument_CLF_v1_MLfunc as MLfunc

"""
INSTRUMENT CLASSIFIER V1 - MAIN EXECUTABLE

Directory paths:
    - 'int_dir' is the initial directory for this program
            where it is saved locally on the HDD or SSD
            also use "int_dir = os.getcwd()" as well
    - 'wav_dir' is the directory path where all raw .wav audio files are stored
    - 'out_dir' is a misc directory use to dump temporary files (created by program if nonexisitant)
"""

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    # These paths for are Landon's Computer 
    # see documentation above to set for you particular machine
    int_dir = 'C:/Users/Landon/Documents/GitHub/Buell-Senior-Thesis/Instrument_Classifier_v1'
    wav_dir = 'C:/Users/Landon/Documents/wav_audio'
    out_dir = int_dir + '/wavdata'
    
    print("Initializing:")
    func.make_paths([out_dir])                      # create output path if non-existant
    wavfiles = func.read_directory(wav_dir)         # make all wav file instances
    classes = MLfunc.label_encoder(wavfiles)        # make numerical labels
    tt_ratio = 0.1                                  # train/test size ratio
    trainpts,testpts = MLfunc.split_train_test(len(wavfiles),tt_ratio)
    
    trainwavs = [wavfiles[I] for I in trainpts] # wavs to train CLFs
    testwavs = [wavfiles[I] for I in testpts]   # wavs to test CLFs
    print("Number of Training Files:",len(trainpts))
    print("Number of Testing Files:",len(testpts))
    
    SGD_CLFs = MLfunc.SGD_CLFs(['time_clf','freq_clf',
                                    'form_clf','spect_clf'])

    """ Train All Classifiers """ 
    t_0 = time.process_time()
    print("Training Classifiers:")
    SGD_CLFs = MLfunc.train_classifiers(trainwavs,SGD_CLFs,
                             wav_dir,int_dir,classes)
    t_1 = time.process_time()
    print("\tTraining Time:",np.round(t_1-t_0,4),"secs.\n")

    """ Test All Classifiers """
    t_2 = time.process_time()
    print("Testing Classisifers:")
    ytrue,ypred = MLfunc.test_classifiers(testwavs,SGD_CLFs,
                             wav_dir,int_dir,classes)
    t_3 = time.process_time()
    print("\tTestingTime:",np.round(t_3-t_2,4),"secs.\n")

    
    MLfunc.confusion_matrix('Testy',ytrue,ypred,classes,show=True)

    print(time.process_time())



