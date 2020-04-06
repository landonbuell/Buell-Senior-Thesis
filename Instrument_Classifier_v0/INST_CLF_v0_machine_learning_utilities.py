"""
Landon Buell
Instrument Classifier v0
Machine Learning Utility Functions
6 April 2020
"""

            #### IMPORTS ####

import numpy as np

from sklearn.neural_network import MLPClassifier
 

"""
INSTRUMENT CLASSIFIER V0 - MACHINE LEARNING UTILITIES


"""

            #### FUNCTION DEFINTIONS ####

def split_train_test (data,tt_ratio=0.6):
    """
    generate a series of indicies for training & testing data
        Adapted from (Geron, 49) (Note: numpy is Psuedo-Random)
    --------------------------------
    data (iterable) : list or collection of data to split
    ratio (float) : ratio of train: test data on the bound (0,1)
    --------------------------------
    return lists of training obj instances & testing obj instances 
    """
    n_samples = len(data)                       # number of samples in data
    shuffled = np.random.permutation(n_samples) # permute idxs
    train_size = int(n_samples*tt_ratio)        # test dataset size
    # create lists of indexes for training/testing
    train_pts = shuffled[:train_size].tolist()      
    test_pts = shuffled[train_size:].tolist()
    # breaking into list of training & testing data
    training_data = [data[x] for x in train_pts]
    testing_data = [data[x] for x in test_pts]
    return training_data,testing_data           # return the two lists

def target_label_encoder(target_vector):
    """
    Create encoding dictiory of strings to classes
    --------------------------------
    target_vector (arr) : array of target classes as strings
    --------------------------------
    Return encoding dictionary
    """
    enc_dict = {}                       # output dictionary
    class_counter = 0                   # class counter
    for instrument in np.unique(target_vector): # unique elements
        key,val = instrument,class_counter
        enc_dict.update({key:val})      # update the dictionary
        class_counter += 1              # incriment class counter
    return enc_dict                     # return the encofing dictionary
