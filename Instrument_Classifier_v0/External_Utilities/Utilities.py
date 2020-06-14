"""
Landon Buell
PHYS 799
External Utilites
11 June 2020
"""

            #### IMPORTS ####

import numpy as np
import pandas as pd
import os

                        #### VARIABLE & OBJECT DECLARATIONS ####   

accepted_instruments = [
        # Woodwinds
        'AltoFlute','AltoSax','BbClarinet','EbClarinet','Flute',
        'Oboe','SopSax','EbClarinet','BassClarinet','BassFlute',
        'Bassoon',
        # Strings
        'Bass','Cello','Viola','Violin',
        # Brass
        'BassTrombone','Horn','TenorTrombone','Trumpet','Tuba',
        # Percussion
        'bells','Marimba','Vibraphone','Xylophone']

percussion = ['woodblock','triangle','castanet','clave',
              'crotale','tambourine']

cymbals = ['crash','chinese','orchcrash','windgong','ride',
           'hihat','splash','thaigong',]

            #### FUNCTION DEFINITIONS ####

def target_label_encoder(target_vector,write=False):
    """
    Create encoding dictiory of strings to classes
    --------------------------------
    target_vector (arr) : array of target classes as strings
    write (bool/str) : If not False, str is path write out dict 
    --------------------------------
    Return encoding & decoding dictionary
    """
    enc_dict = {}                       # output dictionary
    class_counter = 0                   # class counter
    for category in np.unique(target_vector): # unique elements
        key,val = category,class_counter
        enc_dict.update({key:val})      # update the dictionary
        class_counter += 1              # incriment class counter
    y = [enc_dict[x] for x in target_vector]
    # return the encoding/decoding dictionary and number of classes
    return y

def assign_class (filelist):
    """ Assign class value based on name """
    classes = []
    for name in filelist:
        if name in accepted_instruments:      # in valid instruments
            y = name.upper()    # set instrument
        elif name in percussion:# percussion?
            y = 'PERCUSSION'    # set
        elif name in cymbals:   # cymbals?
            y = 'CYMBAL'        # set
        else:                   # not in lists?
            y = 'OTHER'         # set other
        classes.append(y)       # add to class list
    return classes              # list

def read_directory (path,ext='.wav'):
    """ Read through directory and collect all files with ext """
    FILES = []                              # hold filename
    for roots,dirs,files in os.walk(path):  # walk through tree
        for file in files:                  # each file
            if file.endswith(ext):          # .WAV file
                fullpath = roots+'/'+file
                FILES.append(fullpath)  # add to list
    return FILES                            # return list of files