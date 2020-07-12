"""
Landon Buell
PHYS 799
Instrument Classifier v0
12 June 2020
"""

            #### IMPORTS ####

import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split

import System_Utilities as sys_utils
import Feature_Utilities as feat_utils
import Plotting_Utilities as plot_utils
import Machine_Learning_Utilities as ML_utils
import Neural_Network_Utilities


"""
Mode_Utilities.py - 'Mode Utilities'
    Contains Definitions that are only called directly from MAIN script
    Functions are large & perform Groups of important operations
"""

            #### FUNCTION DEFINITIONS ####  

class Program_Mode:
    """
    Base Program mode object from which all programs inhereit from
    --------------------------------
    FILEOBJS (iter) : List-like of File_object instances
    model_names (iter) : List-like of strings calling Network models by name
    group_size (int) : number of file samples in each design matrix
    --------------------------------
    Execute MAIN program in perscribed mode
    """
    def __init__(self,FILEOBJS,model_names,group_size=32):
        """ Inititialize Class Object Instance """
        self.FILEOBJS = FILEOBJS
        self.model_names = model_names
        self.n_files = len(self.FILEOBJS)
        self.group_size = group_size

    def loop_counter(self,cntr,max,text):
        """ Print Loop Counter for User """
        print('\t\t('+str(cntr)+'/'+str(max)+')',text)
        return None

    def collect_features (self,fileobj):
        """ Collected Features from a Given .WAV file object"""
        fileobj = fileobj.read_audio()          # read raw .wav file

        # Time series feature object
        time_features = feat_utils.Time_Series_Features(fileobj.waveform,fileobj.rate)
        
        # Frequency series feature object
        freq_features = feat_utils.Frequency_Series_Features(fileobj.waveform,
                                        fileobj.rate,time_features.frames)

        x1 = ML_utils.Feature_Array(fileobj.target)     # Structure holds MLP features
        x1 = x1.set_features(np.zeros(shape=(15,)))     # set features

        x2 = ML_utils.Feature_Array(fileobj.target)     # strucutre to hold Sxx features
        x2 = x2.set_features(freq_features.Sxx)         # set spectrogram

        x3 = ML_utils.Feature_Array(fileobj.target)     # structure to hold PSC features
        x3 = x3.set_features(np.zeros(shape=(2,2048)))  # set features

        return x1,x2,x3

    def construct_design_matrices (self,FILES):
        """ Collect Features from a subset File Objects """
        X1 = ML_utils.Design_Matrix(ndim=2)     # Design matrix for MLP
        X2 = ML_utils.Design_Matrix(ndim=4)     # Design matrix for Spectrogram
        X3 = ML_utils.Design_Matrix(ndim=4)     # Design matrix for Phase-Space

        for i,FILEOBJ in enumerate(FILES):
            self.loop_counter(i,self.group_size,FILEOBJ.filename)   # print messege
            x1,x2,x3 = self.collect_features(FILEOBJ)               # collect features

            X1.add_sample(x1)   # Add sample to MLP
            X2.add_sample(x2)   # Add sample to Sxx
            X3.add_sample(x3)   # add sample to Psc

        X1 = X1.shape_by_sample()
        X2 = X2.pad_2D(new_shape=Neural_Network_Utilities.spectrogram_shape)
        X3 = X3.pad_2D(new_shape=Neural_Network_Utilities.phasespace_shape)
        
        return [X1,X2,X3]


class Train_Mode (Program_Mode):
    """
    Run Program in 'Train Mode'
        Inherits from 'Program_Mode' parent class
    --------------------------------
    FILEOBJS (iter) : List-like of File_object instances
    model_names (iter) : List-like of strings calling Network models by name
    --------------------------------
    Creates Program Train Mode Object
    """
    def __init__(self,FILEOBJS,model_names):
        """ Instantiate Class Method """
        super().__init__(FILEOBJS,model_names)

        # For each model store a list of history objs
        for model in self.model_names:              
            setattr(self,str(model)+'_history',[])

    def __call__(self,Networks):
        """ Call this Instance to Execute Training and Testing """
        print("\nBegining Training process....")
        group_cntr = 0
        for I in range (0,self.n_files,self.group_size):    # In a given group
            print("\tGroup Number:",group_cntr)
            FILES = self.FILEOBJS[I:I+self.group_size]      # subset of files
            Design_Matrices = super().construct_design_matrices(FILES)
            epcs = 16                                        # trainign epochs

            for dataset,modelpath in zip(Design_Matrices,self.model_names):
                print("\tLoading & Fitting Model:",modelpath)
                X = dataset.__get_X__()                     # Features
                Y = dataset.__get_Y__(Networks.n_classes)   # Labels
                MODEL = Networks.__loadmodel__(modelpath)   # Load network
                # Fit the model!
                history = MODEL.fit(x=X,y=Y,batch_size=32,epochs=epcs,verbose=2,
                                    initial_epoch=(group_cntr*epcs))
                # Save the Model, Store the history
                Networks.__savemodel__(MODEL)               # save model
                self.store_history(history,modelpath)       # store data
            group_cntr += 1

        print("\tTraining Completed! =)")
        return self

    def store_history (self,history_object,model):
        """ Store Keras History Object in lists """
        assert model in self.model_names    # must be a known model
        # Get the list and add the object
        model_history = self.__getattribute__(str(model)+'_history') 
        model_history.append(history_object)
        return self

    def aggregate_history (self):
        """ Aggregate history objects by model type & metric type """
        pass

class Test_Mode (Program_Mode):
    """
    Run Program in 'Test Mode'
        Inherits from 'Program_Mode' parent class
    --------------------------------
    FILEOBJS (iter) : List-like of File_object instances
    model_names (iter) : List-like of strings calling Network models by name
    labels (bool) : If True, labels are used to test, If False, Predictions are make
    --------------------------------
    Creates Program Test Mode Object
    """
    def __init__(self, FILEOBJS,model_names,labels=False):
        super().__init__(FILEOBJS,model_names)
        self.labels = labels        # labels for the trainign set?
        if self.labels == True:     # we have labels!
            self.Y = []             # hold them in list

        # For each model store a list of prediction arrays
        for model in self.model_names:              
            setattr(self,str(model)+'_predicitons',[])

    def __call__(self,Networks):
        """ Call this Instance to Execute Training and Testing """
        print("\nBegining Testing Process...")
        group_cntr = 0
        for I in range (0,self.n_files,self.group_size):    # In a given group
            print("\tGroup Number:",group_cntr)
            FILES = self.FILEOBJS[I:I+self.group_size]      # subset of files
            Design_Matrices = super().construct_design_matrices(FILES)
            epcs = 16                                       # trainign epochs

            for dataset,modelpath in zip(Design_Matrices,self.model_names):
                X = dataset.__get_X__()                     # Features
                if self.labels == True:         # if we have labels
                    self.store_labels(Y)        # store labels as well!
                    Y = dataset.__get_Y__(Networks.n_classes)   # Labels
                MODEL = Networks.__loadmodel__(modelpath)   # Load network
                # Make Predictions on Data
                Z = MODEL.predict(x=X,batch_size=32)
                # Save the Model, Store the history
                Networks.__savemodel__(MODEL)           # save model
                self.store_predictions(Z,modelpath)     # store predictions               
            group_cntr += 1
        print("\tTraining Completed! =)")
        return self

    def store_predictions (self,predictions,model):
        """ Export Predicitions Made by Models to Humna-readable format """
        assert model in self.model_names    # must be a known model
        # Get the array and add the object
        model_predicitons = self.__getattribute__(str(model)+'_predicitons') 
        model_history.append(predictions)
        return self

    def store_labels (self,y):
        """ If Labels are present, store for each sample """
        self.Y.append(y)

    def aggregate_predictions (self):
        """ aggregate predictions by model type """
        pass

class TrainTest_Mode (Program_Mode):
    """
    Run Program in 'Train_Mode' and 'Test Mode' sequentially
        Inherits from 'Train_Mode' and 'Test Mode' parent classes
    --------------------------------
    FILEOBJS (iter) : List-like of File_object instances
    labels (bool) : If True, labels are used to test, If False, Predictions are make
    --------------------------------
    Creates Program Test Mode Object
    """
    def __init__(self, FILEOBJS,model_names,testsize=0.1,labels=True):
        """ Initialize Class oject instance """
        super().__init__(FILEOBJS,model_names)
        self.testsize=testsize  # train/test size
        self.labels = labels    # labels?
        self.Split_Objs()       # split objs
        
       
    def Split_Objs (self):
        """ Split objects into training.testing subsets """
        train,test = train_test_split(self.FILEOBJS,test_size=self.testsize)
        self.TRAIN_FILEOBJS = train
        self.n_train_files = len(self.TRAIN_FILEOBJS)
        self.TEST_FILEOBJS = test
        self.n_test_files = len(self.TEST_FILEOBJS)
        return self

    def __call__(self,Networks):
        """ Call this Instance to Execute Training and Testing """

        Training = Train_Mode(self.TRAIN_FILEOBJS,self.model_names)
        Training.__call__(Networks)

        Testing = Test_Mode(self.TEST_FILEOBJS,self.model_names,labels)
        Testing.__call__(Networks)
            
        return self


        