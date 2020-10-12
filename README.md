# Buell-Senior-Thesis
 Repository For all programs, papers, doucments, and records related to Landon's BS Physics Senior thesis (PHYS 799) for Dec. 2020
 
## SignalClassifier
Contains Solution/Project for Main Signal Classifier Program. a README file can be found within this REPO
#### ClassifierMAIN

## SignalClassifier-Preprocessing
Contains tools used to pre-preprocess data before use by main signal classifier. 
#### FileLabelEncoder
Used to read a collection of like-formatted .wav files. A target label is generated based on the start of the string and mapped to a corresponding integer. The ouput is a file that can be read by the main classifier program.
#### ReadChaoticSynthesizers
Used to read and pre-process waveforms from chaotic synthesizer data. This project/program is unstable - See appropriate MATLAB script

## SignalClassifier-Postprocessing
Contains tools used to post-process data after use by main signal classifier
#### ClassifierAnalysis
Reads prediciton or history files outputted by main classifier. Can compute average metric scores, weighted confusion matrices, rolling averages, and produce visualizations.
#### ClassifierFeatures
Used to visualize/interpret/understand raw data as represented in feature space. Can test variances of inter/intra class data. Use to determine "strength" of features and importance to classificiation

## SignalClassifier-CrossValidation
Contains tools to run a K-Folds Cross-Validation Algorithm using the main Signal Classifier and a chosen set of data.



   
