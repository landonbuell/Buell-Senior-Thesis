"""
Landon buell
PHYS 799
Produce Spectorgram Images
12 Nov 2020
"""

import SystemUtilities as SystUtils
import FeatureUtilities as FeatUtils

if __name__ == '__main__':

    # List of All Files that We Want to Plot
    Initializer = SystUtils.ProgramInitializer()

    # Create Instance of Feature Extractor
    for file in Initializer.wavFiles:           # Each wavefile
        print("\t"+file)
        waveform = SystUtils.ReadFileWAV(file)

        # Create Time-Series Feature Instance
        FrequencyFeatures = FeatUtils.FrequencySeriesFeatures(waveform)
        FrequencyFeatures.__Call__()

        # Get f,t,Sxx
        frequencyAxis = FrequencyFeatures.hertz
        timeAxis = FrequencyFeatures.t  
        spectrogram = FrequencyFeatures.spectrogram
       
        # Plot them
        outputName = file.replace(".wav","")
        SystUtils.PlotSpectrogram(frequencyAxis,timeAxis,spectrogram,outputName)

