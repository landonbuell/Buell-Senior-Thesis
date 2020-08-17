"""
Landon Buell
Classifier Preprocessing Module
PHYS 799
16 August 2020
"""

        #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import os
import Preprocessing_Utils as utils

        #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    Features = utils.FrequencySeriesFeatures()

    banks = Features.MelFrequencyCeptsralCoefficients(n_filters=10)

    for bank in banks:
        plt.plot(Features.hertz,bank)
    plt.show()