"""
Landon 
5 June 2020
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow.keras as keras

def plot_history (hist,model,show=False):
    """
    Visualize Data from Keras History Object Instance
    --------------------------------
    hist (inst) : Keras history object
    --------------------------------
    Return None
    """
    # Initialize Figure

    eps = np.array(hist.epoch)          # arr of epochs
    n_figs = len(hist.history.keys())

    fig,axs = plt.subplots(nrows=n_figs,ncols=1,sharex=True,figsize=(20,8))
    plt.suptitle(model.name+' History',size=50,weight='bold')
    hist_dict = hist.history
    
    for I in range (n_figs):                # over each parameter
        key = list(hist_dict)[I]
        axs[I].set_ylabel(str(key).upper(),size=20,weight='bold')
        axs[I].plot(eps,hist_dict[key])     # plot key
        axs[I].grid()                       # add grid

    plt.xlabel("Epochs",size=20,weight='bold')

    if show == True:
        plt.show()


if __name__ == '__main__':
    
    # JARVIS is stored here
    JARVIS_path = 'C:/Users/Landon/Documents/GitHub/Buell-Senior-Thesis/Instrument_Classifier_v1/INST_CLF_v1/JARVIS'
    JARVIS = keras.models.load_model(filepath=JARVIS_path)
    print(JARVIS.summary())    
    
    # Raw Data is stored here:
    data_path = 'C:/Users/Landon/Documents/wav_data'
    X = pd.read_csv(data_path+'/X.csv',index_col=0).to_numpy()
    y = pd.read_csv(data_path+'/y.csv',index_col=0).to_numpy()
    
    # Pre-processing
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    n_samples , n_features = X.shape
    n_classes = 25
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=0)
    y_train = keras.utils.to_categorical(y_train,n_classes)
    
    # Train
    history = JARVIS.fit(x=X_train,y=y_train,batch_size=128,epochs=100)
    plot_history(history,JARVIS,show=True)
    
    
