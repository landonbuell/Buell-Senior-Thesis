"""
Landon Buell
Kevin Short
Plane of Best Fit + Noise v0
29 March 2020
"""

            #### IMPORTS ####

import numpy as np
import pandas as pd
import time

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

import InputTrainingData_Planar_w_Noise as planardata

            #### FUNCTIONS ####

def percent_errors (approx,exact):
    """ Compute the percent error of arrays """
    return (approx - exact)/exact

def scale_matrix (matrix):
    """ Normalize Columns of matrix s.t. max(col) = 1 """
    if matrix.ndim > 1:                 # more than 1D
        matrix = np.transpose(matrix)   # transpose
        for row in matrix:              # each row
            row = row/np.max(row)       # divide row by maximum of row
        return matrix.transpose()       # return the transpose
    else:
        matrix = matrix/np.max(matrix)  # divide by max
        return matrix

def testing_data (N_samples,scale=True):
    """ Create sample of 'true' data based on linear equation """
    x = np.random.random(size=N_samples)
    y = np.random.random(size=N_samples)
    c = np.ones(N_samples)
    features = np.array([c,x,y]).transpose()
    targets = 0.3 + 0.4*x + 0.5*y
    if scale == True:
        features = scale_matrix(features)
        targets = scale_matrix(targets)
    return features,targets

if __name__ == '__main__':

    # Load in datasets
    X_train = planardata.train_inputs
    y_train = planardata.train_outputs.ravel()

    X_train = scale_matrix(X_train)     # scale the training data
    y_train = scale_matrix(y_train)     # scale the training data

    output_matrix = np.array([])        # this will be written out to CSV

    N_iters = 100

    for I in np.arange(0,N_iters,1):
        # Create & Fit sklearn MLP Regressor
        t_0 = time.process_time()
        MLP_Reg = MLPRegressor(hidden_layer_sizes=(),activation='relu',
                               solver='sgd',max_iter=1000,tol=1e-4,
                               random_state=None)
        t_0 = time.process_time()
        MLP_Reg.fit(X_train,y_train)
        t_f = time.process_time()

        # Test the model
        X_test,z_true = testing_data(N_samples=100)
        z_pred = MLP_Reg.predict(X_test)
        MSE = mean_squared_error(z_true,z_pred)
        dt = np.round(t_f-t_0,8)

        # Collect data from iteration
        row = np.array([])              # Hold data in array
        approx_coefs = np.array(MLP_Reg.coefs_).ravel()
        coeffs_error = percent_errors(approx_coefs,[0.3,0.4,0.5])
        row = np.append(row,approx_coefs)   # add approximate coeffs
        row = np.append(row,coeffs_error)   # add % error 
        row = np.append(row,MSE)            # add MSE to output
        row = np.append(row,dt)             # add train time to output
        output_matrix = np.append(output_matrix,row)

    output_matrix = output_matrix.reshape(N_iters,8)
    frame = pd.DataFrame(data=output_matrix,
                         columns=['A Predicted','B Predicted','C Predicted',
                                  'A Error','B Error','C Error',
                                  'MSE','Train Time'])
    frame.to_csv('Planar_Fit_ReLU_scaled.csv')

        
    
        



    
    






