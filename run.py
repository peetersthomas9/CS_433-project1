# Import the different libraries and functions 

import numpy as np
import matplotlib.pyplot as plt
from functions import *
from get_data import *
from basic_models import *
from advanced_model import *

"""
Main function to write our prediction of the test set on th 'test.csv' file
In total there is 11 different models (6 basics and 5 avdanced models)
"""
def run(model_number,basic_model=True,max_iter=500,gamma=0.2,batch_size=1,lambda_=0):
    # Get our Predictions
    if basic_model==True:
        if model_number==1:
            modelPredictionscase=model_GD(max_iter,gamma)
        if model_number==2:
            modelPredictionscase=model_SGD(max_iter,gamma, batch_size)
        if model_number==3:
            modelPredictionscase=LS()
        if model_number==4:
            modelPredictionscase=model_ridge_regression(lambda_)
        if model_number==5:
            modelPredictionscase=model_LR(max_iter,gamma)
        if model_number==6:
            modelPredictionscase=model_LR_reg(max_iter,gamma,lambda_)

    else:
        if model_number==1:
            modelPredictionscase=model_A(gamma,max_iter)
        if model_number==2:
            modelPredictionscase=model_B(gamma,max_iter)
        if model_number==3:
            modelPredictionscase=model_C(gamma,max_iter)
        if model_number==4:
            modelPredictionscase=model_D(gamma,max_iter,lambda_)
        if model_number==5:
            modelPredictionscase=model_E(gamma,max_iter)
        if model_number==6:
            modelPredictionscase=model_F(gamma,max_iter,lambda_)

    # Write our prediction in 'test.csv'
    final = modelPredictionscase.astype(np.int)
    print(len(final))
    numbers = np.genfromtxt("data/test.csv",delimiter=",", skip_header=1, usecols=[0])
    final  = np.c_[numbers, final]


    np.savetxt("test.csv", final , delimiter=",",header="Id,Prediction",fmt="%i", comments='')

    return 0


# parameters for our model: 
basic_model=False    # If you want a basic model or advance model 
model_number=6      # the number of the model
max_iters=3000        # number of iteration before it stop
gamma=0.2          # step size
batch_size=1         # batch size 
lambda_=0.01         # coeff for the regularization



# run our functions
run(model_number,basic_model,max_iters,gamma,batch_size,lambda_)



