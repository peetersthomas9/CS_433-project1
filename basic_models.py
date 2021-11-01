"""
File where we implement the 6 basic models asked in part 1 : 
"""

# Import the different libraries and functions 
import numpy as np
import matplotlib.pyplot as plt
from functions import *
from get_data import *



def LS():
    xtrain,ytrain=get_data0()
    print('get data train3')
    # train
    loss, ws = least_square(ytrain,xtrain)
    modelPredictionscase = xtrain@ws.T
    modelPredictions = np.where(modelPredictionscase > .50, 1, 0)
    print('total scrore training set',(modelPredictions == ytrain).sum()/len(ytrain))
    print('TP for y=0 training set',(modelPredictions[ytrain==0] == ytrain[ytrain==0]).sum()/len(ytrain[ytrain==0]))
    print('TP for y=1 training set',(modelPredictions[ytrain==1] == ytrain[ytrain==1]).sum()/len(ytrain[ytrain==1]))

    # Test
    xtest,ytest=get_data0(train=False)
    print('get data test')

    modelPredictionscase=xtest@ws.T
    modelPredictionscase = np.where(modelPredictionscase > .50, 1, -1)

    return modelPredictionscase

def model_GD(max_iters,gamma):

    #train :
    xtr,ytr=get_data0()
    print('get data train1')
    # Initialization
    w_initial = np.zeros(xtr.shape[1])

    #compute the final weights
    gradient_losses, gradient_ws, ws = gradient_descent(ytr, xtr, w_initial, max_iters, gamma)
    
    modelPredictionscase = xtr@ws.T
    modelPredictions = np.where(modelPredictionscase > .50, 1, 0)
    print((modelPredictions == ytr).sum()/len(ytr))
    print((modelPredictions[ytr==0] == ytr[ytr==0]).sum()/len(ytr[ytr==0]))
    print((modelPredictions[ytr==1] == ytr[ytr==1]).sum()/len(ytr[ytr==1]))
    
    #test : 
    xte,yte=get_data0(train=False)
    print('get data test')
    modelPredictionscase = xte@ws.T

    modelPredictionscase = np.where(modelPredictionscase > .50, 1, -1)

    return modelPredictionscase

def model_SGD(max_iters,gamma,batch_size):


    #train :
    xtr,ytr=get_data0()
    print('get data train2')
    # Initialization
    w_initial = np.zeros(xtr.shape[1])

    #compute the final weights
    gradient_losses, gradient_ws, ws = stochastic_gradient_descent(ytr, xtr, w_initial, batch_size,max_iters, gamma)
    
    modelPredictionscase = xtr@ws.T
    modelPredictions = np.where(modelPredictionscase > .50, 1, 0)
    print((modelPredictions == ytr).sum()/len(ytr))
    print((modelPredictions[ytr==0] == ytr[ytr==0]).sum()/len(ytr[ytr==0]))
    print((modelPredictions[ytr==1] == ytr[ytr==1]).sum()/len(ytr[ytr==1]))
    
    #test : 
    xte,yte=get_data0(train=False)
    print('get data test')
    modelPredictionscase = xte@ws.T

    modelPredictionscase = np.where(modelPredictionscase > .50, 1, -1)

    return modelPredictionscase

def model_ridge_regression(lambda_):
    
    xtrain,ytrain=get_data0()
    print('get data train4')
    # train
    loss, ws = ridge_regression(ytrain,xtrain,lambda_)
    modelPredictionscase = xtrain@ws.T
    modelPredictions = np.where(modelPredictionscase > .50, 1, 0)
    print('total scrore training set',(modelPredictions == ytrain).sum()/len(ytrain))
    print('TP for y=0 training set',(modelPredictions[ytrain==0] == ytrain[ytrain==0]).sum()/len(ytrain[ytrain==0]))
    print('TP for y=1 training set',(modelPredictions[ytrain==1] == ytrain[ytrain==1]).sum()/len(ytrain[ytrain==1]))

    # Test
    xtest,ytest=get_data0(train=False)
    print('get data test')

    modelPredictionscase=xtest@ws.T
    modelPredictionscase = np.where(modelPredictionscase > .50, 1, -1)

    return modelPredictionscase

def model_LR(max_iters,gamma):


    #train :
    xtr,ytr=get_data0()
    print('get data train5')
    # Initialization
    w_initial = np.zeros(xtr.shape[1])

    #compute the final weights
    gradient_losses, gradient_ws, ws = gradient_descent_logit(ytr, xtr, w_initial,max_iters, gamma)
    
    modelPredictionscase = sigmoid(xtr@ws.T)
    modelPredictions = np.where(modelPredictionscase > .50, 1, 0)
    print((modelPredictions == ytr).sum()/len(ytr))
    print((modelPredictions[ytr==0] == ytr[ytr==0]).sum()/len(ytr[ytr==0]))
    print((modelPredictions[ytr==1] == ytr[ytr==1]).sum()/len(ytr[ytr==1]))
    
    #test : 
    xte,yte=get_data0(train=False)
    print('get data test 5')
    modelPredictionscase = sigmoid(xte@ws.T)
    modelPredictionscase = xte@ws.T

    modelPredictionscase = np.where(modelPredictionscase > .50, 1, -1)
    return modelPredictionscase


def model_LR_reg(max_iters,gamma,lambda_):

    #train :
    xtr,ytr=get_data0()
    print('get data train6')
    # Initialization
    w_initial = np.zeros(xtr.shape[1])

    #compute the final weights
    gradient_losses, gradient_ws, ws = gradient_descent_logit_regularized(ytr, xtr, w_initial,max_iters, gamma,lambda_)
    
    modelPredictionscase = sigmoid(xtr@ws.T)
    modelPredictions = np.where(modelPredictionscase > .50, 1, 0)
    print((modelPredictions == ytr).sum()/len(ytr))
    print((modelPredictions[ytr==0] == ytr[ytr==0]).sum()/len(ytr[ytr==0]))
    print((modelPredictions[ytr==1] == ytr[ytr==1]).sum()/len(ytr[ytr==1]))
    
    #test : 
    xte,yte=get_data0(train=False)
    print('get data test')

    modelPredictionscase = sigmoid(xte@ws.T)
    modelPredictionscase = xte@ws.T

    modelPredictionscase = np.where(modelPredictionscase > .50, 1, -1)
    return modelPredictionscase
