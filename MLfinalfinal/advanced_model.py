"""
Advanced models : 
"""
# Import the different libraries and functions 
import numpy as np
import matplotlib.pyplot as plt
from functions import *
from get_data import *


"""
Model A: Use independent variables modeled asbinary (0/1) + Gradient descent
"""
def model_A(gamma,max_iter):

    #train :
    datacc, yccase, xtr, ytr, index=get_data2()

    # Initialization
    w_initial = np.zeros(xtr.shape[1])

    #compute the final weights
    gradient_losses, gradient_ws, ws = gradient_descent(ytr, xtr, w_initial,max_iter,gamma)
    
    # compute score on the training set
    modelPredictionscase = xtr@ws.T
    modelPredictions = np.where(modelPredictionscase > .50, 1, 0)
    print((modelPredictions == ytr).sum()/len(ytr))
    print((modelPredictions[ytr==0] == ytr[ytr==0]).sum()/len(ytr[ytr==0]))
    print((modelPredictions[ytr==1] == ytr[ytr==1]).sum()/len(ytr[ytr==1]))
    
    #test : 
    datacc, yccase, xte,yte, index=get_data2(train=False)
    modelPredictionscase = xte@ws.T
    modelPredictionscase = np.where(modelPredictionscase > .50, 1, -1)

    return modelPredictionscase

"""
Model B: Use independent variables modeled asbinary (0/1) + logistic gradient descent
"""
def model_B(gamma,max_iter):
     #train :
    datacc, yccase,xtr, ytr, index=get_data2()
    # Initialization
    w_initial = np.zeros(xtr.shape[1])

    #compute the final weights
    gradient_losses, gradient_ws, ws = gradient_descent_logit(ytr, xtr, w_initial,max_iter, gamma)

    # compute score on the training set
    modelPredictionscase = sigmoid(xtr@ws.T)
    modelPredictionscase = xtr@ws.T
    modelPredictions = np.where(modelPredictionscase > .50, 1, 0)
    print((modelPredictions == ytr).sum()/len(ytr))
    print((modelPredictions[ytr==0] == ytr[ytr==0]).sum()/len(ytr[ytr==0]))
    print((modelPredictions[ytr==1] == ytr[ytr==1]).sum()/len(ytr[ytr==1]))
    
    #test : 
    datacc, yccase,xte,yte, index=get_data2(train=False)

    modelPredictionscase = xte@ws.T
    modelPredictionscase = np.where(modelPredictionscase > .50, 1, -1)

    return modelPredictionscase

"""
Model C: Use four strata for the JET number with independent variables modeled as binary + logistic regression
"""
def model_C(gamma,max_iter):
     #train :

     # split the data in four groups (depending on the jet number value)
    xtr0, xtr1, xtr2, xtr3, ytr0, ytr1, ytr2, ytr3, index230, index231, index232, index233=get_data3()
    # Initialization
    w_initial0 = np.zeros(xtr0.shape[1])
    w_initial1 = np.zeros(xtr1.shape[1])
    w_initial2 = np.zeros(xtr2.shape[1])
    w_initial3 = np.zeros(xtr3.shape[1])


    #compute the final weights for each group 
    gradient_losses0, gradient_ws0, ws0 = gradient_descent_logit(ytr0, xtr0, w_initial0,max_iter, gamma)
    gradient_losses1, gradient_ws1, ws1 = gradient_descent_logit(ytr1, xtr1, w_initial1,max_iter, gamma)
    gradient_losses2, gradient_ws2, ws2 = gradient_descent_logit(ytr2, xtr2, w_initial2,max_iter, gamma)
    gradient_losses3, gradient_ws3, ws3 = gradient_descent_logit(ytr3, xtr3, w_initial3,max_iter, gamma)

    # compute score on the training set for each group 
    modelPredictionscase = sigmoid(xtr0@ws0.T)
    modelPredictions = np.where(modelPredictionscase > .50, 1, 0)
    print((modelPredictions == ytr0).sum()/len(ytr0))
    print((modelPredictions[ytr0==0] == ytr0[ytr0==0]).sum()/len(ytr0[ytr0==0]))
    print((modelPredictions[ytr0==1] == ytr0[ytr0==1]).sum()/len(ytr0[ytr0==1]))

    modelPredictionscase = sigmoid(xtr1@ws1.T)
    modelPredictions = np.where(modelPredictionscase > .50, 1, 0)
    print((modelPredictions == ytr1).sum()/len(ytr1))
    print((modelPredictions[ytr1==0] == ytr1[ytr1==0]).sum()/len(ytr1[ytr1==0]))
    print((modelPredictions[ytr1==1] == ytr1[ytr1==1]).sum()/len(ytr1[ytr1==1]))

    modelPredictionscase = sigmoid(xtr2@ws2.T)
    modelPredictions = np.where(modelPredictionscase > .50, 1, 0)
    print((modelPredictions == ytr2).sum()/len(ytr2))
    print((modelPredictions[ytr2==0] == ytr2[ytr2==0]).sum()/len(ytr2[ytr2==0]))
    print((modelPredictions[ytr2==1] == ytr2[ytr2==1]).sum()/len(ytr2[ytr2==1]))

    modelPredictionscase = sigmoid(xtr3@ws3.T)
    modelPredictions = np.where(modelPredictionscase > .50, 1, 0)
    print((modelPredictions == ytr3).sum()/len(ytr3))
    print((modelPredictions[ytr3==0] == ytr3[ytr3==0]).sum()/len(ytr3[ytr3==0]))
    print((modelPredictions[ytr3==1] == ytr3[ytr3==1]).sum()/len(ytr3[ytr3==1]))
    
    #test : 
    xte0, xte1, xte2, xte3, yte0, yte1, yte2, yte3, index230, index231, index232, index233=get_data3(train=False)

    modelPredictionscase0 = sigmoid(xte0@ws0.T)
    modelPredictionscase0 = np.where(modelPredictionscase0 > .50, 1, -1)

    modelPredictionscase1 = sigmoid(xte1@ws1.T)
    modelPredictionscase1 = np.where(modelPredictionscase1 > .50, 1, -1)

    modelPredictionscase2 = sigmoid(xte2@ws2.T)
    modelPredictionscase2 = np.where(modelPredictionscase2 > .50, 1, -1)

    modelPredictionscase3 = sigmoid(xte3@ws3.T)
    modelPredictionscase3 = np.where(modelPredictionscase3 > .50, 1, -1)

    # Put the prediction on a one arra
    modelPredictionscase = np.zeros(xte0.shape[0]+xte1.shape[0]+xte2.shape[0]+xte3.shape[0])

    modelPredictionscase[index233] = modelPredictionscase3
    modelPredictionscase[index232] = modelPredictionscase2 
    modelPredictionscase[index231] = modelPredictionscase1 
    modelPredictionscase[index230] = modelPredictionscase0 

    return modelPredictionscase

"""
Model D: Use polynomial and trigonometric expension on the initial features + logistic regression
"""
def model_D(gamma,max_iter,lambda_):
    
    xtr,ytr=get_data4()
    #train :
    # Initialization
    w_initial = np.zeros(xtr.shape[1])

    #compute the final weights
    gradient_losses, gradient_ws, ws = gradient_descent_logit_regularized(ytr, xtr, w_initial,max_iter, gamma,lambda_)
    
    modelPredictionscase = xtr@ws.T
    modelPredictions = np.where(modelPredictionscase > .50, 1, 0)
    print((modelPredictions == ytr).sum()/len(ytr))
    print((modelPredictions[ytr==0] == ytr[ytr==0]).sum()/len(ytr[ytr==0]))
    print((modelPredictions[ytr==1] == ytr[ytr==1]).sum()/len(ytr[ytr==1]))
    
    #test : 
    xte,yte=get_data4(train=False)
    print('get data test')

    modelPredictionscase = sigmoid(xte@ws.T)

    modelPredictionscase = np.where(modelPredictionscase > .50, 1, -1)
    return modelPredictionscase


"""
Model E: Use four strata for the JET number and use polynomial and trigonometric expension on the features + logistic regression
"""
def model_E(gamma,max_iter):
     #train :
    xtr0, xtr1, xtr2, xtr3, ytr0, ytr1, ytr2, ytr3, index230, index231, index232, index233=get_data5()
    # Initialization
    w_initial0 = np.zeros(xtr0.shape[1])
    w_initial1 = np.zeros(xtr1.shape[1])
    w_initial2 = np.zeros(xtr2.shape[1])
    w_initial3 = np.zeros(xtr3.shape[1])

    #compute the final weights
    gradient_losses0, gradient_ws0, ws0 = gradient_descent_logit(ytr0, xtr0, w_initial0,max_iter, gamma)
    gradient_losses1, gradient_ws1, ws1 = gradient_descent_logit(ytr1, xtr1, w_initial1,max_iter, gamma)
    gradient_losses2, gradient_ws2, ws2 = gradient_descent_logit(ytr2, xtr2, w_initial2,max_iter, gamma)
    gradient_losses3, gradient_ws3, ws3 = gradient_descent_logit(ytr3, xtr3, w_initial3,max_iter, gamma)

    # compute the prediction of the train
    modelPredictionscase = sigmoid(xtr0@ws0.T)
    modelPredictions0 = np.where(modelPredictionscase > .50, 1, 0)
    print((modelPredictions0 == ytr0).sum()/len(ytr0))
    print((modelPredictions0[ytr0==0] == ytr0[ytr0==0]).sum()/len(ytr0[ytr0==0]))
    print((modelPredictions0[ytr0==1] == ytr0[ytr0==1]).sum()/len(ytr0[ytr0==1]))

    modelPredictionscase = sigmoid(xtr1@ws1.T)
    modelPredictions1 = np.where(modelPredictionscase > .50, 1, 0)
    print((modelPredictions1 == ytr1).sum()/len(ytr1))
    print((modelPredictions1[ytr1==0] == ytr1[ytr1==0]).sum()/len(ytr1[ytr1==0]))
    print((modelPredictions1[ytr1==1] == ytr1[ytr1==1]).sum()/len(ytr1[ytr1==1]))

    modelPredictionscase = sigmoid(xtr2@ws2.T)
    modelPredictions2 = np.where(modelPredictionscase > .50, 1, 0)
    print((modelPredictions2 == ytr2).sum()/len(ytr2))
    print((modelPredictions2[ytr2==0] == ytr2[ytr2==0]).sum()/len(ytr2[ytr2==0]))
    print((modelPredictions2[ytr2==1] == ytr2[ytr2==1]).sum()/len(ytr2[ytr2==1]))

    modelPredictionscase = sigmoid(xtr3@ws3.T)
    modelPredictions3 = np.where(modelPredictionscase > .50, 1, 0)
    print((modelPredictions3 == ytr3).sum()/len(ytr3))
    print((modelPredictions3[ytr3==0] == ytr3[ytr3==0]).sum()/len(ytr3[ytr3==0]))
    print((modelPredictions3[ytr3==1] == ytr3[ytr3==1]).sum()/len(ytr3[ytr3==1]))
    
    
    modelPredictionscase = np.zeros(xtr0.shape[0]+xtr1.shape[0]+xtr2.shape[0]+xtr3.shape[0])

    ycase = np.zeros(xtr0.shape[0]+xtr1.shape[0]+xtr2.shape[0]+xtr3.shape[0])
    modelPredictionscase[index233] = modelPredictions3
    modelPredictionscase[index232] = modelPredictions2 
    modelPredictionscase[index231] = modelPredictions1 
    modelPredictionscase[index230] = modelPredictions0 
    ycase[index233]= ytr3
    ycase[index232]= ytr2
    ycase[index231]= ytr1
    ycase[index230] = ytr0
    print((modelPredictionscase == ycase).sum()/len(ycase))
    print((modelPredictionscase[ycase==0] == ycase[ycase==0]).sum()/len(ycase[ycase==0]))
    print((modelPredictionscase[ycase==1] == ycase[ycase==1]).sum()/len(ycase[ycase==1]))
    
    #test : 
    xte0, xte1, xte2, xte3, yte0, yte1, yte2, yte3, index230, index231, index232, index233=get_data5(train=False)

    modelPredictionscase0 = sigmoid(xte0@ws0.T)
    modelPredictionscase0 = np.where(modelPredictionscase0 > .50, 1, -1)

    modelPredictionscase1 = sigmoid(xte1@ws1.T)
    modelPredictionscase1 = np.where(modelPredictionscase1 > .50, 1, -1)

    modelPredictionscase2 = sigmoid(xte2@ws2.T)
    modelPredictionscase2 = np.where(modelPredictionscase2 > .50, 1, -1)

    modelPredictionscase3 = sigmoid(xte3@ws3.T)
    modelPredictionscase3 = np.where(modelPredictionscase3 > .50, 1, -1)

    modelPredictionscase = np.zeros(xte0.shape[0]+xte1.shape[0]+xte2.shape[0]+xte3.shape[0])
    ycase = np.zeros(xte0.shape[0]+xte1.shape[0]+xte2.shape[0]+xte3.shape[0])

    modelPredictionscase[index233] = modelPredictionscase3
    modelPredictionscase[index232] = modelPredictionscase2 
    modelPredictionscase[index231] = modelPredictionscase1 
    modelPredictionscase[index230] = modelPredictionscase0 
    

    
    return modelPredictionscase

"""
Model F: Use four strata for the JET number and use polynomial and trigonometric expension on the features + regularized logistic regression
"""
def model_F(gamma,max_iter,lambda_):
     #train :
    xtr0, xtr1, xtr2, xtr3, ytr0, ytr1, ytr2, ytr3, index230, index231, index232, index233=get_data5()
    # Initialization
    w_initial0 = np.zeros(xtr0.shape[1])
    w_initial1 = np.zeros(xtr1.shape[1])
    w_initial2 = np.zeros(xtr2.shape[1])
    w_initial3 = np.zeros(xtr3.shape[1])

    #compute the final weights
    gradient_losses0, gradient_ws0, ws0 = gradient_descent_logit_regularized(ytr0, xtr0, w_initial0,max_iter, gamma, lambda_)
    gradient_losses1, gradient_ws1, ws1 = gradient_descent_logit_regularized(ytr1, xtr1, w_initial1,max_iter, gamma,lambda_)
    gradient_losses2, gradient_ws2, ws2 = gradient_descent_logit_regularized(ytr2, xtr2, w_initial2,max_iter, gamma, lambda_)
    gradient_losses3, gradient_ws3, ws3 = gradient_descent_logit_regularized(ytr3, xtr3, w_initial3,max_iter, gamma, lambda_)

    # compute the prediction of the train
    modelPredictionscase = sigmoid(xtr0@ws0.T)
    modelPredictions0 = np.where(modelPredictionscase > .50, 1, 0)
    print((modelPredictions0 == ytr0).sum()/len(ytr0))
    print((modelPredictions0[ytr0==0] == ytr0[ytr0==0]).sum()/len(ytr0[ytr0==0]))
    print((modelPredictions0[ytr0==1] == ytr0[ytr0==1]).sum()/len(ytr0[ytr0==1]))

    modelPredictionscase = sigmoid(xtr1@ws1.T)
    modelPredictions1 = np.where(modelPredictionscase > .50, 1, 0)
    print((modelPredictions1 == ytr1).sum()/len(ytr1))
    print((modelPredictions1[ytr1==0] == ytr1[ytr1==0]).sum()/len(ytr1[ytr1==0]))
    print((modelPredictions1[ytr1==1] == ytr1[ytr1==1]).sum()/len(ytr1[ytr1==1]))

    modelPredictionscase = sigmoid(xtr2@ws2.T)
    modelPredictions2 = np.where(modelPredictionscase > .50, 1, 0)
    print((modelPredictions2 == ytr2).sum()/len(ytr2))
    print((modelPredictions2[ytr2==0] == ytr2[ytr2==0]).sum()/len(ytr2[ytr2==0]))
    print((modelPredictions2[ytr2==1] == ytr2[ytr2==1]).sum()/len(ytr2[ytr2==1]))

    modelPredictionscase = sigmoid(xtr3@ws3.T)
    modelPredictions3 = np.where(modelPredictionscase > .50, 1, 0)
    print((modelPredictions3 == ytr3).sum()/len(ytr3))
    print((modelPredictions3[ytr3==0] == ytr3[ytr3==0]).sum()/len(ytr3[ytr3==0]))
    print((modelPredictions3[ytr3==1] == ytr3[ytr3==1]).sum()/len(ytr3[ytr3==1]))
    
    
    modelPredictionscase = np.zeros(xtr0.shape[0]+xtr1.shape[0]+xtr2.shape[0]+xtr3.shape[0])

    ycase = np.zeros(xtr0.shape[0]+xtr1.shape[0]+xtr2.shape[0]+xtr3.shape[0])
    modelPredictionscase[index233] = modelPredictions3
    modelPredictionscase[index232] = modelPredictions2 
    modelPredictionscase[index231] = modelPredictions1 
    modelPredictionscase[index230] = modelPredictions0 
    ycase[index233]= ytr3
    ycase[index232]= ytr2
    ycase[index231]= ytr1
    ycase[index230] = ytr0
    print((modelPredictionscase == ycase).sum()/len(ycase))
    print((modelPredictionscase[ycase==0] == ycase[ycase==0]).sum()/len(ycase[ycase==0]))
    print((modelPredictionscase[ycase==1] == ycase[ycase==1]).sum()/len(ycase[ycase==1]))
    
    #test : 
    xte0, xte1, xte2, xte3, yte0, yte1, yte2, yte3, index230, index231, index232, index233=get_data5(train=False)

    modelPredictionscase0 = sigmoid(xte0@ws0.T)
    modelPredictionscase0 = np.where(modelPredictionscase0 > .50, 1, -1)

    modelPredictionscase1 = sigmoid(xte1@ws1.T)
    modelPredictionscase1 = np.where(modelPredictionscase1 > .50, 1, -1)

    modelPredictionscase2 = sigmoid(xte2@ws2.T)
    modelPredictionscase2 = np.where(modelPredictionscase2 > .50, 1, -1)

    modelPredictionscase3 = sigmoid(xte3@ws3.T)
    modelPredictionscase3 = np.where(modelPredictionscase3 > .50, 1, -1)

    modelPredictionscase = np.zeros(xte0.shape[0]+xte1.shape[0]+xte2.shape[0]+xte3.shape[0])
    ycase = np.zeros(xte0.shape[0]+xte1.shape[0]+xte2.shape[0]+xte3.shape[0])

    modelPredictionscase[index233] = modelPredictionscase3
    modelPredictionscase[index232] = modelPredictionscase2 
    modelPredictionscase[index231] = modelPredictionscase1 
    modelPredictionscase[index230] = modelPredictionscase0 
    

    
    return modelPredictionscase