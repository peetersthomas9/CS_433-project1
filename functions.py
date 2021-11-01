"""
File with different functions we are going to use for project 1
"""

import numpy as np

 ######################################### STANDARDIZE FUNCTION ######################################### 
def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x,0)

    x = x - mean_x
    std_x = np.std(x,0)

    for i in range(0,std_x.shape[0]):

        x[:,i] = x[:,i] / std_x[i] 
    return x

 

 ######################################### COMPUTATION OF GRADIENT ######################################### 
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    # ***************************************************
    
    err = (y-tx@w)
    gradient = -(1/len(y))*(tx.T@err)
    # ***************************************************
  
    return gradient



 ######################################### COMPUTATION OF LOSS ######################################### 
def compute_loss(y, tx, w):
    """Calculate the loss.

    """
    e = (y-tx@w)
    loss = (e**2).mean()/2
                      
 
    return loss

 ######################################### MINI BATCH ITERATOR ######################################### 
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices , :]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]



 ######################################### SIGMOID ######################################### 

def sigmoid(x):
    sig = np.exp(x)/(1+np.exp(x))
    return sig   


 ######################################### COMPUTATION OF LOSS FOR LOGISTIC REGRESSION ######################################### 
def compute_loss_logit(y, tx, w):
    """Calculate the loss.

    """

    e1 = (y*np.log(sigmoid(tx @ w)))
    e2 = ((1-y)*np.log(sigmoid(tx @ w)))
    loss = (e1 + e2).mean() / 2


    return loss

 ######################################### COMPUTATION OF GRADIENT FOR LOGISTIC REGRESSION  ######################################### 

def compute_gradient_logit(y, tx, w):
    """Compute the gradient."""

    err = (y-sigmoid(tx@w))
    gradient = -(1/len(y))*(tx.T@err)
    return gradient

 

 ######################################### COMPUTATION OF LOSS FOR LOGISTIC REGRESSION REGULARIZED ######################################### 
def compute_loss_logit_regularized(y, tx, w, lambda_):
    """Calculate the loss.
    """

    e1 = (y*np.log(sigmoid(tx @ w)))
    e2 = ((1-y)*np.log(sigmoid(tx @ w)))
    loss = (e1 + e2).mean() / 2 +lambda_*w.dot(w.T)


    return loss

######################################### COMPUTATION OF GRADIENT FOR LOGISTIC REGRESSION  REGULARIZED ######################################### 
def compute_gradient_logit_regularized(y, tx, w, lambda_):
    """Compute the gradient."""


    err = (y-sigmoid(tx@w))
    gradient = -(1/len(y))*(tx.T@err)+2*lambda_*w

    return gradient

######################################### COMPUTATION OF LOSS FOR RIDGE REGRESSION ######################################### 
def ridge_regression_loss(y, tx, lambda_, w_ridge):
    
    e=y-tx.dot(w_ridge)
    loss = 1/(2*y.shape[0])*e.T.dot(e)+lambda_*w_ridge.dot(w_ridge.T)
    
    return loss

######################################### POLYNOMIAL FUNCTION  ######################################### 

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        a=np.power(x, deg)
        poly = np.c_[poly, a]
    return poly


######################################### FUNCTION TO SPLIT THE DATA #########################################    

def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    
    data_size = len(y)

    shuffle_indices = np.random.permutation(np.arange(data_size))
    shuffled_y = y[shuffle_indices]
    shuffled_x = x[shuffle_indices]
    
    x_train=shuffled_x[0:round(ratio*data_size)]
    y_train=shuffled_y[0:round(ratio*data_size)]
    x_test=shuffled_x[round(ratio*data_size):]
    y_test=shuffled_y[round(ratio*data_size):]
    


    return x_train, y_train, x_test, y_test





######################################### LEAST SQUARE ######################################### 
def least_square(y,tx):

    w=np.linalg.inv(tx.T.dot(tx)).dot(tx.T).dot(y)
    loss=compute_loss(y, tx, w)

    return loss,w
    
 ######################################### GRADIENT DESCENT ######################################### 

def gradient_descent(y, tx, initial_w, max_iters, gamma,verbose=False):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)

        # ***************************************************
        #update the weights 
        w = w - gamma*(gradient)

        # ***************************************************
        
        ws.append(w)
        losses.append(loss)
        if verbose:
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws, w

 ######################################### STOCHASTIC GRADIENT DESCENT ######################################### 

def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma,verbose=False):
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    ws = [initial_w]
    losses = []
    w = initial_w
    n_iter = 0
    for n_iter in range(max_iters):

        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):

        # ***************************************************
            gradient = compute_gradient(minibatch_y, minibatch_tx, w)
            loss = compute_loss(minibatch_y, minibatch_tx, w)
     
        # ***************************************************

            w = w - gamma*(gradient)
    
            ws.append(w)
            losses.append(loss)
            n_iter +=1
            if verbose:
                print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                      bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))


    return losses, ws, w


######################################### RIDGE REGRESSION ######################################### 

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
   
    w_ridge = np.linalg.inv(tx.T.dot(tx)+2*y.shape[0]*lambda_*np.eye(tx.shape[1])).dot(tx.T).dot(y)
    loss = compute_loss_logit(y, tx, w_ridge)
    return loss, w_ridge



 ######################################### LOGISTIC REGRESSION USING GRADIENT DESCENT ######################################### 

def gradient_descent_logit(y, tx, initial_w, max_iters, gamma_init, verbose=False):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        gradient = compute_gradient_logit(y, tx, w)
        loss = compute_loss_logit(y, tx, w)
        # ***************************************************
        
        gamma=gamma_init*(1-n_iter/(max_iters*1.5)) # change the value of gamma 
        w = w - gamma*(gradient)
        ws.append(w)
        losses.append(loss)
        if verbose:
            print("Gradient Descent({bi}/{ti}): loss={l}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss))

    return losses, ws, w

######################################### REGULARIZED LOGISTIC REGRESSION USING GRADIENT DESCENT ######################################### 

def gradient_descent_logit_regularized(y, tx, initial_w, max_iters, gamma_init, lambda_, verbose=False):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        gradient = compute_gradient_logit_regularized(y, tx, w, lambda_)
        loss = compute_loss_logit_regularized(y, tx, w, lambda_)
        # ***************************************************
        
        gamma=gamma_init*(1-n_iter/(max_iters*2.5)) # change the value of gamma 
        w = w - gamma*(gradient)
       
        ws.append(w)
        losses.append(loss)
       
        if verbose:
            print("Gradient Descent({bi}/{ti}): loss={l}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss))

    return losses, ws, w