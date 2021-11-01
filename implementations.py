import numpy as np


def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = (y - tx @ w)
    loss = (e ** 2).mean() / 2

    # sqerror = 0
    # ***************************************************
    # for i in range(tx.shape[0]):
    #    sqerror += (1/2)*(y[i]-tx[i,]@np.transpose(w))**2
    # ***************************************************
    # raise NotImplementedError
    return loss


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    # ***************************************************
    err = (y - tx @ w)
    print(err)
    gradient = -(1 / len(y)) * (tx.T @ err)
    # ***************************************************
    # raise NotImplementedError
    return gradient


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss

    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        # ***************************************************
        # raise NotImplementedError
        # ***************************************************
        w = w - gamma * (gradient)
        # ***************************************************
        # raise NotImplementedError
        # store w and loss

        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return loss, w


def least_squares_SGD(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # ***************************************************

    w = initial_w
    n_iter = 0
    for n_iter in range(max_iters):

        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):

        # ***************************************************
            print(tx.shape)
            print(w.shape)
            gradient = compute_gradient(minibatch_y, minibatch_tx, w)
            loss = compute_loss(minibatch_y, minibatch_tx, w)
        # ***************************************************
        #raise NotImplementedError
        # ***************************************************
            w = w + gamma*(gradient)
        # ***************************************************
        #raise NotImplementedError
        # store w and loss

            n_iter +=1
           # print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
           #   bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    # ***************************************************
    #raise NotImplementedError
    return loss, w

def least_squares(y, tx):
    """calculate the least squares solution."""
    # ***************************************************
    a = tx.T@tx
    b = tx.T@y
    #w = np.linalg.inv((tx.T@tx))@tx.T@y
    w = np.linalg.solve(a, b)
    e = (y-tx@w)
    mse = (e**2).mean()/2
    # returns mse, and optimal weights
    # ***************************************************
    #raise NotImplementedError
    return mse, w


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    # ***************************************************
    a1 = tx.T@tx
    idm = np.identity(tx.shape[1])
    a2 = a1 +lambda_*idm*2*tx.shape[0]
    b = tx.T@y
    w = np.linalg.solve(a2, b)
    mse = compute_loss(y,tx, w)
    # returns mse, and optimal weights
    # ***************************************************
    # ***************************************************
    return mse, w

def compute_gradient_logit(y, tx, w):
    """Compute the gradient."""
    # ***************************************************
    # TODO: compute gradient and error vector

    err = (sigmoid(tx@w)-y)
    gradient = (tx.T@err)*(1/len(y))
    # ***************************************************
    #raise NotImplementedError
    return -gradient

def sigmoid(x):
    sig = np.exp(x)/(1+np.exp(x))
    return sig


def compute_loss_logit(y, tx, w):
    """compute the loss: negative log likelihood."""
    # ***************************************************
    pred= sigmoid(tx@w)
    loss = y*np.log(pred)+(1-y)*np.log(1-pred)
    # ***************************************************
    return -loss.mean()


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss

    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        gradient = compute_gradient_logit(y, tx, w)
        loss = compute_loss_logit(y, tx, w)
        # ***************************************************
        # ***************************************************
        w = w + gamma*(gradient)
        # ***************************************************


    return loss, w

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


def reg_logistic_regression(y, tx, lambda , initial w, max iters, gamma)):
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
       
       
    return loss, w
