from get_data import *
from implementations import *


datcc, yccase, dat, y, ccindex= get_data2(train=True)
initial_w = np.zeros(datcc.shape[1])

#CROSS VALIDATION MODEL 2

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)
    
    
def cross_validation(y, x, k_indices, k, gamma):
    """return the loss of ridge regression."""
    # ***************************************************
    te_indices = k_indices[k]
    tr_indices = k_indices[[i for i in range(len(k_indices)) if i !=k]]
    tr_indices = tr_indices.reshape(-1)
    xtrain = x[tr_indices]
    ytrain = y[tr_indices]
    
    xtest = x[te_indices]
    
    ytest = y[te_indices]
    # ***************************************************
    loss_tr, ws, w = gradient_descent_logit(ytrain, xtrain, initial_w, 300, gamma)
    loss_te = compute_loss_logit(ytest ,xtest , w )

    # ***************************************************
    return loss_tr[-1], loss_te


def cross_validation_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.xlim(1e-4, 2)
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation {}".format(i))

i = "ccase"
seed = 907
k_fold = 4
gammas = np.linspace(0.01,1, 30)
# split data in k fold
k_indices = build_k_indices(yccase, k_fold, seed)
# define lists to store the loss of training data and test data
rmse_tr = []
rmse_te = []
# ***************************************************
for gamma in gammas:
    rmtr = np.zeros(k_fold)
    rmte = np.zeros(k_fold)
    for i in range(0,k_fold):
        rmtr[i] = cross_validation(yccase, datcc, k_indices, i, gamma)[0]
        rmte[i] = cross_validation(yccase, datcc, k_indices, i, gamma)[1]


    rmse_tr.append(rmtr.mean())
    rmse_te.append(rmte.mean())
    # ridge regression with a given lambda
    # ***************************************************
    # ***************************************************   


cross_validation_visualization(gammas, rmse_tr, rmse_te)


initial_w = np.zeros(dat.shape[1])
i = "all"

seed = 907
k_fold = 4
gammas = np.linspace(0.01,1, 30)
# split data in k fold
k_indices = build_k_indices(y, k_fold, seed)
# define lists to store the loss of training data and test data
rmse_tr = []
rmse_te = []
# ***************************************************
for gamma in gammas:
    rmtr = np.zeros(k_fold)
    rmte = np.zeros(k_fold)
    for i in range(0,k_fold):
        rmtr[i] = cross_validation(y, dat, k_indices, i, gamma)[0]
        rmte[i] = cross_validation(y, dat, k_indices, i, gamma)[1]


    rmse_tr.append(rmtr.mean())
    rmse_te.append(rmte.mean())
    # ridge regression with a given lambda
    # ***************************************************
    # ***************************************************


cross_validation_visualization(gammas, rmse_tr, rmse_te)


#CROSS VALIDATION MODEL 3
datcc0, datcc1, datcc2, datcc3, y0, y1, y2, y3, index230, index231, index232, index233=get_data3(train=True)

initial_w = np.zeros(datcc0.shape[1])


seed = 907
k_fold = 4
gammas = np.linspace(0.01,1, 30)
# split data in k fold
k_indices = build_k_indices(y0, k_fold, seed)
# define lists to store the loss of training data and test data
rmse_tr = []
rmse_te = []
# ***************************************************
for gamma in gammas:
    rmtr = np.zeros(k_fold)
    rmte = np.zeros(k_fold)
    for i in range(0,k_fold):
        rmtr[i] = cross_validation(y0, datcc0, k_indices, i, gamma)[0]
        rmte[i] = cross_validation(y0, datcc0, k_indices, i, gamma)[1]


    rmse_tr.append(rmtr.mean())
    rmse_te.append(rmte.mean())
    # ridge regression with a given lambda
    # ***************************************************
    # ***************************************************

i = "1"
cross_validation_visualization(gammas, rmse_tr, rmse_te)

initial_w = np.zeros(datcc1.shape[1])


seed = 907
k_fold = 4
gammas = np.linspace(0.01,1, 30)
# split data in k fold
k_indices = build_k_indices(y1, k_fold, seed)
# define lists to store the loss of training data and test data
rmse_tr = []
rmse_te = []
# ***************************************************
for gamma in gammas:
    rmtr = np.zeros(k_fold)
    rmte = np.zeros(k_fold)
    for i in range(0,k_fold):
        rmtr[i] = cross_validation(y1, datcc1, k_indices, i, gamma)[0]
        rmte[i] = cross_validation(y1, datcc1, k_indices, i, gamma)[1]


    rmse_tr.append(rmtr.mean())
    rmse_te.append(rmte.mean())
    # ridge regression with a given lambda
    # ***************************************************
    # ***************************************************

i = "2"
cross_validation_visualization(gammas, rmse_tr, rmse_te)

initial_w = np.zeros(datcc2.shape[1])


seed = 907
k_fold = 4
gammas = np.linspace(0.01,1, 30)
# split data in k fold
k_indices = build_k_indices(y2, k_fold, seed)
# define lists to store the loss of training data and test data
rmse_tr = []
rmse_te = []
# ***************************************************
for gamma in gammas:
    rmtr = np.zeros(k_fold)
    rmte = np.zeros(k_fold)
    for i in range(0,k_fold):
        rmtr[i] = cross_validation(y2, datcc2, k_indices, i, gamma)[0]
        rmte[i] = cross_validation(y2, datcc2, k_indices, i, gamma)[1]


    rmse_tr.append(rmtr.mean())
    rmse_te.append(rmte.mean())
    # ridge regression with a given lambda
    # ***************************************************
    # ***************************************************

i = "3"

cross_validation_visualization(gammas, rmse_tr, rmse_te)

initial_w = np.zeros(datcc3.shape[1])


seed = 907
k_fold = 4
gammas = np.linspace(0.01,1, 30)
# split data in k fold
k_indices = build_k_indices(y3, k_fold, seed)
# define lists to store the loss of training data and test data
rmse_tr = []
rmse_te = []
# ***************************************************
for gamma in gammas:
    rmtr = np.zeros(k_fold)
    rmte = np.zeros(k_fold)
    for i in range(0,k_fold):
        rmtr[i] = cross_validation(y3, datcc3, k_indices, i, gamma)[0]
        rmte[i] = cross_validation(y3, datcc3, k_indices, i, gamma)[1]


    rmse_tr.append(rmtr.mean())
    rmse_te.append(rmte.mean())
    # ridge regression with a given lambda
    # ***************************************************
    # ***************************************************

i = "4"

cross_validation_visualization(gammas, rmse_tr, rmse_te)