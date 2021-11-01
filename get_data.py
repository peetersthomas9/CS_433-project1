"""
Python file to extract the data and do the preprocessing. 

"""


import numpy as np
import matplotlib.pyplot as plt
from functions import *

def get_data0(train=True):
# Extract the data, take all the features and standardize them

    if train:
        importdata = np.zeros((250000, 30))
    else:
        importdata = np.zeros((568238, 30))
    for i in range(0, 30):
        if train: 


            importdata[:,i] = np.genfromtxt("data/train.csv", delimiter=",", names = True, dtype = None, usecols=[i+2])
        else:


            importdata[:,i] = np.genfromtxt("data/test.csv", delimiter=",", names = True, dtype = None, usecols=[i+2])
    if train :
        y = np.genfromtxt("data/train.csv",delimiter=",", skip_header=1, usecols=[1],
                          converters={1: lambda x: 0 if b"b" in x else 1})
    else:
        y = np.genfromtxt("data/test.csv",delimiter=",", skip_header=1, usecols=[1],
                          converters={1: lambda x: 0 if b"b" in x else 1})
      


    datatest = np.copy(importdata)
    datatest[datatest == -999] = np.nan

    for i in range(0,30):
        datatest[datatest[:,i]>np.nanquantile(datatest[~np.isnan(datatest[:,i]),i],0.95), i] = np.nanquantile(datatest[~np.isnan(datatest[:,i]),i],0.95)
        datatest[datatest[:,i]<np.nanquantile(datatest[~np.isnan(datatest[:,i]),i],0.05), i] = np.nanquantile(datatest[~np.isnan(datatest[:,i]),i],0.05)

    for i in range(0,30):
        datatest[np.isnan(datatest[:,i]),i] = np.mean(datatest[~np.isnan(datatest[:,i]),i]) # take average value instead 

    #standardize the datas (mean=0 and std=1): 
    data_std=standardize(datatest)

    #add a column of 1 in front for w0:
    data_final=np.c_[np.ones(len(y)), data_std]

    return data_final , y

def get_data1(train=True):
# extract the data, standardize them and take only 18 features instead of 30

    if train:
        importdata = np.zeros((250000, 30))
    else:
        importdata = np.zeros((568238, 30))
    for i in range(0, 30):
        if train: 


            importdata[:,i] = np.genfromtxt("data/train.csv", delimiter=",", names = True, dtype = None, usecols=[i+2])
        else:


            importdata[:,i] = np.genfromtxt("data/test.csv", delimiter=",", names = True, dtype = None, usecols=[i+2])
    if train :
        y = np.genfromtxt("data/train.csv",delimiter=",", skip_header=1, usecols=[1],
                          converters={1: lambda x: 0 if b"b" in x else 1})
    else:
        y = np.genfromtxt("data/test.csv",delimiter=",", skip_header=1, usecols=[1],
                          converters={1: lambda x: 0 if b"b" in x else 1})
      
    d = np.copy(importdata)
    datatest=np.c_[d[:,0],d[:,1],d[:,2],d[:,4],d[:,5],d[:,7],d[:,8],d[:,9],d[:,11],d[:,12],d[:,14],d[:,17],d[:,21],d[:,22],d[:,23],d[:,24],d[:,27],d[:,29]]
    datatest[datatest == -999] = np.nan

    for i in range(0,datatest.shape[1]):
        datatest[datatest[:,i]>np.nanquantile(datatest[~np.isnan(datatest[:,i]),i],0.97), i] = np.nanquantile(datatest[~np.isnan(datatest[:,i]),i],0.97)
        datatest[datatest[:,i]<np.nanquantile(datatest[~np.isnan(datatest[:,i]),i],0.03), i] = np.nanquantile(datatest[~np.isnan(datatest[:,i]),i],0.03)

    for i in range(0,datatest.shape[1]):
        datatest[np.isnan(datatest[:,i]),i] = np.mean(datatest[~np.isnan(datatest[:,i]),i]) # take average value instead 

    #standardize the datas (mean=0 and std=1): 
    data_std=standardize(datatest)

    #add a column of 1 in front for w0:
    data_final=np.c_[np.ones(len(y)), data_std]
    return data_final , y



def get_data2(train=True):

    if train:
        importdata = np.zeros((250000, 30))
    else:
        importdata = np.zeros((568238, 30))
    for i in range(0, 30):

        if train: 

            importdata[:,i] = np.genfromtxt("data/train.csv", delimiter=",", names = True, dtype = None, usecols=[i+2])
        else:

            importdata[:,i] = np.genfromtxt("data/test.csv", delimiter=",", names = True, dtype = None, usecols=[i+2])
    if train :
        y = np.genfromtxt("data/train.csv",delimiter=",", skip_header=1, usecols=[1],
                          converters={1: lambda x: 0 if b"b" in x else 1})
    else:
        y = np.genfromtxt("data/test.csv",delimiter=",", skip_header=1, usecols=[1],
                          converters={1: lambda x: 0 if b"b" in x else 1})

    #Check prevelence of outcome
    data = np.c_[np.ones(len(y)), importdata]
    
    tx = np.copy(data)
    tx[tx == -999] = np.nan

    ccindex =~np.isnan(tx[:,7])&~np.isnan(tx[:,1])
    ccase = tx[ccindex, :]  
    yccase = y[ccindex]


    ccase[:,2] = np.where(ccase[:,2] < 53, 1, 0)
    ccase[:,3] = np.where(np.absolute(ccase[:,3]-90) <30, 1, 0)
    ccase[:,4] = np.where(ccase[:,4] > 100, 1, 0)
    ccase[:,10] = np.where(ccase[:,10]>125, 1, 0)

    ccase[:,11] = np.where(ccase[:,11] <1.05, 1, 0)
    ccase[:,12] = np.where(ccase[:,12] >0.75, 1, 0)
    ccase[:,14] = np.where(ccase[:,14] > 35.8, 1, 0)
    ccase[:,22] = np.where(ccase[:,22] < 300, 1, 0)
    ccase[:,23] = np.where(ccase[:,23] > 1, 1, 0)
    ccase[:,30] = np.where(ccase[:,30] <200, 1, 0)
    newvar = -(ccase[:,18]**2)
    newvar1 = -(ccase[:,15]**2)
    ccase[:,1] =np.where(np.absolute(ccase[:,1] -135)> 20, 1, 0)
    var5 =np.where(ccase[:,5] > 2, 1, 0)
    var5[ccase[:,5] > 5] =2
    var6 =np.where(ccase[:,6] > 400, 1, 0)
    var6[ccase[:,6] <175] = 1
    var6[ccase[:,6] >800] = 2

    ccase[:,7] =np.where(ccase[:,7] < -2, 1, 0)
    ccase[:,8] =np.where(np.absolute(ccase[:,8]-2)<0.75, 1, 0)
    ccase[:,9] =np.where(ccase[:,9] < 20, 1, 0)

    ccase[:,13] =np.where(ccase[:,13] > .6, 1, 0)
    ccase[:,25] =np.where(np.absolute(ccase[:,25])>2, 1, 0)
    ccase[:,28] =np.where(np.absolute(ccase[:,28])>2, 1, 0)

    #, newvar , newvar1
    datacc = np.c_[ccase[:,0], newvar , newvar1,ccase[:,1],var5, var6, ccase[:,7],ccase[:,8],ccase[:,9], ccase[:,13],ccase[:,25],ccase[:,28],ccase[:,2],ccase[:,3],ccase[:,4],ccase[:,10],ccase[:,11],ccase[:,12], ccase[:,14], ccase[:,15], ccase[:,18], ccase[:,22], ccase[:,30]]
    
    tx = np.copy(data)
    tx[tx == -999] = np.nan

    tx[:,2] = np.where(tx[:,2] < 60, 1, 0)
    tx[:,3] = np.where(np.absolute(tx[:,3]-90) <20, 1, 0)
    tx[:,4] = np.where(tx[:,4] > 40, 1, 0)
    tx[:,10] = np.where(tx[:,10]>125, 1, 0)
    tx[:,11] = np.where(tx[:,11] <1.05, 1, 0)
    tx[:,12] = np.where(tx[:,12] >0, 1, 0)
    tx[:,14] = np.where(tx[:,14] > 35.8, 1, 0)
    tx[:,22] = np.where(tx[:,22] > 150, 1, 0)
    tx[:,23] = np.where(tx[:,23] > 1, 1, 0)
    tx[:,30] = np.where(tx[:,30] > 1, 1, 0)
    newvar = -(tx[:,18]**2)
    newvar1 = -(tx[:,15]**2)

    na1 = np.where(~np.isnan(tx[:,1]), 1, 0)
    na5 = np.where(~np.isnan(tx[:,5]), 1, 0)
    na6 = np.where(~np.isnan(tx[:,6]), 1, 0)
    na7 = np.where(~np.isnan(tx[:,7]), 1, 0)
    na13 = np.where(~np.isnan(tx[:,13]) , 1, 0)
    na25 = np.where(~np.isnan(tx[:,25]) , 1, 0)
    na28 = np.where(~np.isnan(tx[:,28]) , 1, 0)

    tx[:,1] = np.where(~np.isnan(tx[:,1]), tx[:,1], 0)
    tx[:,5] = np.where(~np.isnan(tx[:,5]), tx[:,5], 0)
    tx[:,6] = np.where(~np.isnan(tx[:,6]), tx[:,6], 0)
    tx[:,7] = np.where(~np.isnan(tx[:,7]), tx[:,7], 0)
    tx[:,13] = np.where(~np.isnan(tx[:,13]) , tx[:,13], 0)
    tx[:,25] = np.where(~np.isnan(tx[:,25]) , tx[:,25], 0)
    tx[:,28] = np.where(~np.isnan(tx[:,28]) , tx[:,28], 0)

    i1 =np.where(np.absolute(tx[:,1] -130)> 20, 1, 0)
    i5 =np.where(tx[:,5] > 2.5, 1, 0)
    i6 =np.where(tx[:,6] > 400, 1, 0)
    i6[tx[:,6] <175] = 1
    i7 =np.where(tx[:,7] < 0, 1, 0)
    i13 =np.where(tx[:,13] > .6, 1, 0)
    i25 =np.where(np.absolute(tx[:,25])>2, 1, 0)
    i28 =np.where(np.absolute(tx[:,28])>2, 1, 0)

    ind1 = i1*na1
    ind5 = i5*na5
    ind6 = i6*na6
    ind7 = i7*na7
    ind13 = i13*na13
    ind25 = i25*na25
    ind28 = i28*na28

    dat = np.c_[tx[:,0],tx[:,2],tx[:,3],tx[:,4],tx[:,10],tx[:,11],tx[:,12], tx[:,14], tx[:,15], tx[:,18], tx[:,22], tx[:,23], tx[:,30], newvar, newvar1, na5,na6,na7,na13, ind5, ind6, ind7, ind13, na1, ind1, na25, ind25, na28, ind28]
    print(dat.shape)
    
    return datacc, yccase, dat, y, ccindex

def get_data3(train=True):
    # Extract the data and separate them in four strata for theJET number with independent variables modeled 
    
    if train:
        importdata = np.zeros((250000, 30))
    else:
        importdata = np.zeros((568238, 30))

    for i in range(0, 30):
        if train: 

            importdata[:,i] = np.genfromtxt("data/train.csv", delimiter=",", names = True, dtype = None, usecols=[i+2])
        else:

            importdata[:,i] = np.genfromtxt("data/test.csv", delimiter=",", names = True, dtype = None, usecols=[i+2])
    if train :
        y = np.genfromtxt("data/train.csv",delimiter=",", skip_header=1, usecols=[1],
                          converters={1: lambda x: 0 if b"b" in x else 1})
    else:
        y = np.genfromtxt("data/test.csv",delimiter=",", skip_header=1, usecols=[1],
                          converters={1: lambda x: 0 if b"b" in x else 1})
    datatrain = np.copy(importdata)
    datatrain[datatrain == -999] = np.nan



    index230 = (datatrain[:,22]==0)
    index231 =datatrain[:,22]==1

    index232 =datatrain[:,22]==2
    index233 =datatrain[:,22]==3

    dat0 = datatrain[index230, :]
    y0 = y[index230]

    dat1 = datatrain[index231, :]
    y1 = y[index231]
    dat2 = datatrain[index232, :]
    y2 = y[index232]
    dat3 = datatrain[index233, :]
    y3 = y[index233]
    dat0 = np.c_[np.ones(len(y0)), dat0]  
    dat0[:,1] =np.where(np.absolute(dat0[:,1] -140)<15, 1, 0)
    dat0[:,2] = np.where(dat0[:,2] < 40, 1, 0)
    dat0[:,3] = np.where(np.absolute(dat0[:,3]-90) <20, 1, 0)
    dat0[:,4] = np.where(dat0[:,4] >200, 1, 0)
    dat0[:,8] =np.where(np.absolute(dat0[:,8] -3)<0.2, 1, 0)

    dat0[:,10] = np.where(dat0[:,10] >90, 1, 0)
    dat0[:,11] = np.where(dat0[:,11] <.5, 1, 0)
    dat0[:,12] = np.where(dat0[:,12] >0.75, 1, 0)
    dat0[:,14] = np.where(dat0[:,14] >40, 1, 0)

    dat0[:,15] =np.where(np.absolute(dat0[:,15])<1, 1, 0)
    dat0[:,17] = np.where(dat0[:,17] <40, 1, 0)
    dat0[:,18] =np.where(np.absolute(dat0[:,18])<1.25, 1, 0)

    dat0[:,20] = np.where(dat0[:,20] <20, 1, 0)

    dat0[:,22]=np.where(dat0[:,22] >175, 1, 0)
    datcc0 = np.c_[dat0[:,0],dat0[:,1] , dat0[:,2],dat0[:,3],dat0[:,4],dat0[:,8],dat0[:,10],dat0[:,11],dat0[:,12],dat0[:,14],dat0[:,15],dat0[:,17],dat0[:,18],dat0[:,20],dat0[:,22]]
    dat1 = np.c_[np.ones(len(y1)), dat1]

    dat1[:,1] =np.where(np.absolute(dat1[:,1] -140)<15, 1, 0)

    dat1[:,2] = np.where(dat1[:,2] < 40, 1, 0)
    dat1[:,3] = np.where(np.absolute(dat1[:,3]-90) <20, 1, 0)
    dat1[:,4] = np.where(dat1[:,4] >200, 1, 0)
    dat1[:,8] =np.where(np.absolute(dat1[:,8] -2.5)<0.5, 1, 0)

    dat1[:,9] = np.where(np.absolute(dat1[:,9] -30)<10, 1, 0)
    dat1[:,10] = np.where(dat1[:,10] <120, 1, 0)
    dat1[:,11] = np.where(dat1[:,11] <.75, 1, 0)
    dat1[:,12] = np.where(dat1[:,12] >0.75, 1, 0)
    dat1[:,14] = np.where(dat1[:,14] >40, 1, 0)

    dat1[:,15] =np.where(np.absolute(dat1[:,15])<1, 1, 0)
    dat1[:,16] =np.where(np.absolute(dat1[:,16]-1)<1, 1, 0)

    dat1[:,17] = np.where(dat1[:,17] <60, 1, 0)

    dat1[:,18] =np.where(np.absolute(dat1[:,18])<1.25, 1, 0)
    dat1[:,20] = np.where(dat1[:,20] >100, 1, 0)
    dat1[:,22] =np.where(dat1[:,22] >250, 1, 0)
    dat1[:,24] =np.where(dat1[:,24] >80, 1, 0)
    dat1[:,25] =np.where(np.absolute(dat1[:,25])>2, 1, 0)
    dat1[:,30] = np.where(dat1[:,30] <250, 1, 0)
    datcc1 = np.c_[dat1[:,0],dat1[:,1],dat1[:,2],dat1[:,3],dat1[:,4],dat1[:,8],dat1[:,9],dat1[:,10],dat1[:,11],dat1[:,12], dat1[:,14],dat1[:,15],dat1[:,16],dat1[:,17],dat1[:,18],dat1[:,20],dat1[:,22],dat1[:,24],dat1[:,25],dat1[:,30]]


    dat2 = np.c_[np.ones(len(y2)), dat2]
    dat2[:,1] =np.where(np.absolute(dat2[:,1] -140)> 30, 1, 0)
    dat2[:,2] = np.where(dat2[:,2] < 40, 1, 0)
    dat2[:,3] = np.where(np.absolute(dat2[:,3]-90) <20, 1, 0)
    dat2[:,4] = np.where(dat2[:,4] >110, 1, 0)
    dat2[:,5] = np.where(dat2[:,5] >3, 1, 0)
    dat2[:,6] = np.where(dat2[:,6] >400, 1, 0)
    dat2[:,7] = np.where(dat2[:,7] <-2, 1, 0)
    dat2[:,8] = np.where(np.absolute(dat2[:,8] -2)<0.5, 1, 0)
    dat2[:,9] = np.where(dat2[:,9] <10, 1, 0)
    dat2[:,11] = np.where(dat2[:,11] <1, 1, 0)
    dat2[:,12] = np.where(dat2[:,12] >0.75, 1, 0)
    dat2[:,13] = np.where(dat2[:,13] >0.75, 1, 0)

    dat2[:,14] = np.where(dat2[:,14] >60, 1, 0)
    dat2[:,15] =np.where(np.absolute(dat2[:,15])<1, 1, 0)
    dat2[:,16] =np.where(np.absolute(dat2[:,16]+1)<1, 1, 0)

    dat2[:,17] = np.where(dat2[:,17] <70, 1, 0)

    dat2[:,18] =np.where(np.absolute(dat2[:,18])<1.25, 1, 0)
    dat2[:,19] =np.where(np.absolute(dat2[:,19])<1, 1, 0)

    dat2[:,20] = np.where(dat2[:,20] >150, 1, 0)
    dat2[:,22] =np.where(dat2[:,22] >400, 1, 0)
    dat2[:,24] =np.where(dat2[:,24] >80, 1, 0)

    dat2[:,25] =np.where(np.absolute(dat2[:,25])>2, 1, 0)
    dat2[:,26] = np.where(np.absolute(dat2[:,26] -.5)>1.75, 1, 0)
    dat2[:,27] = np.where(dat2[:,27] >50, 1, 0)
    dat2[:,28] =np.where(np.absolute(dat2[:,28])>2, 1, 0)
    dat2[:,30] = np.where(dat2[:,30] <250, 1, 0)


    datcc2 = np.c_[dat2[:,0],dat2[:,1],dat2[:,2],dat2[:,3],dat2[:,4], dat2[:,5], dat2[:,6],dat2[:,7],dat2[:,8],dat2[:,9],dat2[:,11],dat2[:,12],dat2[:,13],dat2[:,14],dat2[:,15],dat2[:,17],dat2[:,18],dat2[:,20],dat2[:,22],dat2[:,24],dat2[:,25],dat2[:,26], dat2[:,27],dat2[:,28],dat2[:,30]]
    
    dat3 = np.c_[np.ones(len(y3)), dat3]

    dat3[:,1] =np.where(np.absolute(dat3[:,1] -140)> 20, 1, 0)
    dat3[:,2] = np.where(dat3[:,2] < 40, 1, 0)
    dat3[:,3] = np.where(np.absolute(dat3[:,3]-90) <10, 1, 0)
    dat3[:,4] = np.where(dat3[:,4] >200, 1, 0)
    dat3[:,5] = np.where(dat3[:,5] > 2.5, 1, 0)
    dat3[:,6] = np.where(dat3[:,6] >800, 1, 0)
    dat3[:,7] = np.where(dat3[:,7] < 0, 1, 0)
    dat3[:,8] = np.where(np.absolute(dat3[:,8]-2)<0.75, 1, 0)
    dat3[:,9] = np.where(dat3[:,9] <50, 1, 0)

    dat3[:,11] = np.where(dat3[:,11] <.75, 1, 0)
    dat3[:,12] = np.where(dat3[:,12] >0.75, 1, 0)
    dat3[:,13] = np.where(dat3[:,13] >0.7, 1, 0)

    dat3[:,14] = np.where(dat3[:,14] >60, 1, 0)

    dat3[:,15] =np.where(np.absolute(dat3[:,15])<1, 1, 0)
    dat2[:,16] =np.where(np.absolute(dat2[:,16])<1, 1, 0)

    dat3[:,17] = np.where(dat3[:,17] <50, 1, 0)
    dat3[:,18] =np.where(np.absolute(dat3[:,18])<1.25, 1, 0)
    dat3[:,19] = np.where(dat3[:,19] >-0.5, 1, 0)
    dat3[:,20] = np.where(dat3[:,20] >100, 1, 0)
    dat3[:,22] =np.where(dat3[:,22] < 300, 1, 0)

    dat3[:,24] =np.where(np.absolute(dat3[:,24]-135)<35, 1, 0)
    dat3[:,25] =np.where(np.absolute(dat3[:,25])>2, 1, 0)
    dat3[:,26] = np.where(dat3[:,26] <-2, 1, 0)
    dat3[:,27] = np.where(dat3[:,27] <80, 1, 0)
    dat3[:,28] =np.where(np.absolute(dat3[:,28])>2, 1, 0)
    dat3[:,30] = np.where(dat3[:,30] <250, 1, 0)

    datcc3 = np.c_[dat3[:,0],dat3[:,1],dat3[:,2],dat3[:,3],dat3[:,4],dat3[:,5], dat3[:,6],dat3[:,7],dat3[:,8],dat3[:,9],dat3[:,11],dat3[:,12],dat3[:,13],dat3[:,14],dat3[:,15],dat3[:,16],dat3[:,17],dat3[:,18],dat3[:,19],dat3[:,20],dat3[:,22] ,dat3[:,24],dat3[:,25],dat3[:,26],dat3[:,28],dat3[:,30]]


    return datcc0, datcc1, datcc2, datcc3, y0, y1, y2, y3, index230, index231, index232, index233

def get_data4(train=True):
# Extract the data, take all the features and standardize them

    if train:
        importdata = np.zeros((250000, 30))
    else:
        importdata = np.zeros((568238, 30))
    for i in range(0, 30):
        if train: 


            importdata[:,i] = np.genfromtxt("data/train.csv", delimiter=",", names = True, dtype = None, usecols=[i+2])
        else:


            importdata[:,i] = np.genfromtxt("data/test.csv", delimiter=",", names = True, dtype = None, usecols=[i+2])
    if train :
        y = np.genfromtxt("data/train.csv",delimiter=",", skip_header=1, usecols=[1],
                          converters={1: lambda x: 0 if b"b" in x else 1})
    else:
        y = np.genfromtxt("data/test.csv",delimiter=",", skip_header=1, usecols=[1],
                          converters={1: lambda x: 0 if b"b" in x else 1})
      


    data = np.copy(importdata)
    data[data == -999] = np.nan

    for i in range(0,30):
        data[data[:,i]>np.nanquantile(data[~np.isnan(data[:,i]),i],0.95), i] = np.nanquantile(data[~np.isnan(data[:,i]),i],0.95)
        data[data[:,i]<np.nanquantile(data[~np.isnan(data[:,i]),i],0.05), i] = np.nanquantile(data[~np.isnan(data[:,i]),i],0.05)

    for i in range(0,30):
        data[np.isnan(data[:,i]),i] = np.mean(data[~np.isnan(data[:,i]),i]) # take average value instead 

    degree=5

    data = standardize(data)
    data_poly=build_poly(data,degree)
    data_poly=np.c_[data_poly, np.sin(data[:,1:]), np.cos(data[:,1:]), np.power(np.sin(data[:,1:]),2), np.power(np.cos(data[:,1:]),2), np.power(np.sin(data[:,1:]),3), np.power(np.cos(data[:,1:]),3), np.power(np.sin(data[:,1:]),4), np.power(np.cos(data[:,1:]),4)]
    data_poly_std=standardize(data_poly[:,1:])
    data_poly_std=np.c_[data_poly[:,0],data_poly_std]
    return data_poly_std , y

def get_data5(train=True):
    # Extract the data and separate them in four strata for the JET number with independent variables modeled and use polynomial and trigonometric feature expensions
    degree=5
    if train:
        importdata = np.zeros((250000, 30))
    else:
        importdata = np.zeros((568238, 30))

    for i in range(0, 30):
        if train: 

            importdata[:,i] = np.genfromtxt("data/train.csv", delimiter=",", names = True, dtype = None, usecols=[i+2])
        else:

            importdata[:,i] = np.genfromtxt("data/test.csv", delimiter=",", names = True, dtype = None, usecols=[i+2])
    if train :
        y = np.genfromtxt("data/train.csv",delimiter=",", skip_header=1, usecols=[1],
                          converters={1: lambda x: 0 if b"b" in x else 1})
    else:
        y = np.genfromtxt("data/test.csv",delimiter=",", skip_header=1, usecols=[1],
                          converters={1: lambda x: 0 if b"b" in x else 1})
    datatrain = np.copy(importdata)
    datatrain[datatrain == -999] = np.nan

    # remove the outlier 
    for i in range(0,30):

        datatrain[datatrain[:,i]>np.nanquantile(datatrain[~np.isnan(datatrain[:,i]),i],0.95), i] = np.nanquantile(datatrain[~np.isnan(datatrain[:,i]),i],0.95)
        datatrain[datatrain[:,i]<np.nanquantile(datatrain[~np.isnan(datatrain[:,i]),i],0.05), i] = np.nanquantile(datatrain[~np.isnan(datatrain[:,i]),i],0.05)
   
    # replace the nan value with the median or the mean of the features values
    for i in range(0,30):
        datatrain[np.isnan(datatrain[:,i]),i] = np.mean(datatrain[~np.isnan(datatrain[:,i]),i]) # take average value instead
        #datatrain[np.isnan(datatrain[:,i]),i] = np.median(datatrain[~np.isnan(datatrain[:,i]),i])

        index230 = (datatrain[:,22]==0)
        index231 =(datatrain[:,22]==1)

        index232 =(datatrain[:,22]==2)
        index233 =(datatrain[:,22]==3)

        degree=5

        # get data with jet number = 0
        dat0 = datatrain[index230, :]
        # take only a part of the features 
        datcc0 =np.c_[dat0[:,0],dat0[:,1],dat0[:,2],dat0[:,3],dat0[:,7],dat0[:,8],dat0[:,9],dat0[:,10],dat0[:,11],dat0[:,13],dat0[:,14],dat0[:,15],dat0[:,16],dat0[:,17],dat0[:,18],dat0[:,19],dat0[:,20],dat0[:,21]]
        # standardize 
        datcc0 = standardize(datcc0)
        # extend with polynomial features 
        datcc0_poly=build_poly(datcc0,degree)
        # extend with trigonometric features
        datcc0_poly=np.c_[datcc0_poly, np.sin(datcc0[:,1:]), np.cos(datcc0[:,1:]), np.power(np.sin(datcc0[:,1:]),2), np.power(np.cos(datcc0[:,1:]),2), np.power(np.sin(datcc0[:,1:]),3), np.power(np.cos(datcc0[:,1:]),3), np.power(np.sin(datcc0[:,1:]),4), np.power(np.cos(datcc0[:,1:]),4)]
        # standardize again
        datcc0_poly_std=standardize(datcc0_poly[:,1:])
        # add a line of one in front 
        datcc0_poly_std=np.c_[datcc0_poly[:,0],datcc0_poly_std]

        # get data with jet number = 1
        dat1 = datatrain[index231, :]
        # take only a part of the features 
        datcc1 =np.c_[dat1[:,0],dat1[:,1],dat1[:,2],dat1[:,3],dat1[:,7],dat1[:,8],dat1[:,9],dat1[:,10],dat1[:,11],dat1[:,13],dat1[:,14],dat1[:,15],dat1[:,16],dat1[:,17],dat1[:,18],dat1[:,19],dat1[:,20],dat1[:,21],dat1[:,23],dat1[:,24],dat1[:,25],dat1[:,29]]
        # standardize 
        datcc1 = standardize(datcc1)
        # extend with polynomial feature 
        datcc1_poly=build_poly(datcc1,degree)
        # extend with trigonomial feature
        datcc1_poly=np.c_[datcc1_poly, np.sin(datcc1[:,1:]), np.cos(datcc1[:,1:]), np.power(np.sin(datcc1[:,1:]),2), np.power(np.cos(datcc1[:,1:]),2), np.power(np.sin(datcc1[:,1:]),3), np.power(np.cos(datcc1[:,1:]),3), np.power(np.sin(datcc1[:,1:]),4), np.power(np.cos(datcc1[:,1:]),4)]
        datcc1_poly_std=standardize(datcc1_poly[:,1:])
        datcc1_poly_std=np.c_[datcc1_poly[:,0],datcc1_poly_std]

        # get data with jet number = 2
        dat2 = datatrain[index232, :]
        # remove the column 22 corresponding to jet number
        datcc2 =np.delete(dat2, 22, 1)
        # standardize
        datcc2 = standardize(datcc2)
        # extend with polynomial feature
        datcc2_poly=build_poly(datcc2,degree)
        # extend with trigonomial feature
        datcc2_poly=np.c_[datcc2_poly, np.sin(datcc2[:,1:]), np.cos(datcc2[:,1:]), np.power(np.sin(datcc2[:,1:]),2), np.power(np.cos(datcc2[:,1:]),2), np.power(np.sin(datcc2[:,1:]),3), np.power(np.cos(datcc2[:,1:]),3), np.power(np.sin(datcc2[:,1:]),4), np.power(np.cos(datcc2[:,1:]),4)]
        datcc2_poly_std=standardize(datcc2_poly[:,1:])
        datcc2_poly_std=np.c_[datcc2_poly[:,0],datcc2_poly_std]

        # get data with jet number = 3
        dat3 = datatrain[index233, :]
        # remove the column 22 corresponding to jet number 
        datcc3 =np.delete(dat3, 22, 1)
        # standardize
        datcc3 = standardize(datcc3)
        # build polynomial 
        datcc3_poly=build_poly(datcc3,degree)
        # add trigo expension 
        datcc3_poly=np.c_[datcc3_poly, np.sin(datcc3[:,1:]), np.cos(datcc3[:,1:]), np.power(np.sin(datcc3[:,1:]),2), np.power(np.cos(datcc3[:,1:]),2), np.power(np.sin(datcc3[:,1:]),3), np.power(np.cos(datcc3[:,1:]),3), np.power(np.sin(datcc3[:,1:]),4), np.power(np.cos(datcc3[:,1:]),4)]
        datcc3_poly_std=standardize(datcc3_poly[:,1:])
        datcc3_poly_std=np.c_[datcc3_poly[:,0],datcc3_poly_std]

        y0 = y[index230]
        y1 = y[index231]
        y2 = y[index232]
        y3 = y[index233]

    return datcc0_poly_std, datcc1_poly_std, datcc2_poly_std, datcc3_poly_std, y0, y1, y2, y3, index230, index231, index232, index233


