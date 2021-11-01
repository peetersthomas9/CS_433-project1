In this repository you will find different files that will allow us to predict if the collision of 
particles produced a Bozon de Huggs particle. 


To make the prediction, you should run 'run.py'. You czn modify the hyper parameter and the choice of the model.

!!! You should add the folder data at the same level as run.py so that you can extract the training and testing sets. !!!


run.py : This is the main file to extract the data, compute our model and write the prediction in a 'test.csv' file. There are 11 different models: 6 basics (the one asked in part 1) and 5 advanced

get_data.py : In this file we put all the functions for the preprocessiong. In total we have 5 different methods for preprocessiong that we will use.

basic_models.py : In this file we have 6 functions that compute the prediction using 6 different approaches : 
		  Least Square, Linear Regression, Stochastic linear regression, Ridge Regression, logistic regression and regularized_logistic regression.
		  
Advanced Model : In this file we have 5 functions that compute the prediction using 5 different approaches with more advanced technique : 
		  Model A : Use independent variables modeled asbinary (0/1) + Gradient descent
		  Model B : Use independent variables modeled asbinary (0/1) + logistic gradient descent
		  Model C : Use four strata for the JET number with independent variables modeled as binary + logistic regression
		  Model D : Use polynomial and trigonometric expension on the initial features + logistic regression
		  Model E : Use four strata for the JET number and use polynomial and trigonometric expension on the features + logistic regression
		  Model F : Use four strata for the JET number and use polynomial and trigonometric expension on the features + regularized logistic regression

Implementation.py: In this file we added the 6 function asked in part 1

function.py : In this file we added all the other functions needed for the other files 

crossvalA.py : run crossvalidation on model A

crossvalBC.py : run crossvalidation on model B and C

crossvalE.IPYNB : run crossvalidation on model E

