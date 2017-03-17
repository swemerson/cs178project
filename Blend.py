import numpy as np
import mltools as ml
import matplotlib.pyplot as plt
import Dtree
import Knn
import Linear

from sklearn.metrics import mean_squared_error



################################################################################
###################   Settings   ###############################################
################################################################################

# General Settings
generateKaggleFile = 1

if (generateKaggleFile):
    sampleSize = 100000  # Don't change this
    split = 1  # Don't change this
else:
    sampleSize = 50000
    split = 0.75

# Using
usingDtree = 1
usingKnn = 1
usingLinear = 0

# Dtree Settings
maxDepth = 6
minLeaf = 128
nFeatures = 5
nTrees = 100

# Knn Settings
n_neighbors = 15

# Linear Settings
deg = 1



################################################################################
###################   Setup   ##################################################
################################################################################

# Read in and set up data
X = np.genfromtxt('X_train.txt', delimiter=None, max_rows=sampleSize)
Y = np.genfromtxt('Y_train.txt', delimiter=None, max_rows=sampleSize)
Xte = np.genfromtxt('X_test.txt', delimiter=None, max_rows=sampleSize)

# Split data for training and validation
Xtr, Xva, Ytr, Yva = ml.splitData(X, Y, split)


################################################################################
###################   Functions   ##############################################
################################################################################

# Get average soft prediction from all learners
def predictSoft(Xte):
    Yhat = [None] * len(Xte)

    if (usingDtree):
        Yhat = np.column_stack((Yhat, Dtree.predictSoft(Xtr, Ytr, Xte, maxDepth, minLeaf, nFeatures, nTrees)))

    if (usingKnn):
        Yhat = np.column_stack((Yhat, Knn.predictSoft(Xtr, Ytr, Xte, n_neighbors)))

    if (usingLinear):
        Yhat = np.column_stack((Yhat, Linear.predictSoft(Xtr, Ytr, Xte, deg)))

    return np.mean(Yhat[:, 1:], axis=1)


# Calculate MSE
def mse():
    mean = mean_squared_error(predictSoft(Xva), Yva)
    print 'Mean squared error: %.10f' % mean
    return mean

# Generate file for Kaggle submission
def toKaggle():
    Yhat = predictSoft(Xte)
    np.savetxt('Yhat_knn.txt', np.vstack((np.arange(len(Yhat)), Yhat)).T, '%d, %.2f', header='ID,Prob1', comments='', delimiter=',')



################################################################################
###################   Main   ###################################################
################################################################################

# Use split=1 and sampleSize=100000 to output Kaggle file
if generateKaggleFile:
    toKaggle()
else:
    global n_neighbors
    means = [0] * 25
    for i in range(30, 55):
        print i
        n_neighbors = i
        means[i-30] = mse()

    plt.plot(means)
    plt.show()

