import numpy as np
import mltools as ml
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
    sampleSize = 10000
    split = 0.80

# Using
usingDtree = 1
usingKnn = 1
usingLinear = 1

# Dtree Settings
maxDepth = 4
minLeaf = 512
nFeatures = 5

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
        Yhat = np.column_stack((Yhat, Dtree.predictSoft(Xtr, Ytr, Xte, maxDepth, minLeaf, nFeatures)[:, 1]))

    if (usingKnn):
        Yhat = np.column_stack((Yhat, Knn.predictSoft(Xtr, Ytr, Xte, n_neighbors)[:, 1]))

    if (usingLinear):
        Yhat = np.column_stack((Yhat, Linear.predictSoft(Xtr, Ytr, Xte, deg)))

    return np.mean(Yhat[:, 1:], axis=1)


# Calculate MSE
def mse():
    print 'Mean squared error: %.2f' % mean_squared_error(predictSoft(Xva), Yva)


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
    mse()

