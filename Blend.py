import numpy as np
import mltools as ml
import matplotlib.pyplot as plt
import Dtree
import Knn
import linear
import GradientBoosting
import SVM

from sklearn.metrics import mean_squared_error



################################################################################
###################   Settings   ###############################################
################################################################################

# General Settings
generateKaggleFile = 0

if (generateKaggleFile):
    sampleSize = 100000  # Don't change this
    split = 1  # Don't change this
else:
    sampleSize = 10000
    split = 0.75

# Using
usingDtree = 0
usingKnn = 0
usingLinear = 0
usingGradientBoosting = 0
usingSVM = 1

# Dtree Settings
maxDepth = 19
minLeaf = 64
nFeatures = 12
nTrees = 150

# Knn Settings
n_neighbors = 39

# Linear Settings
deg = 1

# Gradient Boosting Settings
n_estimators = 1000
max_depth = 6
min_samples_split = 100
learning_rate = 0.01



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
        Yhat = np.column_stack((Yhat, linear.predictSoft(Xtr, Ytr, Xte, deg)))

    if (usingGradientBoosting):
        Yhat = np.column_stack((Yhat, GradientBoosting.predictSoft(Xtr, Ytr, Xte, n_estimators, max_depth, min_samples_split, learning_rate)))

    if (usingSVM):
        Yhat = np.column_stack((Yhat, SVM.predictSoft(Xtr, Ytr, Xte)))

    return np.mean(Yhat[:, 1:], axis=1)


# Calculate MSE
def mse():
    mean = mean_squared_error(predictSoft(Xva), Yva)
    print 'MSE: ' + str(mean)
    return mean

# Calculate AUC
def auc(soft, Y):
    """Manual AUC function for applying to soft prediction vectors"""
    indices = np.argsort(soft)  # sort data by score value
    Y = Y[indices]
    sorted_soft = soft[indices]

    # compute rank (averaged for ties) of sorted data
    dif = np.hstack(([True], np.diff(sorted_soft) != 0, [True]))
    r1 = np.argwhere(dif).flatten()
    r2 = r1[0:-1] + 0.5 * (r1[1:] - r1[0:-1]) + 0.5
    rnk = r2[np.cumsum(dif[:-1]) - 1]

    # number of true negatives and positives
    n0, n1 = sum(Y == 0), sum(Y == 1)

    # compute AUC using Mann-Whitney U statistic
    result = (np.sum(rnk[Y == 1]) - n1 * (n1 + 1.0) / 2.0) / n1 / n0
    print 'AUC: ' + str(result)
    return result

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
    #global nFeatures
    #aucs = [0] * 10
    #for i in range(5, 16):
    #    print i
    #    nFeatures = i
    #    aucs[i-5] = auc(predictSoft(Xva), Yva)

    #plt.plot(aucs)
    #plt.show()
    auc(predictSoft(Xva), Yva)
    #mse()
