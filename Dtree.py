import numpy as np
import mltools as ml


def predictSoft(Xtr, Ytr, Xte, maxDepth, minLeaf, nFeatures, nTrees):

    # Set up storage for trees
    trees = [None] * nTrees

    # Make trees
    for i in range(nTrees):
        M = Xtr.shape[0]
        Xi, Yi = ml.bootstrapData(Xtr, Ytr, M)
        trees[i] = ml.dtree.treeClassify(Xi, Yi, maxDepth=maxDepth, minLeaf=minLeaf, nFeatures=nFeatures)

    predictXte = np.zeros((Xte.shape[0], nTrees))

    for i in range(nTrees):
        predictXte[:, i] = trees[i].predict(Xte)

    return np.mean(predictXte, axis=1)
