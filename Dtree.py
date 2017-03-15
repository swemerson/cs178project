import mltools as ml


def predictSoft(Xtr, Ytr, Xte, maxDepth, minLeaf, nFeatures):

    M = Xtr.shape[0]

    Xi, Yi = ml.bootstrapData(Xtr, Ytr, M)
    forest = ml.dtree.treeClassify()
    forest.train(Xi, Yi, maxDepth=maxDepth, minLeaf=minLeaf, nFeatures=nFeatures)

    return forest.predictSoft(Xte)
