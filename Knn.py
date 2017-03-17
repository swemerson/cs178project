from sklearn.neighbors import KNeighborsRegressor


def predictSoft(Xtr, Ytr, Xte, n_neighbors):

    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(Xtr, Ytr)

    return knn.predict(Xte)
