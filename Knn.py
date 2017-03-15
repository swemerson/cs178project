from sklearn.neighbors import KNeighborsClassifier


def predictSoft(Xtr, Ytr, Xte, n_neighbors):

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(Xtr, Ytr)

    return knn.predict_proba(Xte)
