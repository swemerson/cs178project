from sklearn import svm


def predictSoft(Xtr, Ytr, Xte):
    model = svm.SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto', kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
    model.fit(Xtr, Ytr)
    return model.predict(Xte)