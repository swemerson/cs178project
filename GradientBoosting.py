from sklearn import ensemble


def predictSoft(Xtr, Ytr, Xte, n_estimators, max_depth, min_samples_split, learning_rate):
    params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_split': min_samples_split,
              'learning_rate': learning_rate}
    model = ensemble.GradientBoostingRegressor(**params)
    model.fit(Xtr, Ytr)
    return model.predict(Xte)