from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


def predictSoft(Xtr, Ytr, Xte, deg):

    model = make_pipeline(PolynomialFeatures(deg), LinearRegression())
    model.fit(Xtr, Ytr)

    return model.predict(Xte)
