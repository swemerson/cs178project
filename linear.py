from mltools import splitData
from numpy import genfromtxt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

# Settings
sampleSize = 1000
split = 0.75
degs = [1, 2, 3, 4, 5]

# Read in and set up data
X = genfromtxt('X_train.txt', delimiter=None, max_rows=sampleSize)
Y = genfromtxt('Y_train.txt', delimiter=None, max_rows=sampleSize)
Xt, Xv, Yt, Yv = splitData(X, Y, split)

# Create linear models for different polynomial degrees
for ct, deg in enumerate(degs):
    print 'Using degree = %d' % deg

    # Expand features and fit to model
    model = make_pipeline(PolynomialFeatures(deg), LinearRegression())
    model.fit(Xt, Yt)

    # Calculate Mean Squared Error on validation data
    print 'Mean squared error: %.2f' % mean_squared_error(model.predict(Xv), Yv)
    print