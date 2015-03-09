import numpy as np                  #basic linear algebra
import matplotlib.pyplot as plt     #plotting functions
import csv
import datetime
import sklearn.linear_model as sklin
import sklearn.metrics as skmet
import sklearn.cross_validation as skcv
import sklearn.grid_search as skgs
from sklearn.svm import SVR

from numpy.polynomial.polynomial import Polynomial, polyval


def logscore(gtruth,gpred):
    gpred = np.clip(gpred,0,np.inf)
    logdif = np.log(1 + gtruth) - np.log(1 + gpred)
    return np.sqrt(np.mean(np.square(logdif)))


def score(gtruth, gpred):
    gpred = np.clip(gpred, 0, np.inf)
    return np.sqrt(np.mean(np.square(gtruth - gpred)))


def monomials(x, d):
    y = []
    if len(x) == 0:
        return []
    if d == 0:
        return [1]
    elif d == 1:
        return x
    else:
        for i in range(d+1):
            for m in monomials(x[1:], d-i):
                y.append(x[0]**i*m)
        return y


def poly_nd(x, d):
    y = []
    for i in range(d+1):
        y.extend(monomials(x, d))
    return y


def poly(x, d):
    y = []
    for i in range(d):
        y.append(np.power(x, i+1))
    return y


def fourier(x, d, r):
    y = []
    w = (np.pi*2)/r
    for i in range(d):
        y.append(np.sin((i+1)*x*w))
        y.append(np.cos((i+1)*x*w))
    return y


def fourier_md(x, d, r): # TODO
    y = []
    for i in range(d):
        s = 0
        c = 0
        for j in range(len(x)):
            w = (np.pi*2)/r[j]
            s += np.sin((i+1)*x[j]*w)
            c += np.cos((i+1)*x[j]*w)
        y.append(s)
        y.append(c)
    return y


def indicators(vals, x):
    y = []
    for val in vals:
        if x == val:
            y.append(1)
        else:
            y.append(0)
    return y


def read_path(inpath):
    X = []
    with open(inpath,'r') as fin:
        reader = csv.reader(fin, delimiter=',')
        for row in reader:
            t = datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
            x = [t]
            x.extend(row[1:])
            X.append(x)
    return X


def read_features(X, features_fn):
    M = []
    for x in X:
        m = features_fn(x)
        M.append(m)
    return np.matrix(M)


def time_fourier(x):
    y = [1]
    y.extend(poly(float(x[0].year), 1))
    y.extend(fourier(float(x[0].isoweekday()), 4, 7))
    y.extend(fourier(float(x[0].month),        4, 12))
    y.extend(fourier(float(x[0].hour),         8, 24))
#   y.extend(indicators(range(24), x[0].hour))
#   y.extend(fourier(float(x[0].minute),       8, 60))
    return y


def time_dct(x): # Discrete cosine transform over multiple dimensions
    y = []
    t = x[0]
    L1 = 7
    L2 = 12
    L3 = 24
    L4 = 60
    for i in range(4):
        for j in range(4):
            for k in range(8):
                for m in range(8):
                    y.append(np.cos(np.pi/L1*(i+0.5)*float(t.isoweekday()))*
                             np.cos(np.pi/L2*(j+0.5)*float(t.month))*
                             np.cos(np.pi/L3*(k+0.5)*float(t.hour))*
                             np.cos(np.pi/L4*(m+0.5)*float(t.minute)))
    return y


def month_w1356_poly(x):
    y = []
    m = float(x[0].month) + float(x[0].day)/30
    w1 = float(x[1])
    w3 = float(x[3])
    w5 = float(x[5])
    w6 = float(x[6])
    y.extend(poly_nd([(m-7.007)/3.451, (w1-0.5)/0.2341, (w3-0.4773)/0.207,
                      (w5-0.1966)/0.1399, (w6-0.6291)/0.233], 2))
    return y


def w56_poly(x):
    w5 = float(x[5])
    w6 = float(x[6])
    return poly_nd([(w5-0.1966)/0.1399, (w6-0.6291)/0.233], 4)


def w2_ind(x):
    return indicators(range(3), x[2])


def w4_linear(x):
    return [float(x[4])]


def w4_fourier(x):
    return fourier(float(x[4]), 8, 11)


def ortho(fns, x):
    y = []
    for fn in fns:
        y.extend(fn(x))
    return y


def linear_regression(Xtrain, Ytrain):
    regressor = sklin.LinearRegression()
    regressor.fit(Xtrain, Ytrain)
    return regressor


def ridge_regression(Xtrain,Ytrain):
    ridge_regressor = sklin.Ridge(fit_intercept=False, normalize=False)
    param_grid = {'alpha' : np.linspace(0, 5, 10)}
    n_scorefun = skmet.make_scorer(lambda x, y: -score(x,y)) # logscore is always maximizing... but we want the minium
    grid_search = skgs.GridSearchCV(ridge_regressor, param_grid, scoring = n_scorefun, cv = 10)
    grid_search.fit(Xtrain, Ytrain)
    print 'grid_search.best_estimator_: ', grid_search.best_estimator_
    return grid_search.best_estimator_


def regress(feature_fn):
    Xo = read_path('project_data/train.csv')
    Yo = np.genfromtxt('project_data/train_y.csv', delimiter = ',')
    Y = np.log(1 + Yo)
    Xvalo = read_path('project_data/validate.csv')
    Xval = read_features(Xvalo, feature_fn)

    # always split training and test data!
    Xtraino, Xtesto, Ytrain, Ytest = skcv.train_test_split(Xo, Y, train_size = 0.8)

    X = read_features(Xo, feature_fn)
    Xtrain = read_features(Xtraino, feature_fn)
    Xtest = read_features(Xtesto, feature_fn)

    lin = linear_regression(Xtrain, Ytrain)
    print 'regressor.coef_: ', lin.coef_
    print 'regressor.intercept_: ', lin.intercept_
    scorefunction = skmet.make_scorer(score)
    scores = skcv.cross_val_score(lin, X, Y, scoring = scorefunction, cv = 10)
    print 'mean : ', np.mean(scores), ' +/- ', np.std(scores)
    Ypredtrain = lin.predict(Xtrain)
    Ypredtest = lin.predict(Xtest)
    print 'lin score on Xtrain,Ytrain: ', score(Ytrain, Ypredtrain)
    print 'lin score on Xtest,Ytest: ', score(Ytest, Ypredtest)
    plot(np.matrix(Xtesto), Ypredtest, Ytest)

#   ridge = ridge_regression(Xtrain, Ytrain)
#   print 'score of ridge (train): ', score(Ytrain, ridge.predict(Xtrain))
#   print 'score of ridge (test): ', score(Ytest, ridge.predict(Xtest))

    Ypred = lin.predict(Xval)
    Ypred = np.exp(Ypred) - 1
    print Ypred
    np.savetxt('project_data/validate_y.txt', Ypred)
    return Ypred


def plot(Xo, Ypred, Ytruth):
    plt.figure(1)
    input, = plt.plot(Xo[:, 5], Ypred, 'bo', label='Ypred')
    predict, = plt.plot(Xo[:, 5], Ytruth, 'ro', label='Ytruth')
    plt.legend(handles=[input, predict])
    plt.savefig("plot.png")

if __name__ == "__main__":
    regress(lambda x: ortho([time_fourier, month_w1356_poly], x))
