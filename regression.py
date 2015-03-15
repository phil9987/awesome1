import numpy as np                  #basic linear algebra
import matplotlib.pyplot as plt     #plotting functions
import csv
import datetime
import sklearn.linear_model as sklin
import sklearn.ensemble as rf
import sklearn.metrics as skmet
import sklearn.cross_validation as skcv
import sklearn.grid_search as skgs
from sklearn import linear_model
import math

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors.dist_metrics import DistanceMetric

def logscore(gtruth, gpred):
    gtruth = np.clip(gtruth, 0, np.inf)
    gpred = np.clip(gpred, 0, np.inf)
    logdiff = np.log(1 + gtruth) - np.log(1 + gpred)
    return np.sqrt(np.mean(np.square(logdiff)))


def score(gtruth, gpred):
#   return logscore(gtruth, gpred)
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
        if int(x)== int(val):
            y.append(1)
        else:
            y.append(-1)
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
    x_rows =  len(X)
    i = 1
    for x in X:
        m = features_fn(x)
        M.append(m)
        if i % 1000 == 0:
            print str(i) + ' of ' + str(x_rows) + ' rows processed...'
        i = i+1
    return np.matrix(M)


def days_since(x):
    epoch = datetime.datetime(1970, 1, 1)
    return fourier(float((x[0] - epoch).days), 100, 365)


def time_parts(x):
    return [float(x[0].year), float(x[0].month), float(x[0].isoweekday()), float(x[0].day), float(x[0].hour)]

def time_fourier(x):
    y = [1]
    y.extend(poly(float(x[0].year), 2))
    y.extend(fourier(float(x[0].isoweekday()), 4, 7))
    y.extend(fourier(float(x[0].month),        4, 12))
    y.extend(fourier(float(x[0].day),          4, 30))
    y.extend(fourier(float(x[0].hour),         4, 24))
#   y.extend(indicators(range(24), x[0].hour))
#   y.extend(fourier(float(x[0].minute),       4, 60))
    return y


def time_dct(x): # Discrete cosine transform over multiple dimensions
    y = []
    t = x[0]
    L1 = 7
    L2 = 12
    L3 = 24
    for i in range(4):
        for j in range(4):
            for k in range(8):
                y.append(np.cos(np.pi/L1*(i+0.5)*float(t.isoweekday()))*
                         np.cos(np.pi/L2*(j+0.5)*float(t.month))*
                         np.cos(np.pi/L3*(k+0.5)*float(t.hour)))
    return y


def w_parts(x):
    return [float(x[1]), float(x[2]), float(x[3]),
            float(x[4]), float(x[5]), float(x[6])]


def month_w1356_poly(x):
    y = []
    m = float(x[0].month) + float(x[0].day)/30
    w1 = float(x[1])
    w3 = float(x[3])
    w5 = float(x[5])
    w6 = float(x[6])
    y.extend(poly_nd([w1, w3, w5, w6], 3))
#   y.extend(poly_nd([(m-7.007)/3.451,
#                     (w1-0.5)/0.2341, (w3-0.4773)/0.207,
#                     (w5-0.1966)/0.1399, (w6-0.6291)/0.233], 2))
    return y


def w_poly(n_w, d):
    return lambda x: poly(float(x[n_w]), d)


def w2_ind(x):
    #print indicators(range(3), x[2])
    return indicators(range(3), x[2])

def rushhour_ind(x):
    return indicators([7,8,17,18],int(x[2]))

def weekend_ind(x):
    return indicators([5,6],x[1])


def w4_linear(x):
    return [float(x[4])]


def w4_fourier(x):
    return fourier(float(x[4]), 8, 1)


# Assume that all values in x are ready-to-use features (i. e. no timestamps)
def simple_implementation(x):
    y = []
    # Make x[0], x[1], x[2]
    xf = [x[0].month, x[0].isoweekday(), x[0].hour]
    xf.extend(x[1:])
    xf = [float(i) for i in xf]
    y.extend(xf)
    base_feature_len = len(x)
    #print 'DEBUG: number base features: %d' %len(x)
    y.extend(month_w1356_poly(x))
    y.extend(w2_ind(xf))
    y.extend(poly_nd([float(xf[2])], 5))
    y.extend(rushhour_ind(xf))
    y.extend(weekend_ind(xf))
    for xk in xf:
        y.append(math.sqrt(float(xk)))
        y.append(math.log(float(xk) + 1, math.e))
        y.append(1/(float(xk) + 1))
    #generate all possible (unique) combination between the features.
    #num_y_cols = len(y)
    for idx in range(0, base_feature_len-1):
        for idx2 in range(idx, base_feature_len-1):
            y.append(float(y[idx]*y[idx2]))
        #for idx2 in range(idx,num_y_cols):
        #    y.append(float(yk*y[idx2]))
    return y


def ortho(fns, x):
    y = []
    for fn in fns:
        y.extend(fn(x))
    return np.array(y)


def linear_regression(Xtrain, Ytrain):
    regressor = sklin.LinearRegression()
    regressor.fit(Xtrain, Ytrain)
    print 'regressor.coef_: ', regressor.coef_
    print 'regressor.intercept_: ', regressor.intercept_
    return regressor


def nearest_neighbors_regression(Xtrain, Ytrain):
    param_grid = {'n_neighbors': np.linspace(2,7,6), 'weights': ['uniform', 'distance']}
    regressor = KNeighborsRegressor(algorithm='auto')
    regressor.fit(Xtrain,Ytrain)
    scorefun = skmet.make_scorer(lambda x, y: -score(x, y))
#   grid_search = skgs.GridSearchCV(regressor, param_grid, scoring = scorefun, cv = 5)
#   grid_search.fit(Xtrain, Ytrain)
#   print 'grid_search.best_estimator_: ', grid_search.best_estimator_
#   return grid_search.best_estimator_
    return regressor


def cheating_regression(Xtrain, Ytrain):
    regressor = rf.RandomForestRegressor(n_jobs=-1,verbose=1)
    #regressor.transform(Xtrain, threshold=None)
    regressor.fit(Xtrain, Ytrain)
    return regressor


def ridge_regression(Xtrain,Ytrain):
    ridge_regressor = sklin.Ridge(fit_intercept=False, normalize=False)
    param_grid = {'alpha' : np.linspace(0,10,10)}
    n_scorefun = skmet.make_scorer(lambda x, y: -score(x,y)) # logscore is always maximizing... but we want the minium
    grid_search = skgs.GridSearchCV(ridge_regressor, param_grid, scoring = n_scorefun, cv = 5)
    grid_search.fit(Xtrain, Ytrain)
    print 'grid_search.best_estimator_: ', grid_search.best_estimator_
    return grid_search.best_estimator_


def lasso_regression(Xtrain, Ytrain, Xtest, Ytest):
    #Xt = lin.transform(X,threshold=None)
    #regressor = linear_model.LassoLars(alpha=0.01,verbose=1)
    alphas = np.logspace(-6, -1, 10)
    regressor = linear_model.Lasso(max_iter=10000, normalize=True, tol=1e-100)
    scores = [regressor.set_params(alpha=alpha).fit(Xtrain, Ytrain).score(Xtest, Ytest)
              for alpha in alphas]
    best_alpha = alphas[scores.index(max(scores))]
    print 'best_alpha: ', best_alpha
    regressor.alpha = best_alpha
    regressor.fit(Xtrain,Ytrain)
    print 'number of nonzero coefficients: %d' %sum([1 for coef in regressor.coef_ if coef != 0])
    return regressor


def test_and_print(name, regressor, X, Y, Xtrain, Ytrain, Xtest, Ytest):
    print 'score of ', name, ' (train): ', score(Ytrain, regressor.predict(Xtrain))
    print 'score of ', name, ' (test): ', score(Ytest, regressor.predict(Xtest))
    scorefunction = skmet.make_scorer(score)
#   scores = skcv.cross_val_score(regressor, X, Y, scoring=scorefunction, cv=5)
#   print 'score of ', name, ' (cv) mean : ', np.mean(scores), ' +/- ', np.std(scores)


def regress(feature_fn):
    Xo = read_path('project_data/train.csv')
    print len(Xo)

    Yo = np.genfromtxt('project_data/train_y.csv', delimiter = ',')
    print 'DEBUG: data read'
    Y = np.log(1 + Yo)
    X = read_features(Xo, feature_fn)
    print 'DEBUG: total nb of base-functions: %d' %np.shape(X)[1]
    #np.std(X, axis=0) == 0
    print 'DEBUG: transform training data features'
    Xvalo = read_path('project_data/validate.csv')
    Xtesto = read_path('project_data/test.csv')
    print 'DEBUG: transform validation data features'
    Xval = read_features(Xvalo, feature_fn)
    Xtest = read_features(Xtesto, feature_fn)
    print 'DEBUG: features transformed'

    # always split training and test data!
    Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X, Y, train_size = 0.7)
    print 'DEBUG: data split up into train and test data'

    #------------optimized------------------------
    #Xtrain = read_features(Xtraino, feature_fn)
    #Xtest = read_features(Xtesto, feature_fn)
    #------------optimized------------------------

    print 'X.shape: ', X.shape

    lin = linear_regression(Xtrain, Ytrain)
    test_and_print('linear', lin, X, Y, Xtrain, Ytrain, Xtest, Ytest)

    forest = cheating_regression(Xtrain, Ytrain)
    test_and_print('forest', forest, X, Y, Xtrain, Ytrain, Xtest, Ytest)

    knn = nearest_neighbors_regression(Xtrain, Ytrain)
    test_and_print('k-nn', knn, X, Y, Xtrain, Ytrain, Xtest, Ytest)

    lasso = lasso_regression(Xtrain, Ytrain, Xtest, Ytest)
    test_and_print('lasso', lasso, X, Y, Xtrain, Ytrain, Xtest, Ytest)

    #forest.transform(X=Xval, threshold=None)
    regressor = lasso
    #Ypred = regressor.predict(regressor.transform(Xval, threshold=None))
    #predict validation data
    Ypredval = regressor.predict(Xval)
    Ypredval = np.exp(Ypredval) - 1
    print Ypredval
    np.savetxt('project_data/validate_y.txt', Ypredval)
    #predict test-data
    Ypredtest = regressor.predict(Xtest)
    Ypredtest = np.exp(Ypredtest) -1
    np.savetxt('project_data/test_y.txt', Ypredtest)
    return Ypredval


if __name__ == "__main__":
    regress(lambda x: ortho([simple_implementation, time_fourier, time_dct], x))
    #regress(lambda x: ortho([time_parts, w_parts], x))
    #regress(lambda x: ortho([time_fourier, month_w1356_poly], x))