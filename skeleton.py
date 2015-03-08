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

# TODO: Revert the scoring, so that it corresponds to a least square
def logscore(gtruth,gpred):
    gpred = np.clip(gpred,0,np.inf)
    logdif = np.log(1 + np.exp(gtruth)-1) - np.log(1 + np.exp(gpred)-1)
#   logdif = np.log(1 + gtruth) - np.log(1 + gpred)
    return np.sqrt(np.mean(np.square(logdif)))


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


def get_features_poly_nd(x, d):
    y = []
    for i in range(d+1):
        y.extend(monomials(x,d))
    return y


def eval_poly3d(x):
    y = [1]
    y.extend(x)
    y.extend(x*x)
    y.extend(x[:-1]*x[1:])
    y.extend(x*x*x)
    y.extend(x[:-1]*x[:-1]*x[1:]) # x1^2*x2,x2^2*x3,...x(n-1)^2*x(n)
    y.extend(x[:-1]*x[1:]*x[1:])  # x1*x2^2,...,x(n-1)*x(n)^2
    y.extend(x[:-2]*x[1:-1]*x[2:]) # x1*x2*x3,...,x(n-2)*x(n-2)*x(n-3)
    return y


def get_features_poly3d(x):
    return eval_poly3d(np.array(x))

def get_features_poly(x,d):
    y = []
    for i in range(d):
        y.append(np.power(x, i+1))
    return y

def get_features_exp(x):
    y = [x, np.exp(x)]
    return y

def get_features_fourier(x, d, r):
    y = []
    w = (np.pi*2)/r
    for i in range(d):
        y.append(np.sin((i+1)*x*w))
        y.append(np.cos((i+1)*x*w))
    return y

def get_features_fourier_md(x, d, r): # TODO
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


# TODO: Read in std, mean with python code (not hardcoded)
def read_path(inpath, basefun):
    X = []
    epoch = datetime.datetime.utcfromtimestamp(0)
    with open(inpath,'r') as fin:
        reader = csv.reader(fin, delimiter=',')
        for row in reader:
            x = [1] # just an offset
            t = datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
            d = ((t - epoch).days - 16078)
#           x.extend(get_features_fourier(d, 8, 365))
#           x.extend(get_features_fourier(d, 8, 30))
#           x.extend(get_features_fourier(d, 8, 1.0))
#           x.extend(get_features_fourier(d, 8, 1.0/24))

#           x.extend(get_features_poly((float(t.hour)-11.6)/6.93, 15))
#           x.extend(get_features_poly((float(row[1])-0.5)/0.234, 9))
#           x.extend(get_features_poly((float(row[3])-0.477)/0.207, 5))
            #x.extend(get_features_poly(float(isow),6))
            #x.extend(get_features_poly(float(t.hour),16))
            #x.extend(get_features_poly((float(row[1])),3))
            #x.extend(get_features_poly((float(row[3])),3))
#           x.extend(get_features_exp((float(row[6])-0.623)/0.233))
            if basefun == 'none' :
                x.append(float(t.isoweekday()))
                x.append(float(t.hour))
                x.append(row[1])
                x.append(row[2])
                x.append(row[3])
            elif basefun == 'normalized':
                x.append(float(t.isoweekday()-3.98)/2)
                x.append(float(t.hour-11.6)/6.93)
                x.append((float(row[1])-0.5)/0.234)
                x.append((float(row[2])-0.5)/0.234)
                x.append((float(row[3])-0.477)/0.207)
            elif basefun == 'poly':
                x.extend(get_features_poly(float(t.isoweekday()-3.98)/2, 6))
                x.extend(get_features_poly(float(t.hour-11.6)/6.93, 16))
                x.extend(get_features_poly((float(row[1])-0.5)/0.234, 3))
                x.extend(get_features_poly((float(row[3])-0.477)/0.207, 3))
            X.append(x)

    return np.matrix(X)


def linear_regression(X,Y,Xtrain,Ytrain,Xtest,Ytest):
    regressor = sklin.LinearRegression()
    regressor.fit(Xtrain,Ytrain)
    print 'regressor.coef_: ', regressor.coef_
    print 'regressor.intercept_: ', regressor.intercept_
    scorefunction = skmet.make_scorer(logscore)
    scores = skcv.cross_val_score(regressor,X,Y,scoring=scorefunction,cv = 10)
    print 'mean : ', np.mean(scores),' +/- ' , np.std(scores)
    return regressor

def ridge_regression(Xtrain,Ytrain,Xval):
    regressor_ridge = sklin.Ridge(fit_intercept=False, normalize=False)
    param_grid = {'alpha' : np.linspace(0,1,100)}
    n_scorefun = skmet.make_scorer(lambda x, y: -logscore(x,y)) # logscore is always maximizing... but we want the minium
    grid_search = skgs.GridSearchCV(regressor_ridge, param_grid, scoring = n_scorefun, cv = 10)
    grid_search.fit(Xtrain,Ytrain)
    print 'grid_search.best_estimator_: ', grid_search.best_estimator_
    Ypred = grid_search.best_estimator_.predict(Xval)
    #Yplot = grid_search.best_estimator_.predict(Xplot)                #predictions

    #plt.plot(Xtrain[:,0], Ytrain, 'bo')             #input data
    #plt.plot(Hplot,Yplot,'r',linewidth = 3)         #prediction
    #plt.show()
    return grid_search.best_estimator_

def main():
    X = read_path('project_data/train.csv', 'poly')
    Yo = np.genfromtxt('project_data/train_y.csv',delimiter = ',')
#   Y = (Y - np.mean(Y))/np.std(Y)
    Y = np.log(Yo)
    Xval = read_path('project_data/validate.csv', 'poly')
    print X

    # always split training and test data!
    Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X,Y,train_size=0.75)

    lin = linear_regression(X,Y,Xtrain,Ytrain,Xtest,Ytest)
    Ypredtrain = lin.predict(Xtrain)
    Ypredtest = lin.predict(Xtest)
    print 'lin score on Xtrain,Ytrain: ', logscore(Ytrain,Ypredtrain)
    print 'lin score on Xtest,Ytest: ', logscore(Ytest,Ypredtest)

#   ridge = ridge_regression(Xtrain,Ytrain,Xtest)
#   print 'score of ridge (train): ', logscore(Ytrain, ridge.predict(Xtrain))
#   print 'score of ridge (test): ', logscore(Ytest, ridge.predict(Xtest))

    Ypred = lin.predict(Xval)
    Ypred = np.exp(Ypred)
    print Ypred
    np.savetxt('project_data/validate_y.txt', Ypred)
    plt.figure(1)
    plt.subplot(211)
    inputdata1, = plt.plot(Xtrain[:,1], Ytrain, 'bo', label='Ytrain')             #input data
    predictdata1, = plt.plot(Xtrain[:,1],Ypredtrain,'ro',label='prediction Ytrain')
    plt.legend(handles=[inputdata1,predictdata1])
    plt.title('Train-data')
    plt.subplot(212)
    inputdata2, =plt.plot(Xtest[:,1],Ytest,'bo', label='Ytest')
    predictdata2, = plt.plot(Xtest[:,1], Ypredtest,'ro', label='prediction Ytest')
    plt.legend(handles=[inputdata2,predictdata2])
    plt.title('Test-data')

    plt.show()


if __name__ == "__main__":
    main()