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

def eval_custom(x):
    y = eval_poly3d(x)
    for i in range(3):
         y.append(np.power(x[0],i+4))
    for i in range(3):
         y.append(np.power(x[1],i+4))
    for i in range(13):
        y.append(np.power(x[2],i+4))
    for i in range(6):
        y.append(np.power(x[3],i+4))
    return y

def get_features(x):
    return eval_custom(np.array(x))

def read_path(inpath):
    X = []
    n = 0
    with open(inpath,'r') as fin:
        reader = csv.reader(fin, delimiter=',')
        for row in reader:
            x = []
            t = datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
            (isoy, isow, isowd) = t.isocalendar()
#           x.append(t.year-2014)
            x.append((float(t.month)-6.5)/3.43)
            x.append((float(isow)-26.74)/15)
            x.append((float(t.hour)-11.6)/6.93)
            x.append((float(row[1])-0.5)/0.234)
#           x.append((float(isowd)-3.98)/2)
#           x.append((float(t.day)-15.6)/8.76)
#           x.append((float(row[2])-2.09)/0.765)
#           x.append((float(row[3])-0.477)/0.207)
#           x.append((float(row[4])-18.04)/2.87)
#           x.append((float(row[5])-0.20)/0.14)
#           x.append((float(row[6])-0.623)/0.233)
            c = len(get_features(x))
            X.append(get_features(x))
            n = n+1
    return np.reshape(X, [n, c])


def linear_regression(X,Y,Xtrain,Ytrain,Xtest,Ytest,Xval):
    regressor = sklin.LinearRegression()
    regressor.fit(Xtrain,Ytrain)
    print 'regressor.coef_: ', regressor.coef_
    print 'regressor.intercept_: ', regressor.intercept_
    #Hplot = range(25)
    #Xplot = np.atleast_2d([get_features(x) for x in Hplot])
    #Yplot = regressor.predict(Xplot)                # predictions
    #plt.plot(Xtrain[:,0], Ytrain, 'bo')             # input data
    #plt.plot(Hplot,Yplot,'r',linewidth = 3)         # prediction
    #plt.show()
    print 'lin score on Xtrain,Ytrain: ', logscore(Ytrain,regressor.predict(Xtrain))
    print 'lin score on Xtest,Ytest: ', logscore(Ytest,regressor.predict(Xtest))
    scorefunction = skmet.make_scorer(logscore)
    scores = skcv.cross_val_score(regressor,X,Y,scoring=scorefunction,cv = 10)
    print 'mean : ', np.mean(scores),' +/- ' , np.std(scores)
    return regressor.predict(Xval)

def ridge_regression(Xtrain,Ytrain,Xval):
    regressor_ridge = sklin.Ridge(fit_intercept=False)
    param_grid = {'alpha' : np.linspace(0,5,10)}
    n_scorefun = skmet.make_scorer(lambda x, y: -logscore(x,y)) # logscore is always maximizing... but we want the minium
    grid_search = skgs.GridSearchCV(regressor_ridge, param_grid, scoring = n_scorefun, cv = 1000)
    grid_search.fit(Xtrain,Ytrain)
    print 'grid_search.best_estimator_: ', grid_search.best_estimator_
    Ypred = grid_search.best_estimator_.predict(Xval)
    #Yplot = grid_search.best_estimator_.predict(Xplot)                #predictions

    #plt.plot(Xtrain[:,0], Ytrain, 'bo')             #input data
    #plt.plot(Hplot,Yplot,'r',linewidth = 3)         #prediction
    #plt.show()
    return grid_search.best_estimator_

def main():
    X = read_path('project_data/train.csv')
    Y = np.genfromtxt('project_data/train_y.csv',delimiter = ',')
    Xval = read_path('project_data/validate.csv')
    print X

    # always split training and test data!
    Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X,Y,train_size=0.75)

    linear_regression(X,Y,Xtrain,Ytrain,Xtest,Ytest,Xval)

    ridge = ridge_regression(Xtrain,Ytrain,Xtest)
    print 'score of ridge (train): ', logscore(Ytrain, ridge.predict(Xtrain))
    print 'score of ridge (test): ', logscore(Ytest, ridge.predict(Xtest))

    Ypred = ridge.predict(Xval)
    np.savetxt('project_data/validate_y.txt', Ypred)

if __name__ == "__main__":
    main()
