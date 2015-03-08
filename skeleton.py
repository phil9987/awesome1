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
    logdif = np.log(1 + np.exp(gtruth)) - np.log(1 + np.exp(gpred))
#   logdif = np.log(1 + gtruth) - np.log(1 + gpred)
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

def get_features_poly(x,d):
    y = []
    for i in range(d):
        y.append(np.power(x, i+1))
    return y

def get_features_exp(x):
    y = [x, np.exp(x)]
    return y

def read_path(inpath):
    X = []
    with open(inpath,'r') as fin:
        reader = csv.reader(fin, delimiter=',')
        for row in reader:
            x = [1] # just an offset
            t = datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
            (isoy, isow, isowd) = t.isocalendar()
#            x.append((float(t.month)-6.5)/3.43)
#           x.append((float(isow)-26.74)/15)
#           x.append((float(t.hour)-11.6)/6.93)
#           x.append((float(row[1])-0.5)/0.234)
#           x.append((float(isowd)-3.98)/2)
#           x.append((float(t.day)-15.6)/8.76)
#           x.append((float(row[2])-2.09)/0.765)
#           x.append((float(row[3])-0.477)/0.207)
#           x.append((float(row[4])-18.04)/2.87)
#           x.append((float(row[5])-0.20)/0.14)
#           x.append((float(row[6])-0.623)/0.233)
#           x.extend(get_features_poly((float(t.month)-6.5)/3.43, 3))
#           x.extend(get_features_poly((float(t.hour)-11.6)/6.93, 15))
#           x.extend(get_features_poly((float(row[1])-0.5)/0.234, 9))
#           x.extend(get_features_poly((float(row[3])-0.477)/0.207, 5))
            #x.extend(get_features_poly(float(isow),6))
            #x.extend(get_features_poly(float(t.hour),16))
            #x.extend(get_features_poly((float(row[1])),3))
            #x.extend(get_features_poly((float(row[3])),3))
#           x.extend(get_features_exp((float(row[6])-0.623)/0.233))
            x.extend(get_features_poly(float(isow-3.98)/2, 6))
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
    param_grid = {'alpha' : np.linspace(0,5,10)}
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
    X = read_path('project_data/train.csv')
    Yo = np.genfromtxt('project_data/train_y.csv',delimiter = ',')
#   Y = (Y - np.mean(Y))/np.std(Y)
    Y = np.log(Yo)
    Xval = read_path('project_data/validate.csv')
    print X

    # always split training and test data!
    Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X,Y,train_size=0.75)

    lin = linear_regression(X,Y,Xtrain,Ytrain,Xtest,Ytest)
    Ypredtrain = lin.predict(Xtrain)
    Ypredtest = lin.predict(Xtest)
    print 'lin score on Xtrain,Ytrain: ', logscore(Ytrain,Ypredtrain)
    print 'lin score on Xtest,Ytest: ', logscore(Ytest,Ypredtest)

    #ridge = ridge_regression(Xtrain,Ytrain,Xtest)
    #print 'score of ridge (train): ', logscore(Ytrain, ridge.predict(Xtrain))
    #print 'score of ridge (test): ', logscore(Ytest, ridge.predict(Xtest))

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