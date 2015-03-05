import numpy as np                  #basic linear algebra
import matplotlib.pyplot as plt     #plotting functions
import csv
import datetime
import sklearn.linear_model as sklin
import sklearn.metrics as skmet
import sklearn.cross_validation as skcv
import sklearn.grid_search as skgs

def logscore(gtruth,gpred):
    gpred = np.clip(gpred,0,np.inf)
    logdif = np.log(1 + gtruth) - np.log(1 + gpred)
    return np.sqrt(np.mean(np.square(logdif)))


def get_features(t):
    return [1, t, np.power(t,2), np.power(t,3)]
    #return [t, np.power(t,2)]
    #return [t,t^2 + 5*t + 5]

def read_path(inpath):
    X = []
    n = 0
    c = (3+6)*len(get_features(1)) # number of features * columns per feature
    with open(inpath,'r') as fin:
        reader = csv.reader(fin, delimiter=',')
        for row in reader:
            t = datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
            X.append(get_features(t.year))
            X.append(get_features(t.hour))
            X.append(get_features(t.weekday()))
            for i in range(6):
                X.append(get_features(float(row[i+1])))
            n = n+1
    return np.reshape(X, [n, c])


def linear_regression(Xtrain,Ytrain,Xtest,Ytest,Xval):
    regressor = sklin.LinearRegression()
    regressor.fit(Xtrain,Ytrain)
    print 'regressor.coef_: ', regressor.coef_
    print 'regressor.intercept_: ', regressor.intercept_
    Hplot = range(25)
    #Xplot = np.atleast_2d([get_features(x) for x in Hplot])
    #Yplot = regressor.predict(Xplot)                # predictions
    #plt.plot(Xtrain[:,0], Ytrain, 'bo')             # input data
    #plt.plot(Hplot,Yplot,'r',linewidth = 3)         # prediction
    #plt.show()
    print 'score on Xtest,Ytest: ', logscore(Ytest,regressor.predict(Xtest))
    scorefunction = skmet.make_scorer(logscore)
    scores = skcv.cross_val_score(regressor,X,Y,scoring=scorefunction,cv = 10)
    print 'mean : ', np.mean(scores),' +/- ' ,np.std(scores)
    return regressor.predict(Xval)

def ridge_regression(X,Y,Xtrain,Ytrain,Xval):
    regressor_ridge = sklin.Ridge()
    param_grid = {'alpha' : np.linspace(0,100,10)}              # number of alphas is arbitrary
    n_scorefun = skmet.make_scorer(lambda x, y: -logscore(x,y)) # logscore is always maximizing... but we want the minium
    grid_search = skgs.GridSearchCV(regressor_ridge, param_grid, scoring = n_scorefun, cv = 10)
    grid_search.fit(Xtrain,Ytrain)
    print 'grid_search.best_estimator_: ', grid_search.best_estimator_
    print 'grid_search.best_score_: ', grid_search.best_score_
    Ypred = grid_search.best_estimator_.predict(Xval)
    #Yplot = grid_search.best_estimator_.predict(Xplot)                #predictions

    plt.plot(Xtrain[:,0], Ytrain, 'bo')             #input data
    #plt.plot(Hplot,Yplot,'r',linewidth = 3)         #prediction
    plt.show()
    return Ypred


def main():
    X = read_path('project_data/train.csv')
    Y = np.genfromtxt('project_data/train_y.csv',delimiter = ',')
    Xval = read_path('project_data/validate.csv')

    # always split training and test data!
    Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X,Y,train_size=0.75)

    #linear_regression(Xtrain,Ytrain,Xtest,Ytest,Xval)

    Ypred = ridge_regression(X,Y,Xtrain,Ytrain,Xval)

    np.savetxt('project_data/validate_y.txt', Ypred)

if __name__ == "__main__":
    main()
