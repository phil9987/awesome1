import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from skeleton import read_path

#Tutorial: http://web.stanford.edu/~mwaskom/software/seaborn/tutorial/quantitative_linear_models.html

X = read_path('project_data/train.csv', 'none')
Yo = np.genfromtxt('project_data/train_y.csv',delimiter = ',')
Yo = np.log(Yo)
Yo = Yo.reshape(-1,1)       #Yo to column-vector
all_data = np.append(X,Yo,1)
#all_data = all_data.T
print 'DEBUG: shape of all_data is: ' + str(all_data.shape)
#   Y = (Y - np.mean(Y))/np.std(Y)
#sns.lmplot("hour", "number passengers",)
feature_dataframe = pd.DataFrame(all_data, columns=['offset','week','hour','W1','W2','W3','Y'],dtype=float)
print feature_dataframe.columns
print feature_dataframe
sns.lmplot('hour','Y', feature_dataframe,x_jitter=0.2)
sns.lmplot('hour','Y', feature_dataframe,x_jitter=0.2,x_estimator=np.mean)
sns.lmplot('hour','Y',feature_dataframe, hue='W2',x_estimator=np.mean)        #lol...it's easy to visualize how this relationship changes in different subsets of your dataset
sns.lmplot('hour','Y',feature_dataframe,order = 3)
plt.show()