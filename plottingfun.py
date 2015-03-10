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
sns.lmplot('hour','Y', feature_dataframe,x_jitter=0.2, order = 3)
sns.lmplot('hour','Y', feature_dataframe,x_jitter=0.2,x_estimator=np.mean, order = 3)
#sns.lmplot('hour','Y',feature_dataframe, hue='W2',x_estimator=np.mean)        #lol...it's easy to visualize how this relationship changes in different subsets of your dataset
#sns.lmplot('W2','Y',feature_dataframe,order = 2)
f, (ax1, ax2) = plt.subplots(1,2,sharey=True)
sns.regplot('hour','Y',feature_dataframe, ax = ax1)
sns.boxplot(feature_dataframe['Y'],feature_dataframe['W2'], color = 'Blues_r', ax = ax2).set_label('')
f.tight_layout()
plt.show()