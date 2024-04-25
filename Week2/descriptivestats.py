import matplotlib.pyplot as pyplot
import pandas as pd
import numpy as np

#load data and get shape
filename = './data/pima-indians-diabetes.data.csv'
header = ['Pregnancy_Count','Glucone_conc','Blood_pressure','Skin_thickness','Insulin','BMI','DPF','Age','Class']

data = pd.read_csv(filename, names=header)
print(data.shape)

pd.set_option('display.width', 100)
#pd.set_option('precision', 3)
pd.set_option('display.precision',3)
description = data.describe()

#get distribution of target (class)
class_counts = data.groupby('Class').size()

#get correlation between attributes
correlations = data.corr(method='pearson')

#get skew of data
skew = data.skew()

#visualise (univariate)
data.hist(figsize=[20, 20])
pyplot.show()

data.plot(kind='density', subplots=True, layout=(3,3), sharex=False, figsize=[20, 20]) 
pyplot.show()

data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False,figsize=[20, 20]) 
pyplot.show()

#visualise (multivariate)

# Correlation Matrix Plot
correlations = data.corr()

# Plot correlation matrix
'''fig = pyplot.figure(figsize=[10, 10])
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)

ticks = np.arange(0,9,1)

ax.set_xticks(ticks)
ax.set_yticks(ticks)

short_names = ['#Preg','Gluco','BloodP','Skin_Th','Insulin','BMI','DPF','Age','Class']

ax.set_xticklabels(short_names)
ax.set_yticklabels(short_names)
pyplot.show()'''

# Correlation Matrix Plot (generic)

# Plot correlation matrix
fig = pyplot.figure(figsize=[10, 10])

ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)

pyplot.show()



pd.plotting.scatter_matrix(data, figsize=[20, 20])
pyplot.show()



#data transformations

#RESCALE
array = data.values

# separate array into input and output components
X = array[:,0:8]
Y = array[:,8]

from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)


#STANDARDISE
# Standardise data (0 mean, 1 stdev)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)

# summarise transformed data
set_printoptions(precision=3)
print(rescaledX[0:5,:])

#NORMALISE
from sklearn.preprocessing import Normalizer 
# Normalise data (length of 1)
scaler = Normalizer().fit(X)

normalisedX = scaler.transform(X)

# Summarise transformed data
set_printoptions(precision=3)
print(normalisedX[0:5,:])


#BINARISE
from sklearn.preprocessing import Binarizer

binariser = Binarizer(threshold=0.0).fit(X)

binaryX = binariser.transform(X)

# summarise transformed data
set_printoptions(precision=3)
print(binaryX[0:5,:])