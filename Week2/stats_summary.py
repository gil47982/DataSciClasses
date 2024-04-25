# import your libraries here and write your code

import pandas as pd
from sklearn.preprocessing import StandardScaler
 

pd.set_option("display.precision", 3)
#read data
df = pd.read_csv('./data/census_adult_income.csv')

#quick visual inspection
print(df.head(5))
print(df.shape)
print(df.describe())


#check distribution of target variable
class_counts = df.groupby('target').size()
print(class_counts)


#check skew
skew = df.skew()
#print(df.skew())

# correlations = df.corr(method='pearson')
# print(correlations)

# #transform data

# #extract numeric columns
# # Create a list of the columns with numeric data
# filter_names = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

# # Select the columns
# numeric_features = df[filter_names]
