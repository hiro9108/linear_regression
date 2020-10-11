# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# import seaborn as sns

"""
Multiple Linear Regression
"""

# Read csv file
df = pd.read_csv('sample_multiple_liner_data.csv')

# Show plot with seaborn
# sns.distplot(df['y'], bins=50)
# plt.show()

# check correlation
# print(df.corr())

# check correlation with graph
# sns.pairplot(df, height=0.75, aspect=1.8)
# plt.show()

# Separate Input(x) and Output(y) values

X = df.iloc[:, :-1]
y = df.iloc[:, -1]


"""Using sklearn"""
# Declare the model
model = LinearRegression()

"""(Not separate data -> Using all data for creating model)
# Learning the model
model.fit(X, y)

# test
print("All data (100%):", model.score(X, y))
"""


"""Separate train and test data"""
# test data is 40%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# Learning the model with train data
model.fit(X_train, y_train)

# test with test data
print("Train data (60%)", model.score(X_test, y_test))

# test with train data (sample)
# print("Train data (60%)", model.score(X_train, y_train))

"""Predict value"""
x = X.iloc[0, :]
y_predict = model.predict([x])
# print(X)
# print(x)
print(y_predict)


"""Save the model"""
joblib.dump(model, 'model.pkl')

"""load the model"""
model_load = joblib.load('model.pkl')
print(model_load.predict([x]))

# Check parameter
# print(model.coef_)

# Easy to read
# np.set_printoptions(precision=3, suppress=True)
# print(model.coef_)