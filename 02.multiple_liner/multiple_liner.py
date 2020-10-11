import numpy as np
import pandas as pd
from sklearn import Liner
import matplotlib.pyplot as plt

import seaborn as sns

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

# Input and Output

X = df.iloc[:, :-1]
y = df.iloc[:, -1]