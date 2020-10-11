# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def predict(predict_value):
    """Simple linear Regression"""

    # Read csv file
    df = pd.read_csv('sample_data.csv')
    print(df.head(3))

    # Scatter of Centering data
    # plt.scatter(df['x'], df['y'])
    # plt.show()

    # Centering ( average: df.mean() )
    df_c = df - df.mean()
    print (df_c.head(3))

    # Collect centering data
    x = df_c['x']
    y = df_c['y']

    # Scatter of Centering data
    # plt.scatter(x,y)
    # plt.show()

    # multiple elements
    xx = x * x
    xy = x * y

    a = xy.sum() / xx.sum()

    # Calculate predict value
    mean = df.mean()
    x_c = predict_value - mean['x']

    # Output predict value
    print(a * x_c + mean['y'])

    # Real data
    plt.scatter(x, y, label='Real Data')
    # Predict linear
    plt.plot(x, a * x, label='Predict', color='red')
    # Show the Graph of simple linear regression model
    plt.legend()
    plt.show()


predict(40)    #   145006.92036590326 -> This works

# predict(10) -> -157063.75521261862 -> This doesn't work because not enough data (out of range)
