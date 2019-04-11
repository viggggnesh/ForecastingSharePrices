import numpy as np
import pandas as pd
from TFANN import ANNR
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

#Load dataset and plot
data = pd.read_csv('2016.csv')

A = np.loadtxt(data, delimiter=",", skiprows=1, usecols=(2, 5))
A = scale(A)
# y is the dependent variable
y = A[:, 1].reshape(-1, 1)
# A contains the independent variable
A = A[:, 0].reshape(-1, 1)
# Plot the high value of the stock price
plt.plot(A[:, 0], y[:, 0])
plt.show()