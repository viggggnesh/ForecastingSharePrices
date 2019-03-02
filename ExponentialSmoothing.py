from statsmodels.tsa.api import Holt, SimpleExpSmoothing, ExponentialSmoothing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('2014.csv')

x = df['Date']
y = df['Turnover (Rs. Cr)']

plt.plot(x,y)

plt.show()