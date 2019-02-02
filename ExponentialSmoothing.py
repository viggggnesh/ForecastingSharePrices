from statsmodels.tsa.api import Holt, SimpleExpSmoothing, ExponentialSmoothing
import pandas as pd

df = pd.read_csv('2014.csv')
print(df.head())
