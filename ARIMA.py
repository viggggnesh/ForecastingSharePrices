#Import required modules
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import statsmodels.tsa.seasonal as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace import sarimax
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima.arima.utils import ndiffs

#Load dataset and plot
data = pd.read_csv('2016.csv')
data = data[['Open','High','Low','Close']]

#Plot the label we want to predict and check for its seasonality
#Seasonality can be predicted using statistical methods or ADF test
#In this module, the ADF test has been taken, as it is more accurate in assessing whether the data is stationary or not.
rolmean = data.Close.rolling(30).mean()
rolstd = data.Close.rolling(30).std()
plt.plot(data.Close)
plt.plot(rolmean,color='red')
plt.plot(rolstd, color='black')
plt.title('Rolling Statistics vs Time Series')
plt.show()
result = adfuller(data.Close, autolag='AIC')
print('ADF Statistic: %f'% result[0])
print('p-value: %f'% result[1])
print('#Lags Used: ', result[2])
print('Observations Used: ', result[3])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

#Plot the ACF value
plot_acf(data.Close)
plt.show()

#Prints the required order of differencing to make the data stationary
#The suggested value may not be the best differencing for the dataset
#However, using more than the suggested difference will make the data overstationary and unfit for use
print('Best value of difference to use is - ',ndiffs(data.Close,test='adf'))

#Plot MA of the log
data1 = np.log(data[['Close']])
moving_avg = data1.rolling(30).mean()
plt.plot(data1)
plt.plot(moving_avg, color='red')
plt.show()

#Take the moving average of the data by subtracting the moving average for 12 time units from the original data
series_ma = data1 - moving_avg
series_ma.dropna(inplace=True)

#Perfrom rolling statistics and the ADF test again on the log values
rolmean = series_ma.rolling(30).mean()
rolstd = series_ma.rolling(30).std()
plt.plot(series_ma)
plt.plot(rolmean,color='red')
plt.plot(rolstd, color='black')
plt.title('Rolling Statistics vs Log Transform')
plt.show()
series_ma = series_ma.iloc[:,0].values
result = adfuller(series_ma, autolag='AIC')
print('ADF Statistic: %f'% result[0])
print('p-value: %f'% result[1])
print('#Lags Used: ', result[2])
print('Observations Used: ', result[3])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

#Perform order differencing based on ndiffs using .shift() method
ts_log_diff = data1 - data1.shift()
ts_log_diff.dropna(inplace=True)
plt.plot(ts_log_diff)
plt.title('After Log Differencing')
plt.show()
rolmean = ts_log_diff.rolling(30).mean()
rolstd = ts_log_diff.rolling(30).std()
plt.plot(ts_log_diff)
plt.plot(rolmean,color='red')
plt.plot(rolstd, color='black')
plt.title('Rolling Statistics vs Log Differencing')
plt.show()
ts_log_diff = ts_log_diff.iloc[:,0].values
result = adfuller(ts_log_diff, autolag='AIC')
print('ADF Statistic: %f' % result[0])
print('p-value: %f'% result[1])
print('#Lags Used: ', result[2])
print('Observations Used: ', result[3])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

#Plot ACF:
plot_acf(ts_log_diff, lags=20)
plt.title('Autocorrelation Function')

#Plot PACF:
plot_pacf(ts_log_diff, lags=20, method='ols')
plt.title('Partial Autocorrelation Function')
plt.show()

#Plot ARIMA
model = ARIMA(ts_log_diff, order=(5, 1, 5))
results_ARIMA = model.fit(disp=1)
plt.plot(data.Close)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.show()
print(results_ARIMA.fittedvalues)

'''#Print MAE, MAPE and Accuracy
print('Mean Squared Error - ',mse)
print('Mean Average Percent Error - ',mape)
print('Estimated Accuracy - ',(100-mape))'''