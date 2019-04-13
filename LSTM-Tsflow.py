#Import required modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Load the required dataset
dataset_train = pd.read_csv('2016.csv')
training_set = dataset_train.iloc[:, 4:5].values
print(dataset_train.head())

#Scaling the data using MinMax scaler
MMScl = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = MMScl.fit_transform(training_set)

#Using the first 60 stamps as features
#Using the 61 stamp onwards as label
X_train = []
y_train = []
for i in range(60, 247):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Create the LSTM model using Keras and TF API
#Using the Sequential LSTM, 4 Dropout layers and 1 Dense layer
#Training for 100 Epochs with batch size as 32
regressor = Sequential()
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

#Load the testing data set.
dataset_test = pd.read_csv('2017.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

#Concatenate the testing set to match the values for training set
#Then scale, reshape and fit the data into model
dataset_total = pd.concat((dataset_train['Close'], dataset_test['Close']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = MMScl.transform(inputs)
X_test = []
for i in range(60, 248):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#Create a prediction model based on testing data
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = MMScl.inverse_transform(predicted_stock_price)

#Plot the comparison for real and predicted values
plt.plot(real_stock_price, color = 'red', label = 'Actual Close Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted Close Price')
plt.title('Close Prediction')
plt.xlabel('Time')
plt.ylabel('Close')
plt.legend()
plt.show()

#Estimate the MSE and MAPE between the real and predicted values
data = pd.read_csv('2016.csv', nrows=188)
dy_true = data[['Close']]
dy_predicted = predicted_stock_price
mape = mean_absolute_error(dy_true,dy_predicted)
mse = mean_squared_error(dy_true,dy_predicted)

#Print MAE, MAPE and Accuracy
print(mse)
print(mape)
