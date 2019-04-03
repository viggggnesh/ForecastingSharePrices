import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import tree

#Open dataset to read from
data = pd.read_csv('2009.csv')
data = data[['Open','High','Low','Close']]

#Plot all the concerned attributes to establish correlation
plt.rcParams['figure.figsize']=(20,10)
plt.style.use('ggplot')
data.plot(subplots=True)
plt.show()

#Split dataset attributes into features and prediction label
dX = data[['Open','High','Low']]
dy = data[['Close']]

#Split data into train and test
dX_train = dX[dX.index < 198]
dy_train = dy[dy.index < 198]

dX_test = dX[dX.index >= 198]
dy_test = dy[dy.index >= 198]

#Initiate Random Forest Model which makes use of 3 features to predict 'Close' for given data
RF_Model = RandomForestRegressor(n_estimators=100, max_features=3, oob_score=True)

#Fit data into the model
#Set 'Close' as the label to predict
label = dy_train[:]
features = dX_train[:]
rgr=RF_Model.fit(features, label)

#Using training features, predict the 'Close' value and store data into file
dX_test_predict=pd.DataFrame(
    rgr.predict(dX_test[:])).rename(
    columns={0:'predicted_close'}).set_index('predicted_close')
dX_train_predict=pd.DataFrame(
    rgr.predict(dX_train[:])).rename(
    columns={0:'predicted_close'}).set_index('predicted_close')

#Add predict values to the forest and append
RF_predict = dX_train_predict.append(dX_test_predict)

#Display file with new column of 'predicted_close'
data = data.join((RF_predict.reset_index()))
print(data.head())

#Plot 'Close' vs 'predicted_close'
data[['Close','predicted_close']].plot()
plt.show()

#Calculate the Mean Squared Error
data['Diff']=data.predicted_close - data.Close
data['Diff'].plot(kind='bar')
plt.show()

#Calculate the Mean Absolute Percent Error
dy_true = data[['Close']]
dy_predicted = data[['predicted_close']]
mape = mean_absolute_error(dy_true,dy_predicted)
mse = mean_squared_error(dy_true,dy_predicted)

#Print MAE, MAPE and Accuracy
print('Mean Squared Error - ',mse)
print('Mean Average Percent Error - ',mape)
print('Estimated Accuracy - ',(100-mape))

