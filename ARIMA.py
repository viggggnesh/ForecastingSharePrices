import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from pyramid.arima import auto_arima

data = pd.read_csv("2014.csv")

x = data['Date']
y = data['Turnover (Rs. Cr)']

result = seasonal_decompose(x,y,model='multiplicative')

fig = result.plot()

fig.show()

model = auto_arima(data, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)
print(model.aic())

train = data.loc['1985-01-01':'2016-12-01']
test = data.loc['2017-01-01':]

future_forecast = stepwise_model.predict(n_periods=37)
future_forecast = pd.DataFrame(future_forecast,index = test.index,columns=['Prediction'])
pd.concat([test,future_forecast],axis=1).iplot()
pd.concat([data,future_forecast],axis=1).iplot()



