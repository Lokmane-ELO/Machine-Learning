import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
# load dataset
df = pd.read_csv('AirPassengers.csv', usecols=['#Passengers'])

# overview of the data
print(df.head())
df.plot()
plt.show()
# convert to numpy array
data = df.values

# split into train and test
train_size = int(len(data) * 0.66)
train, test = data[0:train_size], data[train_size:len(data)]
# build and train ARIMA model
model = ARIMA(train, order=(5,1,0))
model_fit = model.fit(disp=0)

# make predictions
predictions = model_fit.forecast(steps=len(test))[0]
# calculate out of sample error
error = mean_squared_error(test, predictions)
print(f'Test MSE: {error:.3f}')

# plot results
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()
