# Stock_Price_Prediction
the main aim of this project is Modelling the price movements and volatilities in stock prices and prediction of future stock prices using Artificial Neural Networks and LSTM. 
At the same time it is important to verify the performance of the model on other stocks.

The Closing Stock prices of Apple from 2012-2017 data was retireved from Yahoo Finance for craeting the model.
Tensorflow is the most preferred library for deeplearning. Keras Library was used for Nueral networks.

#Import the libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#Get the stock quote
df = web.DataReader('AAPL', data_source='yahoo',  start='2012-01-01', end='2019-12-17')
#Show the data
df.head()
df.shape()

the primary step and most important step for any data set is to unde.rstand the data. This is achieved through data visualization. 
Hence, here verify if there are any null values(which ishighly unlikely given stock prices) and plot the dat for better undertsanding

#Visualize the closing price history
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])

plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()

df.isnull.sum()

Since there areno null values,we can proceed with thenext steps.
Since our main agaenda is to predict the  closing stock price, we filter the Close prices and create a new data frame

#Create a new dataframe with only the 'Close column
data = df.filter(['Close'])
#Convert the dataframe to a numpy array
dataset = data.values

The next step in the proccessing is to divide the data in train and test data. Train data is used for training the data and test set is used to check the accuracy of our model through a loss function.#Get the number of rows to train the model on
work_data=len(dataset)
training_data_len = math.ceil( len(dataset) * .8 )
type(training_data_len)

Since neural networks is sensitive to high magnitudes, and in order to improve the processing speed, aid the backpropagation, we do a min maxscaler on the data.
#Scale the data
#scaling done by (x-min)/range
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
scaled_data

for training the model, each stock price was taken as a function of previous 60 days stock prices for a company(Apple)
.i.e current stock price is predicted on the basis of previous 60 time steps.

#Create the scaled training data set
train_data = scaled_data[0:training_data_len , :]
#Split the data into x_train and y_train data sets
x_train = []
y_train = []

print(len(train_data))

for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i, 0])
  y_train.append(train_data[i, 0])
  if i<= 62:
    print(x_train)
    print(y_train)
    print()
    
#Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)

LSTM modelrequires data in 3 dimesnions, hence the data is transformed to that fromat.
#A LSTM network expects the input to be 3-Dimensional in the form [samples, time steps, features]:
# samples is the number of data points (or rows/ records) we have, 
# time steps is the number of time-dependent steps that are there in a single data point (60),
# features/indicators refers to the number of variables we have for the corresponding true value in Y, since we are only using one feature 'Close',
# the number of features/indicators will be one
#Reshape the data into the shape accepted by the LSTM
#everytime we use LSTM, we have to take the tiemsteps.

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape

Here, we build a 4-layer LSTM model with 2 LSTM layers and 2 neural network layers
#Build the LSTM model
model = Sequential() #type of model #we imported sequential above
#First LSTM layer, input_shape = the number of time steps (60 or x_train.shape[1]) while the last parameter is the number of features (1)
model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1)))#Units=number of neurons, return_sequences is set to true since we will add more layers to the model.
model.add(LSTM(50, return_sequences= False))#LSTM layer with 50 neurons, return sequence by default is False but I put it here just to show
model.add(Dense(25))#Just your regular densely-connected Neural Network layer with 25 neurons
model.add(Dense(1))#Just your regular densely-connected Neural Network layer with 1 neuron



#Compile the model
# optimizer = adam and the optimizer is used to improve upon the loss
# loss function = mean_squared_error (MSE) and loss functions are used to measure how well the model did on training
model.compile(optimizer='adam', loss='mean_squared_error')

#Train the model
model.fit(x_train, y_train, batch_size=60, epochs=1)
#A sequence prediction problem makes a good case for a varied batch size as you may want to have a batch size equal to the training dataset size (batch learning) during training and a batch size of 1 when making predictions for one-step outputs

We fit the model.the loss function used for assessing performance the model is Mean Squared error. The model is testing on the Testing data.
#Create the testing data set
#Create a new array containing scaled values from index 1543 to 2002 
test_data = scaled_data[training_data_len - 60: , :]
#Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
 #Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))
#Get the models predicted price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)#to convert the scaled vakues to original values of prices
predictions.shape
#Get the root mean squared error (RMSE)
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
rmse

#Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
valid.shape

the model has performed satisfactorily well on the data. hence we use it to predict the future stock prices-post 2017.
#Get the quote
apple_quote = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2019-12-17')
#Create a new dataframe
new_df = apple_quote.filter(['Close'])

#Get teh last 60 day closing price values and convert the dataframe to an array
last_60_days = new_df[-60:].values

#Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)

#Create an empty list
X_test = []

#Append teh past 60 days
X_test.append(last_60_days_scaled)

#Convert the X_test data set to a numpy array
X_test = np.array(X_test)

#Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#Get the predicted scaled price
pred_price = model.predict(X_test)

#undo the scaling 
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)
#Get the quote
apple_quote2 = web.DataReader('AAPL', data_source='yahoo', start='2019-12-18', end='2019-12-18')
print(apple_quote2['Close'])
