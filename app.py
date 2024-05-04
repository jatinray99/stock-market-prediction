import numpy as np 
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

model=load_model('Stock Predictions Model')

st.header('Stock Market Predictor')

stock=st.text_input("Enter Stock Symbol",'GOOG') #as default

start='2012-01-01'
end='2012-12-21'

data=yf.download(stock,start,end)

st.subheader('Stock Data')
st.write(data)

data_train=pd.DataFrame(data.Close[0:int(len(data)*0.80)])
data_test=pd.DataFrame(data.Close[int(len(data)*0.80):len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

past_100_days=data_train.tail(100)
data_test=pd.concat([past_100_days,data_test],ignore_index=True)
data_test_scale=scaler.fit_transform(data_test)

st.subheader('Price vs MA50')
ma_50_days=data.Close.rolling(50).mean()
fig1=plt.figure(figsize=(10,8))
plt.plot(ma_50_days,'orchid')
plt.plot(data.Close,'r')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days=data.Close.rolling(100).mean()
fig2=plt.figure(figsize=(8,6))
plt.plot(ma_50_days,'darkgoldenrod')
plt.plot(ma_100_days,'navy')
plt.plot(data.Close,'r')
plt.show()
st.pyplot(fig1)

x=[]
y=[]
for i in range(100,data_test_scale.shape[0]):
  x.append(data_test_scale[i-100:i])
  y.append(data_test_scale[i,0])

x,y=np.array(x),np.array(y)
predict=model.predict(x)
scale =1/scaler.scale_

predict=predict*scale
y=y*scale

rmse = np.sqrt(mean_squared_error(y_true, predictions))

st.subheader('Original Price Vs Predicted Price ')
fig4=plt.figure(figsize=(8,6))
plt.plot(predict,'chocolate',label='Original Price')
plt.plot(y,'orangered',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title(f'RMSE: {rmse}')
plt.show()
st.pyplot(fig4)
