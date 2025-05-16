import streamlit as st
import pandas as pd 
import numpy as np 
from keras.models import load_model
import yfinance as yf
import matplotlib.pyplot as plt


st.title("Stock Price Predictor App")

stock= st.text_input("Enter the Stock ID","GOOG")

from datetime import datetime
end = datetime.now()
start= datetime(end.year-20,end.month,end.day)

df=yf.download(stock, start, end)

model = load_model("Latest_stack_price_model.keras")


st.subheader("Stock Data")
st.write(df)

splitting_length = int(len(df)*0.7)
x_test = df[['Close']][splitting_length:]



def plot_gragh(figsize, values, full_data,extra_data =0,extra_dataset = None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values,'Orange')
    plt.plot(full_data.Close,'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

st.subheader('Original Close Price and MA for 250 days')
df['MA_for_250_days']=df.Close.rolling(250).mean()
st.pyplot(plot_gragh((15,6), df['MA_for_250_days'], df,0))

st.subheader('Original Close Price and MA for 200 days')
df['MA_for_200_days']=df.Close.rolling(200).mean()
st.pyplot(plot_gragh((15,6), df['MA_for_200_days'], df,0))

st.subheader('Original Close Price and MA for 100 days')
df['MA_for_100_days']=df.Close.rolling(100).mean()
st.pyplot(plot_gragh((15,6), df['MA_for_100_days'], df,0))

st.subheader('Original Close Price and MA for 100 daysand MA for 250 days')
df['MA_for_250_days']=df.Close.rolling(250).mean()
st.pyplot(plot_gragh((15,6), df['MA_for_250_days'], df,1,df['MA_for_250_days']))

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
st.write("x_test columns:", x_test.columns)
scaled_data = scaler.fit_transform(x_test[['Close']])


x_data = []
y_data =[]

for i in range(100,len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_data)

inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

plot_index = df.index[splitting_length + 100 : splitting_length + 100 + len(inv_pre)]

ploting_data = pd.DataFrame(
    {
        'original_test_data': inv_y_test.reshape(-1),
        'predicted_data': inv_pre.reshape(-1)
    },
    index=plot_index
)

st.subheader('Original values vs predicted values' )
st.write(ploting_data)

st.subheader('Original close price vs predicted close price')
fig= plt.figure(figsize=(15,6))
plt.plot(pd.concat([df.Close[:splitting_length+100],ploting_data], axis=0))
plt.legend(["Data- not used", "Original Test data", "Predicted Test data"])
st.pyplot(fig)

