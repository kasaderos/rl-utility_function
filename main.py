from data_process import process_data
import pandas as pd
import matplotlib.pyplot as plt
from model import train, predict, get_X, get_y
import config as conf

df_path = 'data/train.csv' 
df = pd.read_csv(df_path)

df = process_data(df)

model = train(df)

test_df_path = 'data/test.csv'
test_df = pd.read_csv(test_df_path)
test_df = process_data(test_df)

pred = predict(model, test_df)
y = test_df['Close'].iloc[conf.past+conf.h:].to_numpy()

### visualize
x = pd.to_datetime(df['Date'], format='%Y-%m-%d')
plt.figure(figsize=(10, 4))
plt.plot(x, df['Close'])
plt.savefig('plots/train.png')

plt.figure(figsize=(10, 6))

h = conf.h
for i in range(len(y) - h):
    price = y[i]
    future_price = y[i + h]
    expected_price = y[i] * (1+pred[i])
    if pred[i] > 0.01 and future_price >= y[i]:
        plt.scatter(i, y[i], color='green', marker='o')
        print('GREEN', pred[i], price, future_price, expected_price)
    elif pred[i] > 0.01 and future_price < y[i]:
        plt.scatter(i, y[i], color='red', marker='o')
        print('RED  ', pred[i], price, future_price, expected_price)

plt.plot(y)
plt.savefig('plots/prediction.png')