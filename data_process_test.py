from data_process import process_data
import pandas as pd

df_path = './data/train.csv' 
df = pd.read_csv(df_path)

df = process_data(df)

df.to_csv('data/processed_data.csv')