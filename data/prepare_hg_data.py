import pandas as pd

df = pd.read_csv('consumption_hg.csv', sep=';', thousands=',')
df['ds'] = pd.to_datetime(df['ds'], format='mixed')
df.set_index('ds', inplace=True)
df = df.resample('h').mean().reset_index()
df = df.fillna(0)
df['y'] = df['y'].round().astype(int)

df.to_csv('cleaned_consumption_hg.csv', index=False)