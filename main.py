import time
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

import matplotlib
matplotlib.use('TkAgg')

df = pd.read_csv('data.csv', sep=';')  # Verwende das korrekte Trennzeichen ';'
df['ds'] = pd.to_datetime(df['ds'])
df['ds'] = df['ds'].dt.tz_localize(None)
df['y'] = df['y'].abs()

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)


plt.show()
