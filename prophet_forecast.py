import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import os

# Erstelle das Verzeichnis, falls es nicht existiert
os.makedirs('plots/prophet', exist_ok=True)

df = pd.read_csv('data.csv', sep=';')
df['ds'] = pd.to_datetime(df['ds'])
df['ds'] = df['ds'].dt.tz_localize(None)

# Fehlende Werte überprüfen und handhaben
df = df.dropna()

# Prophet Modell initialisieren und trainieren
m = Prophet()
m.fit(df)

# Zukünftige Daten für den nächsten Tag erstellen
last_date = df['ds'].max()
future = m.make_future_dataframe(periods=7*24*60, freq='min')  # Periode ist hier für alle Minuten im nächsten Tag

# Vorhersage
forecast = m.predict(future)

# Daten filtern, um nur den nächsten Tag anzuzeigen
forecast_next_day = forecast[forecast['ds'] > last_date]

# Plotting der Vorhersage für den nächsten Tag
plt.figure(figsize=(10, 6))
plt.plot(forecast_next_day['ds'], forecast_next_day['yhat'], label='Predicted')
plt.fill_between(forecast_next_day['ds'], forecast_next_day['yhat_lower'], forecast_next_day['yhat_upper'], alpha=0.2, label='Confidence Interval')
plt.xlabel('Date')
plt.ylabel('Predicted Value')
plt.title('Prediction for the Next Day')
plt.legend()

# Save the plot
next_day_plot_path = 'plots/prophet/forecast_next_day.png'
plt.savefig(next_day_plot_path)

plt.show()

print(f"Plot for next day's prediction saved at: {next_day_plot_path}")
