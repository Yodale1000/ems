import pandas as pd

# CSV-Dateien einlesen
print("--reading--")
df_phase_a = pd.read_csv('phase_a.csv', skiprows=3)
print(f"Phase a: {len(df_phase_a)} Zeilen")
df_phase_b = pd.read_csv('phase_b.csv', skiprows=3)
print(f"Phase b: {len(df_phase_b)} Zeilen")
df_phase_c = pd.read_csv('phase_c.csv', skiprows=3)
print(f"Phase c: {len(df_phase_c)} Zeilen")

# Relevante Spalten auswählen und umbenennen
print("--collecting and renaming--")
df_phase_a = df_phase_a[['_time', '_value']].rename(columns={'_time': 'ds', '_value': 'y_phase_a'})
df_phase_b = df_phase_b[['_time', '_value']].rename(columns={'_time': 'ds', '_value': 'y_phase_b'})
df_phase_c = df_phase_c[['_time', '_value']].rename(columns={'_time': 'ds', '_value': 'y_phase_c'})

# Zeitstempel formatieren und auf Minuten runden
print("--formatting--")
df_phase_a['ds'] = pd.to_datetime(df_phase_a['ds'], format='mixed').dt.floor('min')
df_phase_b['ds'] = pd.to_datetime(df_phase_b['ds'], format='mixed').dt.floor('min')
df_phase_c['ds'] = pd.to_datetime(df_phase_c['ds'], format='mixed').dt.floor('min')

# Zusammenführen der Phasen mit concat
print("--concatenating--")
df = pd.concat([df_phase_a, df_phase_b, df_phase_c], axis=0)

# Gruppieren nach Zeitstempel und Aggregieren
df = df.groupby('ds').agg({
    'y_phase_a': 'sum',  # Behalte die Werte von Phase A
    'y_phase_b': 'sum',  # Behalte die Werte von Phase B
    'y_phase_c': 'sum'   # Behalte die Werte von Phase C
}).reset_index()

df = df.resample('h').mean().reset_index()

# Neue Spalte 'y' für die Summe der Phasen-Werte hinzufügen
df['y'] = df['y_phase_a'] + df['y_phase_b'] + df['y_phase_c']
df = df.fillna(0)
df['y'] = df['y'].round().astype(int)

# Ausgabe der ersten Zeilen des DataFrames
print(f"Summed: {len(df)} Zeilen")
print(df.head())

# Speichern als CSV
df.to_csv('cleaned_consumption_bp.csv', index=False)
