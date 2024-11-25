import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Datei einlesen
file_path = r"D:\DS\3_Projekte\Case Study\autoscout24.csv"
autos_data = pd.read_csv(file_path)

# Überblick über die Daten
print(f"Anzahl der verkauften Autos: {len(autos_data)}")
print(f"Zeitraum: {autos_data['year'].min()} bis {autos_data['year'].max()}")
print(f"Anzahl der Marken: {autos_data['make'].nunique()}")
print("Marken:", autos_data['make'].unique())

### Verbesserte Fragestellungen und Analysen

# 1. Verteilung der PS (Pferdestärken) nach Marke
plt.figure(figsize=(12, 6))
ps_verteilung = autos_data.groupby('make')['hp'].mean().sort_values(ascending=False)
ps_verteilung[:15].plot(kind='bar', color='skyblue', edgecolor='black')  # Top 15 Marken
plt.title("Durchschnittliche PS nach Marke (Top 15)", fontsize=14)
plt.xlabel("Marke", fontsize=12)
plt.ylabel("Durchschnittliche PS", fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 2. Durchschnittlicher Preis nach Marke und Modell
price_by_make_model = autos_data.groupby(['make', 'model'])['price'].mean().reset_index()
price_by_make_model_top = price_by_make_model.sort_values(by='price', ascending=False).head(20)

plt.figure(figsize=(14, 8))
sns.barplot(data=price_by_make_model_top, x='price', y='make', hue='model', dodge=False)
plt.title("Durchschnittlicher Preis nach Marke und Modell (Top 20)", fontsize=14)
plt.xlabel("Durchschnittlicher Preis (€)", fontsize=12)
plt.ylabel("Marke", fontsize=12)
plt.legend(title="Modell", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 3. Verteilung der Kraftstoffarten
fuel_verteilung = autos_data['fuel'].value_counts()
plt.figure(figsize=(10, 6))
fuel_verteilung.plot(kind='bar', color='orange', edgecolor='black')
plt.title("Verteilung der Kraftstoffarten", fontsize=14)
plt.xlabel("Kraftstoffart", fontsize=12)
plt.ylabel("Anzahl der Fahrzeuge", fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 4. Einfluss der Kraftstoffarten auf den Preis
fuel_price_influence = autos_data.groupby('fuel')['price'].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
fuel_price_influence.plot(kind='bar', color='lightblue', edgecolor='black')
plt.title("Durchschnittlicher Preis nach Kraftstoffart", fontsize=14)
plt.xlabel("Kraftstoffart", fontsize=12)
plt.ylabel("Durchschnittlicher Preis (€)", fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 5. Veränderungen über die Jahre (Preis, Anzahl der Autos)
autos_by_year = autos_data.groupby('year').agg({'price': 'mean', 'mileage': 'mean', 'year': 'count'}).rename(columns={'year': 'count'})
autos_by_year.reset_index(inplace=True)

plt.figure(figsize=(10, 6))
plt.plot(autos_by_year['year'], autos_by_year['price'], marker='o', color='green')
plt.title("Durchschnittlicher Preis pro Jahr", fontsize=14)
plt.xlabel("Jahr", fontsize=12)
plt.ylabel("Durchschnittlicher Preis (€)", fontsize=12)
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(autos_by_year['year'], autos_by_year['count'], color='coral', edgecolor='black')
plt.title("Anzahl der Autos pro Jahr", fontsize=14)
plt.xlabel("Jahr", fontsize=12)
plt.ylabel("Anzahl der Autos", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

### Zusätzliche Analysen (Expert-Level)

# 6. Scatterplot: Preis vs. PS (Wie beeinflussen PS den Preis?)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=autos_data, x='hp', y='price', hue='fuel', alpha=0.6)
plt.title("Preis vs. PS nach Kraftstoffart", fontsize=14)
plt.xlabel("PS", fontsize=12)
plt.ylabel("Preis (€)", fontsize=12)
plt.legend(title="Kraftstoffart")
plt.grid(alpha=0.7)
plt.tight_layout()
plt.show()

# 7. Histogramm: Verteilung der Kilometerstände
plt.figure(figsize=(10, 6))
autos_data['mileage'].plot(kind='hist', bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.title("Verteilung der Kilometerstände", fontsize=14)
plt.xlabel("Kilometerstand", fontsize=12)
plt.ylabel("Anzahl der Fahrzeuge", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
