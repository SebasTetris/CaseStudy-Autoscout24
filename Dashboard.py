import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Dashboard Titel
st.title("AutoScout24 Experten Dashboard")

# Daten laden
file_path = r"D:\DS\3_Projekte\Case Study\autoscout24.csv"
autos_data = pd.read_csv(file_path)

# Überblick über die Daten
st.header("1. Überblick über die Daten")
st.write("### Datensample:")
st.write(autos_data.head())

# Top 5 Hersteller analysieren
st.header("2. Top 5 meistverkaufte Hersteller")
top_5_manufacturers = autos_data['make'].value_counts().head(5)
st.bar_chart(top_5_manufacturers)

# Filter auf Top 5 Hersteller anwenden
top_5_data = autos_data[autos_data['make'].isin(top_5_manufacturers.index)]

# Durchschnittspreis der Top 5 Hersteller
st.header("3. Durchschnittspreise der Top 5 Hersteller")
average_prices = top_5_data.groupby('make')['price'].mean()
st.bar_chart(average_prices)

# Interaktive Filterung nach Hersteller
st.header("4. Daten filtern nach Hersteller")
selected_manufacturer = st.selectbox("Wähle einen Hersteller aus:", top_5_manufacturers.index)
filtered_data = top_5_data[top_5_data['make'] == selected_manufacturer]
st.write(f"### Gefilterte Daten für {selected_manufacturer}:")
st.write(filtered_data)

# Features und Ziel
features = ['mileage', 'hp', 'year']
target = 'price'

X = top_5_data[features]
y = top_5_data[target]

# Fehlende Werte auffüllen
X = X.fillna(X.mean())

# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Modell trainieren
rf_model = RandomForestRegressor(random_state=42, n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=2)
rf_model.fit(X_train, y_train)

# Vorhersagen
y_pred = rf_model.predict(X_test)

# Metriken berechnen
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Modellgüte darstellen
st.header("5. Güte des Random Forest Modells")
st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
st.write(f"**R²-Score:** {r2:.2f}")

# Visualisierung der Modellvorhersagen
st.header("6. Vorhersagen vs. Tatsächliche Preise")
fig, ax = plt.subplots()
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='blue', ax=ax)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
ax.set_title("Vorhersagen vs. Tatsächliche Preise")
ax.set_xlabel("Tatsächliche Preise (€)")
ax.set_ylabel("Vorhergesagte Preise (€)")
st.pyplot(fig)

# Verteilung der Kraftstoffarten
st.header("7. Verteilung der Kraftstoffarten")
fuel_distribution = autos_data['fuel'].value_counts()
fig, ax = plt.subplots()
fuel_distribution.plot(kind='bar', color='orange', edgecolor='black', ax=ax)
ax.set_title("Verteilung der Kraftstoffarten")
ax.set_xlabel("Kraftstoffart")
ax.set_ylabel("Anzahl der Fahrzeuge")
st.pyplot(fig)

# Preisverteilung für ausgewählte Hersteller
st.header("8. Preisverteilung für ausgewählten Hersteller")
if not filtered_data.empty:
    fig, ax = plt.subplots()
    sns.histplot(filtered_data['price'], bins=30, kde=True, color='green', ax=ax)
    ax.set_title(f"Preisverteilung für {selected_manufacturer}")
    ax.set_xlabel("Preis (€)")
    ax.set_ylabel("Häufigkeit")
    st.pyplot(fig)

# Option zum Download von gefilterten Daten
st.header("9. Datenexport")
csv_data = filtered_data.to_csv(index=False)
st.download_button(
    label="Gefilterte Daten als CSV herunterladen",
    data=csv_data,
    file_name=f"{selected_manufacturer}_data.csv",
    mime='text/csv'
)
