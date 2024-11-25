import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

# Datei einlesen
file_path = r"D:\DS\3_Projekte\Case Study\autoscout24.csv"
autos_data = pd.read_csv(file_path)

# 1. Die fünf meistverkauften Hersteller ermitteln
top_5_manufacturers = autos_data['make'].value_counts().head(5)
print("Top 5 meistverkaufte Hersteller:")
print(top_5_manufacturers)

# Daten filtern, um nur die Top 5 Hersteller zu betrachten
top_5_data = autos_data[autos_data['make'].isin(top_5_manufacturers.index)]

# 2. Durchschnittspreis eines Autos von Hersteller X
average_prices = top_5_data.groupby('make')['price'].mean()
print("\nDurchschnittspreise der Top 5 Hersteller:")
print(average_prices)

# 3. Modelltraining: Preis vorhersagen (Lineare Regression)
# Relevante Features auswählen
features = ['mileage', 'hp', 'year']  # Beispiel für relevante Features
target = 'price'

# Daten vor dem Modelltraining vorbereiten
X = top_5_data[features]
y = top_5_data[target]

# Fehlende Werte durch den Mittelwert auffüllen (Imputation)
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Daten in Trainings- und Test-Sets aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lineare Regression trainieren
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Vorhersagen
y_pred = lin_reg.predict(X_test)

# Güte des Modells bewerten
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nGüte des Modells (Lineare Regression):")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R²-Score: {r2:.2f}")

# 4. Entscheidungsbaum ausprobieren
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)
y_tree_pred = tree_model.predict(X_test)

# Fehler für Entscheidungsbaum berechnen
tree_mae = mean_absolute_error(y_test, y_tree_pred)
tree_r2 = r2_score(y_test, y_tree_pred)
print("\nGüte des Modells (Entscheidungsbaum):")
print(f"Mean Absolute Error (MAE): {tree_mae:.2f}")
print(f"R²-Score: {tree_r2:.2f}")

# 5. Visualisierung der Vorhersagen (Lineare Regression vs. Tatsächliche Preise)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue', label='Lineare Regression')
plt.scatter(y_test, y_tree_pred, alpha=0.6, color='green', label='Entscheidungsbaum')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Perfekte Vorhersage')
plt.title("Vorhersagen vs. Tatsächliche Preise")
plt.xlabel("Tatsächliche Preise (€)")
plt.ylabel("Vorhergesagte Preise (€)")
plt.legend()
plt.tight_layout()
plt.show()



from sklearn.ensemble import RandomForestRegressor

# Random Forest trainieren
rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

# Vorhersagen
y_rf_pred = rf_model.predict(X_test)

# Güte des Random Forest-Modells
rf_mae = mean_absolute_error(y_test, y_rf_pred)
rf_r2 = r2_score(y_test, y_rf_pred)

print("\nGüte des Modells (Random Forest):")
print(f"Mean Absolute Error (MAE): {rf_mae:.2f}")
print(f"R²-Score: {rf_r2:.2f}")




import matplotlib.pyplot as plt

# Vergleich der Modellgüte
models = ['Lineare Regression', 'Entscheidungsbaum', 'Random Forest']
mae_scores = [3054.49, 2493.30, 2145.11]
r2_scores = [0.76, 0.77, 0.85]

# Barplot für MAE
plt.figure(figsize=(10, 5))
plt.bar(models, mae_scores, color=['blue', 'green', 'orange'], edgecolor='black')
plt.title("Mean Absolute Error (MAE) Vergleich", fontsize=14)
plt.xlabel("Modelle", fontsize=12)
plt.ylabel("MAE (Euro)", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Barplot für R²-Score
plt.figure(figsize=(10, 5))
plt.bar(models, r2_scores, color=['blue', 'green', 'orange'], edgecolor='black')
plt.title("R²-Score Vergleich", fontsize=14)
plt.xlabel("Modelle", fontsize=12)
plt.ylabel("R²-Score", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()





# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import GridSearchCV

# # Hyperparameter-Raster für die Grid-Suche
# param_grid = {
#     'n_estimators': [50, 100, 200],       # Anzahl der Bäume im Wald
#     'max_depth': [10, 20, None],          # Maximale Tiefe der Bäume
#     'min_samples_split': [2, 5, 10],      # Mindestanzahl von Samples für einen Split
#     'min_samples_leaf': [1, 2, 4],        # Mindestanzahl von Samples in einem Blatt
#     'max_features': ['auto', 'sqrt']      # Anzahl der Features pro Split
# }

# # Random Forest Regressor
# rf = RandomForestRegressor(random_state=42)

# # GridSearchCV
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
#                            cv=3, scoring='neg_mean_absolute_error', 
#                            n_jobs=-1, verbose=2)

# # Fitting der Grid-Suche
# grid_search.fit(X_train, y_train)

# # Beste Parameter und Modellleistung
# best_params = grid_search.best_params_
# best_score = -grid_search.best_score_  # Negative MAE, daher das Vorzeichen umkehren

# print("\nBeste Hyperparameter:")
# print(best_params)
# print(f"\nBeste durchschnittliche MAE aus GridSearchCV: {best_score:.2f}")

# # Modell mit besten Parametern
# best_rf_model = grid_search.best_estimator_

# # Vorhersagen auf dem Testset
# y_best_rf_pred = best_rf_model.predict(X_test)

# # Bewertung auf dem Testset
# best_rf_mae = mean_absolute_error(y_test, y_best_rf_pred)
# best_rf_r2 = r2_score(y_test, y_best_rf_pred)

# print("\nGüte des optimierten Modells (Random Forest):")
# print(f"Mean Absolute Error (MAE): {best_rf_mae:.2f}")
# print(f"R²-Score: {best_rf_r2:.2f}")