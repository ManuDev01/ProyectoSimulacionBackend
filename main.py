# Importación de librerías para la descarga de Kaggle
import kagglehub
from kagglehub import KaggleDatasetAdapter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("=== DESCARGANDO DATASET DE KAGGLE ===")
# 1. Cargar el archivo de clima del dataset 'unicon' de CDAC Lab
# Indicamos específicamente el archivo weather_data.csv contenido en el dataset
df_kaggle = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "cdaclab/unicon",
    "weather_data.csv"
)

# Imprimir las primeras filas y columnas disponibles para validar la carga
print("\nColumnas descargadas de Kaggle:", df_kaggle.columns.tolist())
print("Primeros registros de Kaggle:\n", df_kaggle.head())

# =====================================================================
# NOTA DE ADAPTACIÓN:
# El dataset real de Kaggle 'weather_data.csv' contiene estas columnas:
# ['campus_id', 'timestamp', 'apparent_temperature', 'relative_humidity', 'wind_speed', 'wind_direction']
# Para mantener la simulación de URBE de tu taller Green Metric usando datos de un entorno universitario real,
# usaremos 'relative_humidity' (Humedad) y 'wind_speed' (Viento) para predecir 'apparent_temperature' (Clima).
# =====================================================================

# 2. Selección de variables y limpieza de datos nulos
df_clean = df_kaggle[['relative_humidity', 'wind_speed', 'apparent_temperature']].dropna()

# Reducimos un poco el dataset para no sobrecargar la memoria en el entrenamiento
df_clean = df_clean.sample(n=min(5000, len(df_clean)), random_state=42)

# 3. Separar características (X) y variable objetivo (y)
X = df_clean[['relative_humidity', 'wind_speed']]
y = df_clean['apparent_temperature']

# Renombramos las columnas internamente para que la salida visual sea legible en español
X.columns = ['Humedad_Relativa', 'Velocidad_Viento']

# 4. Dividir los datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Crear y entrenar el modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# 6. Realizar predicciones con el conjunto de prueba
predicciones = modelo.predict(X_test)

# 7. Evaluación del modelo
mse = mean_squared_error(y_test, predicciones)
r2 = r2_score(y_test, predicciones)

print("\n=== EVALUACIÓN DEL MODELO CON DATOS REALES (La Trobe Campus / GreenMetric) ===")
print(f"Error Cuadrático Medio (MSE): {mse:.4f}")
print(f"Precisión del modelo (R² Score): {r2 * 100:.2f}%")

# Mostrar la importancia de cada criterio climático
print("\n=== IMPORTANCIA DE CADA VARIABLE (COEFICIENTES) ===")
for col, coef in zip(X.columns, modelo.coef_):
    print(f"- {col}: {coef:.4f}")

# 8. Graficar los resultados
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predicciones, color='dodgerblue', alpha=0.5, label='Muestras de Clima Real')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Predicción Perfecta')
plt.xlabel('Temperatura Aparente Real (°C)')
plt.ylabel('Temperatura Aparente Predicha (°C)')
plt.title('Predicción del Clima Universitario con Datos de Kaggle')
plt.legend()
plt.grid(True)
plt.show()