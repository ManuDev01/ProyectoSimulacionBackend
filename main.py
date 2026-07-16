import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Cargar los datos
# Asegúrate de que tu CSV tenga las columnas que vas a usar
df = pd.read_csv('clima.csv')

# 2. Separar características (X) y el objetivo a predecir (y)
# Supongamos que usamos Humedad y Presión para predecir la Temperatura
X = df[['Humedad', 'Presion']]
y = df['Temperatura']

# 3. Dividir los datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Crear y entrenar el modelo de Machine Learning
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# 5. Realizar predicciones con los datos de prueba
predicciones = modelo.predict(X_test)

# 6. Evaluar qué tan bueno es el modelo
error = mean_squared_error(y_test, predicciones)
print(f"Error Cuadrático Medio: {error:.2f}")

# 7. Graficar los resultados de la comparación
plt.scatter(y_test, predicciones, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Temperatura Real')
plt.ylabel('Temperatura Predicha')
plt.title('Comparación de Clima: Real vs Predicho')
plt.show()
