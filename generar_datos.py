import pandas as pd
import numpy as np

# Generar 500 datos simulados realistas
np.random.seed(42)
n_muestras = 500

# Humedad entre 30% y 90%
humedad = np.random.uniform(30, 90, n_muestras)
# Presión atmosférica en hPa (promedio 1013)
presion = np.random.uniform(990, 1025, n_muestras)
# La temperatura disminuye un poco con la humedad y varía de forma aleatoria
temperatura = 35 - (humedad * 0.15) - ((presion - 1000) * 0.05) + np.random.normal(0, 2, n_muestras)

# Crear el DataFrame y guardarlo
df = pd.DataFrame({
'Humedad': humedad,
'Presion': presion,
'Temperatura': temperatura
})

df.to_csv('clima.csv', index=False)
print("¡Archivo 'clima.csv' generado con éxito con 500 registros!")
