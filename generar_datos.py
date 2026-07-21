import pandas as pd
import numpy as np

# Semilla para que los resultados sean reproducibles
np.random.seed(42)
n_universidades = 500000

# Generar los 6 criterios del UI GreenMetric (puntuaciones de 0 a 100)
entorno_infraestructura = np.random.uniform(40, 95, n_universidades)
energia_cambio_climatico = np.random.uniform(30, 90, n_universidades)
residuos = np.random.uniform(35, 95, n_universidades)
agua = np.random.uniform(40, 90, n_universidades)
transporte = np.random.uniform(25, 85, n_universidades)
educacion_investigacion = np.random.uniform(50, 98, n_universidades)

# El Puntaje Total (Target) dependerá de los criterios anteriores + una pequeña variación aleatoria (ruido)
# Simulamos pesos reales del ranking (Energía y Residuos suelen pesar bastante en la fórmula)
puntaje_total = (
    (entorno_infraestructura * 0.15) +
    (energia_cambio_climatico * 0.21) +
    (residuos * 0.18) +
    (agua * 0.10) +
    (transporte * 0.18) +
    (educacion_investigacion * 0.18) +
    np.random.normal(0, 1.5, n_universidades)  # Ruido del modelo
)

# Limitar el puntaje final a un máximo teórico de 100
puntaje_total = np.clip(puntaje_total, 0, 100)

# Crear el DataFrame y guardarlo
df = pd.DataFrame({
    'Entorno_Infraestructura': entorno_infraestructura,
    'Energia_Clima': energia_cambio_climatico,
    'Residuos': residuos,
    'Agua': agua,
    'Transporte': transporte,
    'Educacion_Investigacion': educacion_investigacion,
    'Puntaje_Sostenibilidad': puntaje_total
})

# Guardar en el archivo que leerá el modelo
df.to_csv('clima.csv', index=False)
print("¡Archivo 'clima.csv' (UI GreenMetric - URBE) generado con éxito con 300 registros!")
