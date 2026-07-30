import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import kagglehub
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Ruta del dataset de Kaggle en caché local
KAGGLE_PATH = os.path.join(BASE_DIR, "data")

# =====================================================================
# NUEVO: DESCARGA AUTOMÁTICA DE LOS DATOS SI NO EXISTEN
# =====================================================================
if not os.path.exists(os.path.join(KAGGLE_PATH, "building_meta.csv")):
    print("La carpeta 'data' está vacía o incompleta. Descargando dataset de Kaggle...")
    try:

        
        # Descarga el dataset a la caché global de kagglehub
        cache_dir = kagglehub.dataset_download('cdaclab/unicon')
        
        # Crea la carpeta data del proyecto si no existe
        os.makedirs(KAGGLE_PATH, exist_ok=True)
        
        # Copia todos los archivos CSV descargados a tu carpeta data/
        for archivo in os.listdir(cache_dir):
            if archivo.endswith('.csv'):
                shutil.copy(os.path.join(cache_dir, archivo), KAGGLE_PATH)
                
        # Corrección por si el archivo viene nombrado con 'a' (calendar) en vez de 'e' (calender)
        src_cal = os.path.join(KAGGLE_PATH, "calendar.csv")
        dst_cal = os.path.join(KAGGLE_PATH, "calender.csv")
        if os.path.exists(src_cal) and not os.path.exists(dst_cal):
            shutil.copy(src_cal, dst_cal)
            
        print("¡Descarga completada con éxito en la carpeta 'data/'!")
    except Exception as e:
        print(f"Error crítico al descargar de Kaggle: {e}")
        print("Por favor, asegúrate de tener internet e instalar kagglehub ('pip install kagglehub').")
        exit(1)
# =====================================================================


print("=== INICIANDO PREPROCESAMIENTO DE DATOS ===")

# 1. Cargar metadatos de los edificios
print("Cargando building_meta.csv...")
df_meta = pd.read_csv(os.path.join(KAGGLE_PATH, "building_meta.csv"))

# 2. Cargar y agregar el consumo de edificios a nivel diario
print("Cargando y procesando building_consumption.csv (esto puede tomar unos segundos)...")
# Cargamos solo las columnas necesarias para optimizar memoria
df_cons = pd.read_csv(os.path.join(KAGGLE_PATH, "building_consumption.csv"), usecols=['campus_id', 'meter_id', 'timestamp', 'consumption'])
df_cons['date'] = pd.to_datetime(df_cons['timestamp']).dt.date
df_cons_daily = df_cons.groupby(['campus_id', 'meter_id', 'date'], as_index=False)['consumption'].sum()
del df_cons # Liberar memoria

# 3. Cargar y agregar datos climáticos a nivel diario
print("Cargando y procesando weather_data.csv...")
df_weather = pd.read_csv(os.path.join(KAGGLE_PATH, "weather_data.csv"), 
                         usecols=['campus_id', 'timestamp', 'apparent_temperature', 'relative_humidity', 'wind_speed'])
df_weather['date'] = pd.to_datetime(df_weather['timestamp']).dt.date
df_weather_daily = df_weather.groupby(['campus_id', 'date'], as_index=False).agg({
    'apparent_temperature': 'mean',
    'relative_humidity': 'mean',
    'wind_speed': 'mean'
})
del df_weather # Liberar memoria

# 4. Cargar y agregar consumo de gas
print("Cargando y procesando gas_consumption.csv...")
df_gas = pd.read_csv(os.path.join(KAGGLE_PATH, "gas_consumption.csv"))
df_gas['date'] = pd.to_datetime(df_gas['timestamp']).dt.date
df_gas_daily = df_gas.groupby(['campus_id', 'date'], as_index=False)['consumption'].sum()
df_gas_daily.rename(columns={'consumption': 'gas_consumption'}, inplace=True)

# 5. Cargar calendario académico
print("Cargando calender.csv...")
df_cal = pd.read_csv(os.path.join(KAGGLE_PATH, "calender.csv"))
df_cal['date'] = pd.to_datetime(df_cal['date']).dt.date

# 6. Integración (Merge) de los datos
print("Consolidando los conjuntos de datos...")
# Unir consumos y metadatos de edificios (meter_id de consumos coincide con id de edificio)
df_merged = pd.merge(df_cons_daily, df_meta, left_on=['campus_id', 'meter_id'], right_on=['campus_id', 'id'], how='inner')

# Unir clima (por campus y fecha)
df_merged = pd.merge(df_merged, df_weather_daily, on=['campus_id', 'date'], how='inner')

# Unir gas (por campus y fecha)
df_merged = pd.merge(df_merged, df_gas_daily, on=['campus_id', 'date'], how='left')
df_merged['gas_consumption'] = df_merged['gas_consumption'].fillna(0) # Si no hay gas, es 0

# Unir calendario académico (por fecha)
df_merged = pd.merge(df_merged, df_cal, on=['date'], how='left')
# Rellenar valores nulos del calendario
for col in ['is_holiday', 'is_semester', 'is_exam']:
    if col in df_merged.columns:
        df_merged[col] = df_merged[col].fillna(0).astype(int)

# 7. Extracción de características temporales adicionales
df_merged['date_dt'] = pd.to_datetime(df_merged['date'])
df_merged['month'] = df_merged['date_dt'].dt.month
df_merged['day_of_week'] = df_merged['date_dt'].dt.dayofweek
df_merged['year'] = df_merged['date_dt'].dt.year

# Calcular antigüedad del edificio (suponiendo año de auditoría = max(built_year) o 2022 si built_year es NaN)
median_built_year = df_merged['built_year'].median()
if pd.isna(median_built_year):
    median_built_year = 2000
df_merged['built_year'] = df_merged['built_year'].fillna(median_built_year)
df_merged['building_age'] = df_merged['year'] - df_merged['built_year']
df_merged['building_age'] = np.clip(df_merged['building_age'], 0, None)

print(f"Total de registros consolidados: {len(df_merged)}")

# =====================================================================
# ENTRENAMIENTO DEL MODELO DE INTELIGENCIA PREDICTIVA
# =====================================================================
print("\n=== ENTRENANDO EL MODELO PREDICTIVO (RANDOM FOREST) ===")

# Definir características (features) y variable objetivo (target)
features = [
    'apparent_temperature', 'relative_humidity', 'wind_speed',
    'gross_floor_area', 'capacity', 'building_age',
    'is_holiday', 'is_semester', 'is_exam',
    'month', 'day_of_week', 'category'
]
target = 'consumption'

# Limpieza de nulos en las características numéricas críticas
num_features = ['apparent_temperature', 'relative_humidity', 'wind_speed', 'gross_floor_area', 'capacity', 'building_age']
cat_features = ['category']

# Crear Pipeline de preprocesamiento
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), num_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_features)
    ],
    remainder='passthrough'
)

# Pipeline completo con el regresor RandomForest (limitar tamaño para ejecución rápida y liviana)
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=30, max_depth=12, random_state=42, n_jobs=-1))
])

# Filtrar nulos en el target para entrenar
df_train_clean = df_merged.dropna(subset=[target])

# Dividir en entrenamiento y prueba
X = df_train_clean[features]
y = df_train_clean[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el pipeline
print("Ajustando RandomForestRegressor...")
model_pipeline.fit(X_train, y_train)

# Evaluar el modelo
train_r2 = model_pipeline.score(X_train, y_train)
test_r2 = model_pipeline.score(X_test, y_test)
print(f"Precisión del modelo en Entrenamiento (R²): {train_r2*100:.2f}%")
print(f"Precisión del modelo en Prueba (R²): {test_r2*100:.2f}%")

# =====================================================================
# GUARDADO DE RESULTADOS PARA EL DASHBOARD
# =====================================================================
print("\n=== GUARDANDO MODELO Y DATOS AGREGADOS ===")

# Guardamos una versión reducida de los datos reales consolidados para usar de línea base en el simulador
# Seleccionamos las columnas útiles para que el archivo sea súper ligero
cols_to_save = [
    'campus_id', 'meter_id', 'date', 'consumption', 'gas_consumption',
    'apparent_temperature', 'relative_humidity', 'wind_speed',
    'gross_floor_area', 'capacity', 'building_age',
    'is_holiday', 'is_semester', 'is_exam',
    'month', 'day_of_week', 'category'
]
df_save = df_merged[cols_to_save].copy()

# Guardar en formato pickle
save_data = {
    'model': model_pipeline,
    'data': df_save,
    'features': features
}

output_file = "modelo_y_datos_simulacion.pkl"
with open(output_file, 'wb') as f:
    pickle.dump(save_data, f)

print(f"¡Preprocesamiento completado con éxito! Archivo guardado como '{output_file}'.")
