# Walkthrough: Simulación de Auditoría Predictiva GreenMetric

Hemos diseñado y construido el módulo de simulación predictiva de **Energía y Cambio Climático** basándonos en los datos climáticos e históricos del campus real de la Universidad de La Trobe de Kaggle.

---

## Componentes Implementados

### 1. Script de Consolidación y Machine Learning
*   **Archivo:** [preprocesar_y_entrenar.py](file:///c:/Users/Usuario/Documents/ProyectoSimulacionBackend/preprocesar_y_entrenar.py)
*   **Función:** 
    *   Lee los archivos de Kaggle en caché y los agrupa a nivel diario para optimizar memoria.
    *   Cruza las tablas de consumo de edificios, consumos de gas, calendario académico y clima.
    *   Entrena un modelo de inteligencia artificial (`RandomForestRegressor`) para aprender cómo influye el clima y la ocupación en el consumo de los 64 edificios.
*   **Resultados de Entrenamiento:**
    *   Total de datos diarios consolidados: **88,766 registros**.
    *   Precisión en datos de entrenamiento: **97.50%**.
    *   Precisión en datos de prueba: **96.36%** (R² score extremadamente alto y preciso).
    *   Guarda el modelo entrenado y el dataset ligero en [modelo_y_datos_simulacion.pkl](file:///c:/Users/Usuario/Documents/ProyectoSimulacionBackend/modelo_y_datos_simulacion.pkl).

### 2. Dashboard Interactivo y Visual
*   **Archivo:** [simulador_greenmetric.py](file:///c:/Users/Usuario/Documents/ProyectoSimulacionBackend/simulador_greenmetric.py)
*   **Función:** Interfaz gráfica nativa de Windows que permite configurar variables climáticas y operacionales y ver de inmediato el comportamiento predictivo del modelo.

---

## Guía de Uso del Dashboard

Al abrirse el panel en tu escritorio, verás dos secciones:

### Panel de Controles (Izquierda)
*   **Filtros de Campus y Categoría:** Te permiten aislar la visualización por Campus (`1`, `2` o `3`) o por tipos de uso de edificios (ej. Aulas, Oficinas, Laboratorios).
*   **Slider de Temperatura ($\Delta T$):** Modifica la temperatura simulada. Verás cómo subir la temperatura altera el consumo debido a la mayor demanda de refrigeración.
*   **Slider Ahorro HVAC (%):** Simula la implementación de aires acondicionados eficientes o automatización de apagado.
*   **Slider Solar (%):** Simula la instalación de paneles fotovoltaicos disminuyendo el consumo neto tomado de la red eléctrica.
*   **Slider de Ocupación (Asistencia):** Aumenta o disminuye el flujo de estudiantes en el campus.

### Visualizaciones y Métricas (Derecha)
*   **Consumo Mensual Predictivo (MWh):** Línea roja (histórico real) vs Línea verde (predicción simulada).
*   **Emisiones de CO2 Totales (toneladas):** Gráfico de barras que muestra la reducción de huella de carbono tras aplicar las medidas de sostenibilidad.
*   **Scorecard GreenMetric:** Calcula y proyecta en tiempo real el puntaje que obtendría el campus en la categoría de energía de **UI GreenMetric** (hasta 2100 puntos).

---

## Cómo Ejecutar el Módulo

Si cierras el simulador y deseas iniciarlo de nuevo, ejecuta la siguiente instrucción en tu consola dentro del proyecto:

```powershell
.\venv\Scripts\python simulador_greenmetric.py
```
