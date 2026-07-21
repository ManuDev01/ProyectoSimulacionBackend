import os
import pickle
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Configurar estilo global de Matplotlib para fondo oscuro
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1A1A24'
plt.rcParams['axes.facecolor'] = '#262636'
plt.rcParams['text.color'] = '#FFFFFF'
plt.rcParams['axes.labelcolor'] = '#FFFFFF'
plt.rcParams['xtick.color'] = '#B0B0C0'
plt.rcParams['ytick.color'] = '#B0B0C0'

# Cargar el modelo y los datos preprocesados
PKL_FILE = "modelo_y_datos_simulacion.pkl"

if not os.path.exists(PKL_FILE):
    raise FileNotFoundError(f"No se encontró el archivo '{PKL_FILE}'. Ejecuta primero 'preprocesar_y_entrenar.py'.")

with open(PKL_FILE, 'rb') as f:
    saved_bundle = pickle.load(f)

model = saved_bundle['model']
df_base = saved_bundle['data']
features_list = saved_bundle['features']

# Asegurar tipo de fecha correcto
df_base['date'] = pd.to_datetime(df_base['date'])
df_base['month_period'] = df_base['date'].dt.to_period('M')

# --- CLASE PRINCIPAL DEL DASHBOARD ---
class GreenMetricSimulatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Auditoría Predictiva GreenMetric - Campus Universitario")
        self.root.geometry("1280x750")
        self.root.configure(bg="#1A1A24")
        
        # Variables de control interactivo
        self.selected_campus = tk.StringVar(value="Todos")
        self.selected_category = tk.StringVar(value="Todas")
        self.temp_adj = tk.DoubleVar(value=0.0)
        self.hvac_eff = tk.DoubleVar(value=0.0)
        self.solar_pct = tk.DoubleVar(value=0.0)
        self.student_factor = tk.DoubleVar(value=1.0)
        
        # Configurar estilos de ttk
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(".", background="#1A1A24", foreground="#FFFFFF")
        style.configure("TLabel", background="#1A1A24", foreground="#FFFFFF", font=("Arial", 11))
        style.configure("Header.TLabel", font=("Arial", 14, "bold"), foreground="#00E676")
        style.configure("Card.TFrame", background="#262636", relief="flat")
        style.configure("Control.TFrame", background="#20202F", relief="flat")
        
        self.create_widgets()
        self.run_simulation() # Ejecución inicial
        
        # Asegurar cierre limpio al hacer clic en la X de la ventana
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def create_widgets(self):
        # --- DISEÑO DE FILAS Y COLUMNAS ---
        # Columna Izquierda (Controles e Indicadores)
        left_panel = ttk.Frame(self.root, width=400, style="Control.TFrame")
        left_panel.pack(side="left", fill="both", expand=False, padx=15, pady=15)
        left_panel.pack_propagate(False)
        
        # Columna Derecha (Gráficos)
        right_panel = ttk.Frame(self.root, style="Card.TFrame")
        right_panel.pack(side="right", fill="both", expand=True, padx=15, pady=15)
        
        # --- CONTENIDO DEL PANEL IZQUIERDO (CONTROLES) ---
        title_lbl = ttk.Label(left_panel, text="AUDITORÍA PREDICTIVA", style="Header.TLabel", background="#20202F")
        title_lbl.pack(anchor="w", padx=15, pady=(15, 2))
        subtitle_lbl = ttk.Label(left_panel, text="Estándares GreenMetric (Energía)", font=("Arial", 10, "italic"), foreground="#8A8A9F", background="#20202F")
        subtitle_lbl.pack(anchor="w", padx=15, pady=(0, 15))
        
        # Separador
        ttk.Separator(left_panel, orient="horizontal").pack(fill="x", padx=15, pady=5)
        
        # Filtros de Selección
        filt_frame = ttk.Frame(left_panel, style="Control.TFrame")
        filt_frame.pack(fill="x", padx=15, pady=10)
        
        ttk.Label(filt_frame, text="Seleccionar Campus:", background="#20202F").grid(row=0, column=0, sticky="w", pady=5)
        campuses = ["Todos"] + [str(c) for c in sorted(df_base['campus_id'].unique())]
        self.campus_combo = ttk.Combobox(filt_frame, textvariable=self.selected_campus, values=campuses, state="readonly")
        self.campus_combo.grid(row=0, column=1, sticky="w", padx=10, pady=5)
        self.campus_combo.bind("<<ComboboxSelected>>", lambda e: self.run_simulation())
        
        ttk.Label(filt_frame, text="Categoría de Edificio:", background="#20202F").grid(row=1, column=0, sticky="w", pady=5)
        categories = ["Todas"] + list(df_base['category'].unique())
        self.cat_combo = ttk.Combobox(filt_frame, textvariable=self.selected_category, values=categories, state="readonly")
        self.cat_combo.grid(row=1, column=1, sticky="w", padx=10, pady=5)
        self.cat_combo.bind("<<ComboboxSelected>>", lambda e: self.run_simulation())
        
        # Separador
        ttk.Separator(left_panel, orient="horizontal").pack(fill="x", padx=15, pady=5)
        
        # Sliders de Parámetros
        slide_frame = ttk.Frame(left_panel, style="Control.TFrame")
        slide_frame.pack(fill="x", padx=15, pady=10)
        
        # Slider 1: Temperatura
        ttk.Label(slide_frame, text="Ajuste Clima (Δ Temperatura °C):", background="#20202F").pack(anchor="w", pady=(5, 2))
        self.temp_slider = tk.Scale(slide_frame, from_=-5.0, to=5.0, resolution=0.5, orient="horizontal", 
                                    variable=self.temp_adj, bg="#20202F", fg="#FFFFFF", troughcolor="#262636",
                                    highlightthickness=0, command=lambda v: self.run_simulation())
        self.temp_slider.pack(fill="x", pady=(0, 10))
        
        # Slider 2: Ahorro HVAC
        ttk.Label(slide_frame, text="Eficiencia Energética (Ahorro HVAC %):", background="#20202F").pack(anchor="w", pady=(5, 2))
        self.hvac_slider = tk.Scale(slide_frame, from_=0, to=50, resolution=5, orient="horizontal", 
                                    variable=self.hvac_eff, bg="#20202F", fg="#FFFFFF", troughcolor="#262636",
                                    highlightthickness=0, command=lambda v: self.run_simulation())
        self.hvac_slider.pack(fill="x", pady=(0, 10))
        
        # Slider 3: Energía Solar
        ttk.Label(slide_frame, text="Energía Renovable (Solar % de Demanda):", background="#20202F").pack(anchor="w", pady=(5, 2))
        self.solar_slider = tk.Scale(slide_frame, from_=0, to=100, resolution=5, orient="horizontal", 
                                     variable=self.solar_pct, bg="#20202F", fg="#FFFFFF", troughcolor="#262636",
                                     highlightthickness=0, command=lambda v: self.run_simulation())
        self.solar_slider.pack(fill="x", pady=(0, 10))
        
        # Slider 4: Factor Ocupación Estudiantil
        ttk.Label(slide_frame, text="Asistencia/Ocupación del Campus (Factor):", background="#20202F").pack(anchor="w", pady=(5, 2))
        self.student_slider = tk.Scale(slide_frame, from_=0.5, to=1.5, resolution=0.1, orient="horizontal", 
                                       variable=self.student_factor, bg="#20202F", fg="#FFFFFF", troughcolor="#262636",
                                       highlightthickness=0, command=lambda v: self.run_simulation())
        self.student_slider.pack(fill="x", pady=(0, 15))
        
        # Separador
        ttk.Separator(left_panel, orient="horizontal").pack(fill="x", padx=15, pady=5)
        
        # Panel de Tarjetas de Resultados Numéricos
        res_frame = ttk.Frame(left_panel, style="Control.TFrame")
        res_frame.pack(fill="both", expand=True, padx=15, pady=10)
        
        # Tarjeta 1: Ahorro Total
        self.saving_lbl = ttk.Label(res_frame, text="Ahorro Total: 0.00 MWh (0.0%)", font=("Arial", 11, "bold"), foreground="#00E676", background="#20202F")
        self.saving_lbl.pack(anchor="w", pady=5)
        
        # Tarjeta 2: Huella CO2 Reducida
        self.co2_lbl = ttk.Label(res_frame, text="Reducción CO2: 0.00 ton", font=("Arial", 11, "bold"), foreground="#00B0FF", background="#20202F")
        self.co2_lbl.pack(anchor="w", pady=5)
        
        # Tarjeta 3: Puntos GreenMetric Estimados
        self.score_lbl = ttk.Label(res_frame, text="Puntos GreenMetric: 0 / 2100", font=("Arial", 12, "bold"), foreground="#FFD700", background="#20202F")
        self.score_lbl.pack(anchor="w", pady=(10, 5))
        
        # --- CONTENIDO DEL PANEL DERECHO (GRÁFICOS MATPLOTLIB) ---
        self.fig, (self.ax_line, self.ax_co2) = plt.subplots(1, 2, figsize=(10, 6.5))
        self.fig.tight_layout(pad=4.0)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        
    def run_simulation(self):
        # 1. Obtener valores de entrada de los sliders y filtros
        campus = self.selected_campus.get()
        category = self.selected_category.get()
        dt = self.temp_adj.get()
        eff = self.hvac_eff.get() / 100.0
        sol = self.solar_pct.get() / 100.0
        s_factor = self.student_factor.get()
        
        # 2. Filtrar DataFrame base según campus y categoría
        df_sim = df_base.copy()
        if campus != "Todos":
            df_sim = df_sim[df_sim['campus_id'] == int(campus)]
        if category != "Todas":
            df_sim = df_sim[df_sim['category'] == category]
            
        if len(df_sim) == 0:
            messagebox.showwarning("Sin datos", "No hay datos para la combinación de filtros seleccionada.")
            return
            
        # 3. Aplicar alteraciones de los sliders para predicción
        df_sim['apparent_temperature'] = df_sim['apparent_temperature'] + dt
        df_sim['capacity'] = df_sim['capacity'] * s_factor
        
        # 4. Ejecutar predicción con el modelo RandomForest
        # Reemplazar nulos temporales con la mediana si la alteración causó NaNs en capacidad
        df_sim['capacity'] = df_sim['capacity'].fillna(df_sim['capacity'].median())
        
        X_pred = df_sim[features_list]
        predicted_values = model.predict(X_pred)
        
        # Reemplazar valores negativos si el modelo genera alguna anomalía
        predicted_values = np.clip(predicted_values, 0, None)
        
        # 5. Aplicar políticas de simulación sobre el resultado predicho
        # Eficiencia Energética (ahorro de HVAC sobre consumo eléctrico)
        # Suponiendo que el HVAC representa el 50% de la carga eléctrica total del edificio,
        # la eficiencia aplica sobre ese 50%.
        electricity_simulated = predicted_values * (1.0 - (eff * 0.5))
        
        # Ahorro por energía solar autogenerada
        solar_generation = electricity_simulated * sol
        net_electricity = electricity_simulated - solar_generation
        
        # Añadir al DataFrame para cálculos grupales
        df_sim['baseline_consumption'] = df_sim['consumption']
        df_sim['simulated_consumption'] = net_electricity
        df_sim['gas_consumption_calc'] = df_sim['gas_consumption']
        
        # 6. Calcular agregaciones de consumo eléctrico total y gas
        total_baseline_kwh = df_sim['baseline_consumption'].sum()
        total_simulated_kwh = df_sim['simulated_consumption'].sum()
        total_gas_kwh = df_sim['gas_consumption_calc'].sum()
        
        # Conversión a MWh para mejor legibilidad
        total_baseline_mwh = total_baseline_kwh / 1000.0
        total_simulated_mwh = total_simulated_kwh / 1000.0
        saving_mwh = total_baseline_mwh - total_simulated_mwh
        saving_pct = (saving_mwh / total_baseline_mwh * 100) if total_baseline_mwh > 0 else 0.0
        
        # 7. Calcular Huella de Carbono (Emisiones CO2)
        # Factor eléctrico Australia: 0.4 kg CO2 / kWh
        # Factor gas natural: 0.18 kg CO2 / kWh
        co2_baseline_ton = ((total_baseline_kwh * 0.4) + (total_gas_kwh * 0.18)) / 1000.0
        co2_simulated_ton = ((total_simulated_kwh * 0.4) + (total_gas_kwh * 0.18)) / 1000.0
        co2_saving_ton = co2_baseline_ton - co2_simulated_ton
        
        # 8. Calcular Puntaje GreenMetric Criterio Energía y Cambio Climático (EC) - Máximo 2100 puntos
        # Evaluamos dinámicamente los subcriterios en función de los sliders:
        # EC1 (Eficiencia): Escala de 50 a 200 pts basados en eff.
        pts_ec1 = int(50 + (eff * 2) * 150) 
        # EC3 (Fuentes de Energía Renovable): Escala de 50 a 300 pts basados en sol.
        pts_ec3 = int(50 + sol * 250)
        # EC4 (Uso de Electricidad per cápita): Menor consumo = Más puntos. Escala 50 a 300 pts.
        capacidad_total = df_sim['capacity'].sum()
        if capacidad_total <= 0: capacidad_total = 1
        kwh_per_capita = total_simulated_kwh / capacidad_total
        # Rango de penalización por consumo per cápita (supone umbral de 1500 kWh/persona)
        pts_ec4 = int(np.clip(300 - (kwh_per_capita / 10.0), 50, 300))
        # EC5 (Proporción Renovables): Solar % del total. Escala de 50 a 200 pts.
        pts_ec5 = int(50 + sol * 150)
        # EC8 (Huella Carbono per cápita): Escala de 50 a 300 pts.
        co2_per_capita_kg = (co2_simulated_ton * 1000.0) / capacidad_total
        pts_ec8 = int(np.clip(300 - (co2_per_capita_kg * 0.5), 50, 300))
        # Puntos estáticos o indirectos (EC2 Smart Buildings, EC6 Green Building Ratio, EC7 Programas)
        pts_estaticos = 500 
        
        total_ec_score = pts_ec1 + pts_ec3 + pts_ec4 + pts_ec5 + pts_ec8 + pts_estaticos
        total_ec_score = int(np.clip(total_ec_score, 0, 2100))
        
        # Actualizar etiquetas de texto
        self.saving_lbl.config(text=f"Ahorro Total: {saving_mwh:.2f} MWh ({saving_pct:.1f}%)")
        self.co2_lbl.config(text=f"Reducción CO2: {co2_saving_ton:.2f} ton")
        self.score_lbl.config(text=f"Puntos GreenMetric (EC): {total_ec_score} / 2100")
        
        # --- RENDERIZAR GRÁFICOS ---
        # Limpiar ejes
        self.ax_line.clear()
        self.ax_co2.clear()
        
        # Gráfico 1: Histórico mensual de consumo eléctrico (MWh)
        df_monthly = df_sim.groupby('month_period').agg({
            'baseline_consumption': 'sum',
            'simulated_consumption': 'sum'
        })
        # Convertir a MWh
        df_monthly = df_monthly / 1000.0
        
        # Formatear el índice mensual para graficar
        months_str = [str(x) for x in df_monthly.index]
        
        self.ax_line.plot(months_str, df_monthly['baseline_consumption'], color="#FF5252", marker="o", label="Línea Base Histórica", linewidth=2)
        self.ax_line.plot(months_str, df_monthly['simulated_consumption'], color="#00E676", marker="s", label="Predicción Simulación", linewidth=2)
        self.ax_line.set_title("Consumo Mensual Predictivo (MWh)", fontsize=11, fontweight="bold", pad=10)
        self.ax_line.set_xlabel("Periodo (Año-Mes)", fontsize=9)
        self.ax_line.set_ylabel("Energía Eléctrica (MWh)", fontsize=9)
        self.ax_line.legend(loc="upper right", fontsize=8)
        self.ax_line.grid(True, linestyle="--", alpha=0.3)
        
        # Limitar número de etiquetas en el eje X para evitar sobreposición
        tick_spacing = max(1, len(months_str) // 8)
        self.ax_line.set_xticks(months_str[::tick_spacing])
        self.ax_line.set_xticklabels(months_str[::tick_spacing], rotation=30, fontsize=8)
        
        # Gráfico 2: Comparativa de Emisiones CO2
        categories_co2 = ['Línea Base', 'Simulado']
        emissions_data = [co2_baseline_ton, co2_simulated_ton]
        colors = ['#FF5252', '#00E676']
        
        bars = self.ax_co2.bar(categories_co2, emissions_data, color=colors, width=0.5)
        self.ax_co2.set_title("Emisiones CO2 Totales (Toneladas)", fontsize=11, fontweight="bold", pad=10)
        self.ax_co2.set_ylabel("Toneladas de CO2", fontsize=9)
        self.ax_co2.grid(True, axis="y", linestyle="--", alpha=0.3)
        
        # Añadir valores numéricos encima de las barras
        for bar in bars:
            height = bar.get_height()
            self.ax_co2.annotate(f"{height:.2f} t",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 puntos de desfase vertical
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
            
        # Redibujar canvas de tkinter
        self.canvas.draw()

    def on_closing(self):
        # Liberar figuras de Matplotlib y cerrar Tkinter garantizando finalizar el proceso
        plt.close('all')
        self.root.quit()
        self.root.destroy()

# --- ARRANQUE DE LA APLICACIÓN ---
if __name__ == "__main__":
    root = tk.Tk()
    app = GreenMetricSimulatorApp(root)
    root.mainloop()
