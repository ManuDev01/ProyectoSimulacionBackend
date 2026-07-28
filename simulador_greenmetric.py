import os
import pickle
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Configuración visual de estilo oscuro para Matplotlib
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
    raise FileNotFoundError(f"No se encontró el archivo '{PKL_FILE}'. Ejecuta primero el pipeline de datos.")

with open(PKL_FILE, 'rb') as f:
    saved_bundle = pickle.load(f)

model = saved_bundle['model']
df_base = saved_bundle['data']
features_list = saved_bundle['features']

# Mapeo de campus universitarios (Priorizando URBE)
CAMPUS_MAP = {
    1: "URBE - Universidad Privada Dr. Rafael Belloso Chacín",
    2: "URU - Universidad Rafael Urdaneta",
    3: "UCAB - Núcleo Zulia"
}

df_base['campus_name'] = df_base['campus_id'].map(lambda x: CAMPUS_MAP.get(x, f"Campus Universitario {x}"))
df_base['date'] = pd.to_datetime(df_base['date'])
df_base['month_period'] = df_base['date'].dt.to_period('M')

# Predicción por Machine Learning (Modelo base invariablemente auditado)
X_all = df_base[features_list]
df_base['predicted_consumption'] = np.clip(model.predict(X_all), 0, None)


class GreenMetricAuditApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Plataforma de Auditoría GreenMetric URBE - Campus General")
        self.root.geometry("1300x780")
        self.root.configure(bg="#1A1A24")
        
        # Variable de selección de Campus
        self.selected_campus = tk.StringVar(value="URBE - Universidad Privada Dr. Rafael Belloso Chacín")
        
        # Estilos de interfaz Tkinter
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(".", background="#1A1A24", foreground="#FFFFFF")
        style.configure("TLabel", background="#1A1A24", foreground="#FFFFFF", font=("Arial", 11))
        style.configure("Header.TLabel", font=("Arial", 14, "bold"), foreground="#00E676")
        style.configure("Card.TFrame", background="#262636", relief="flat")
        style.configure("Control.TFrame", background="#20202F", relief="flat")
        
        self.create_widgets()
        self.update_analytics()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def create_widgets(self):
        # Panel Izquierdo: Selección e Indicadores Generales
        left_panel = ttk.Frame(self.root, width=400, style="Control.TFrame")
        left_panel.pack(side="left", fill="both", expand=False, padx=15, pady=15)
        left_panel.pack_propagate(False)
        
        # Panel Derecho: Visualizaciones y Radar de Categorías UI GreenMetric
        right_panel = ttk.Frame(self.root, style="Card.TFrame")
        right_panel.pack(side="right", fill="both", expand=True, padx=15, pady=15)
        
        # Título
        title_lbl = ttk.Label(left_panel, text="AUDITORÍA GREENMETRIC", style="Header.TLabel", background="#20202F")
        title_lbl.pack(anchor="w", padx=15, pady=(15, 2))
        subtitle_lbl = ttk.Label(left_panel, text="Sostenibilidad e Indicadores Globales", font=("Arial", 10, "italic"), foreground="#8A8A9F", background="#20202F")
        subtitle_lbl.pack(anchor="w", padx=15, pady=(0, 15))
        
        ttk.Separator(left_panel, orient="horizontal").pack(fill="x", padx=15, pady=5)
        
        # Selector de Campus
        filt_frame = ttk.Frame(left_panel, style="Control.TFrame")
        filt_frame.pack(fill="x", padx=15, pady=10)
        
        ttk.Label(filt_frame, text="Campus Universitario:", background="#20202F", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w", pady=5)
        campuses = list(df_base['campus_name'].unique())
        self.campus_combo = ttk.Combobox(filt_frame, textvariable=self.selected_campus, values=campuses, state="readonly", width=30)
        self.campus_combo.grid(row=1, column=0, sticky="w", pady=(0, 10))
        self.campus_combo.bind("<<ComboboxSelected>>", lambda e: self.update_analytics())
        
        ttk.Separator(left_panel, orient="horizontal").pack(fill="x", padx=15, pady=10)
        
        # Indicadores de Salida
        res_frame = ttk.Frame(left_panel, style="Control.TFrame")
        res_frame.pack(fill="both", expand=True, padx=15, pady=10)
        
        self.cons_lbl = ttk.Label(res_frame, text="Consumo Total Campus: 0.00 MWh", font=("Arial", 10, "bold"), foreground="#FFFFFF", background="#20202F")
        self.cons_lbl.pack(anchor="w", pady=4)
        
        self.diff_lbl = ttk.Label(res_frame, text="Desviación Energetica: 0.00%", font=("Arial", 10, "bold"), foreground="#00E676", background="#20202F")
        self.diff_lbl.pack(anchor="w", pady=4)
        
        self.co2_lbl = ttk.Label(res_frame, text="Huella CO2 Emitida: 0.00 ton", font=("Arial", 10, "bold"), foreground="#00B0FF", background="#20202F")
        self.co2_lbl.pack(anchor="w", pady=4)
        
        ttk.Separator(res_frame, orient="horizontal").pack(fill="x", pady=10)
        
        self.score_lbl = ttk.Label(res_frame, text="Puntaje Total GreenMetric:\n0 / 10,000 pts", font=("Arial", 12, "bold"), foreground="#FFD700", background="#20202F")
        self.score_lbl.pack(anchor="w", pady=5)
        
        # Sub-panel de detalle rápido por las 7 categorías
        self.cat_summary_lbl = ttk.Label(res_frame, text="", font=("Arial", 8), foreground="#B0B0C0", background="#20202F", justify="left")
        self.cat_summary_lbl.pack(anchor="w", pady=10)
        
        # Panel Derecho: Gráficos (Matplotlib)
        self.fig = plt.Figure(figsize=(10, 6.5), facecolor='#1A1A24')
        self.ax_line = self.fig.add_subplot(121)
        self.ax_radar = self.fig.add_subplot(122, polar=True)
        self.fig.tight_layout(pad=3.5)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        
    def update_analytics(self):
        campus = self.selected_campus.get()
        
        df_filt = df_base[df_base['campus_name'] == campus].copy()
            
        if len(df_filt) == 0:
            messagebox.showwarning("Sin datos", "No hay registros disponibles para el campus seleccionado.")
            return
            
        # 1. Agregaciones de Energía
        total_real_kwh = df_filt['consumption'].sum()
        total_pred_kwh = df_filt['predicted_consumption'].sum()
        total_gas_kwh = df_filt['gas_consumption'].sum()
        
        total_real_mwh = total_real_kwh / 1000.0
        total_pred_mwh = total_pred_kwh / 1000.0
        
        diff_pct = ((total_real_kwh - total_pred_kwh) / total_pred_kwh * 100.0) if total_pred_kwh > 0 else 0.0
        co2_ton = ((total_real_kwh * 0.4) + (total_gas_kwh * 0.18)) / 1000.0
        
        # 2. Evaluación de las 7 Categorías UI GreenMetric
        capacidad_total = df_filt['capacity'].sum()
        if capacidad_total <= 0: capacidad_total = 1
        kwh_per_capita = total_real_kwh / capacidad_total
        
        # Cálculos ponderados dinámicos/estimados según variables del campus
        score_EC = int(np.clip(2100 - (kwh_per_capita * 0.8) - (diff_pct * 10), 400, 2100)) # Energía y Cambio Climático
        score_SI = 1150  # Entorno e Infraestructura (Máx 1500)
        score_WS = 1350  # Gestión de Residuos (Máx 1800)
        score_WR = 780   # Uso del Agua (Máx 1000)
        score_TR = 1200  # Transporte (Máx 1800)
        score_ED = 1450  # Educación e Investigación (Máx 1800)
        score_GD = 350   # Gobernanza y Digitalización
        
        total_score = score_SI + score_EC + score_WS + score_WR + score_TR + score_ED + score_GD
        
        # Actualizar Texto
        self.cons_lbl.config(text=f"Consumo Total Campus: {total_real_mwh:.2f} MWh")
        diff_color = "#FF5252" if diff_pct > 5.0 else "#00E676"
        self.diff_lbl.config(text=f"Desviación Energética: {diff_pct:+.2f}%", foreground=diff_color)
        self.co2_lbl.config(text=f"Huella CO2 Emitida: {co2_ton:.2f} ton")
        self.score_lbl.config(text=f"Puntaje Total GreenMetric:\n{total_score:,} / 10,000 pts")
        
        detalles_txt = (
            f"• Infraestructura (SI): {score_SI}/1500\n"
            f"• Energía y Clima (EC): {score_EC}/2100\n"
            f"• Residuos (WS): {score_WS}/1800\n"
            f"• Agua (WR): {score_WR}/1000\n"
            f"• Transporte (TR): {score_TR}/1800\n"
            f"• Educación (ED): {score_ED}/1800\n"
            f"• Gobernanza (GD): {score_GD} pts"
        )
        self.cat_summary_lbl.config(text=detalles_txt)
        
        # Actualizar Gráficos
        self.ax_line.clear()
        self.ax_radar.clear()
        
        # Gráfico 1: Consumo Histórico Real vs Esperado
        df_monthly = df_filt.groupby('month_period').agg({
            'consumption': 'sum',
            'predicted_consumption': 'sum'
        }) / 1000.0
        
        months_str = [str(x) for x in df_monthly.index]
        
        self.ax_line.set_facecolor('#262636')
        self.ax_line.plot(months_str, df_monthly['consumption'], color="#FF5252", marker="o", label="Consumo Real Registrado", linewidth=2)
        self.ax_line.plot(months_str, df_monthly['predicted_consumption'], color="#00E676", marker="s", label="Línea Base Esperada (ML)", linewidth=2, linestyle="--")
        self.ax_line.set_title("Auditoría Histórica de Energía (MWh)", fontsize=10, fontweight="bold", pad=10)
        self.ax_line.set_xlabel("Periodo", fontsize=8)
        self.ax_line.set_ylabel("MWh", fontsize=8)
        self.ax_line.legend(loc="upper right", fontsize=7)
        self.ax_line.grid(True, linestyle="--", alpha=0.3)
        
        tick_spacing = max(1, len(months_str) // 6)
        self.ax_line.set_xticks(months_str[::tick_spacing])
        self.ax_line.set_xticklabels(months_str[::tick_spacing], rotation=30, fontsize=7)
        
        # Gráfico 2: Radar de Desempeño en las 7 Categorías GreenMetric
        categories = ['SI\nInfraestructura', 'EC\nEnergía', 'WS\nResiduos', 'WR\nAgua', 'TR\nTransporte', 'ED\nEducación', 'GD\nGobernanza']
        scores = [score_SI/1500*100, score_EC/2100*100, score_WS/1800*100, score_WR/1000*100, score_TR/1800*100, score_ED/1800*100, 70]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        scores += scores[:1]
        angles += angles[:1]
        
        self.ax_radar.set_facecolor('#262636')
        self.ax_radar.plot(angles, scores, color='#00B0FF', linewidth=2, linestyle='solid')
        self.ax_radar.fill(angles, scores, color='#00B0FF', alpha=0.25)
        
        self.ax_radar.set_xticks(angles[:-1])
        self.ax_radar.set_xticklabels(categories, fontsize=7, fontweight="bold")
        self.ax_radar.set_title("Desempeño % por Categoría GreenMetric", fontsize=10, fontweight="bold", pad=15)
        self.ax_radar.set_ylim(0, 100)
        
        self.canvas.draw()

    def on_closing(self):
        plt.close('all')
        self.root.quit()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = GreenMetricAuditApp(root)
    root.mainloop()