import os
import pickle
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==========================================
# PALETAS DE COLORES (CLARO vs OSCURO)
# ==========================================
PALETTE_LIGHT = {
    "PRIMARY": "#1E3A2B",       # Forest Data
    "SECONDARY": "#2D6A4F",     # Eco Metric Green
    "ACCENT": "#52B788",        # Neo Mint Glow
    "BG_MAIN": "#F0F4F1",       # Eco Mist
    "BG_CARD": "#FFFFFF",       # Blanco para tarjetas
    "TEXT_MAIN": "#2F3E46",     # Slate Carbon
    "TEXT_SUB": "#8D99AE",      # Steel Grey
    "CARD_SCORE": "#14281D"
}

PALETTE_DARK = {
    "PRIMARY": "#13251B",       # Forest Data Oscuro (Panel Izquierdo)
    "SECONDARY": "#2D6A4F",     # Eco Metric Green
    "ACCENT": "#52B788",        # Neo Mint Glow (Métricas brillantes)
    "BG_MAIN": "#12181B",       # Fondo super oscuro anti-fatiga
    "BG_CARD": "#1D282E",       # Tarjetas Slate Carbon oscuras
    "TEXT_MAIN": "#E2E8F0",     # Texto claro suave
    "TEXT_SUB": "#94A3B8",      # Subtítulos suaves
    "CARD_SCORE": "#0B1710"
}

# Cargar el modelo y los datos preprocesados
PKL_FILE = "modelo_y_datos_simulacion.pkl"

if not os.path.exists(PKL_FILE):
    raise FileNotFoundError(f"No se encontró el archivo '{PKL_FILE}'. Ejecuta primero el pipeline de datos.")

with open(PKL_FILE, 'rb') as f:
    saved_bundle = pickle.load(f)

model = saved_bundle['model']
df_base = saved_bundle['data']
features_list = saved_bundle['features']

# Mapeo de campus
CAMPUS_MAP = {
    1: "URBE - Universidad Privada Dr. Rafael Belloso Chacín",
    2: "URU - Universidad Rafael Urdaneta",
    3: "UCAB - Núcleo Zulia"
}

df_base['campus_name'] = df_base['campus_id'].map(lambda x: CAMPUS_MAP.get(x, f"Campus Universitario {x}"))
df_base['date'] = pd.to_datetime(df_base['date'])
df_base['month_period'] = df_base['date'].dt.to_period('M')

# Predicción por Machine Learning
X_all = df_base[features_list]
df_base['predicted_consumption'] = np.clip(model.predict(X_all), 0, None)


class GreenMetricAuditApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Plataforma de Auditoría GreenMetric URBE")
        self.root.geometry("1300x780")
        
        # Estado del modo visual (Default: Modo Oscuro)
        self.is_dark_mode = True
        self.colors = PALETTE_DARK
        self.root.configure(bg=self.colors["BG_MAIN"])
        
        # Variable de selección de Campus
        self.selected_campus = tk.StringVar(value="URBE - Universidad Privada Dr. Rafael Belloso Chacín")
        
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        self.create_widgets()
        self.apply_theme()
        self.update_analytics()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def create_widgets(self):
        # PANEL IZQUIERDO (FOREST DATA)
        self.left_panel = tk.Frame(self.root, width=410, bg=self.colors["PRIMARY"])
        self.left_panel.pack(side="left", fill="both", expand=False)
        self.left_panel.pack_propagate(False)
        
        # PANEL DERECHO (ÁREA DE TRABAJO)
        self.right_panel = tk.Frame(self.root, bg=self.colors["BG_CARD"])
        self.right_panel.pack(side="right", fill="both", expand=True, padx=20, pady=20)
        
        # Encabezado con Botón de Ayuda y Toggle de Tema
        self.header_frame = tk.Frame(self.left_panel, bg=self.colors["PRIMARY"])
        self.header_frame.pack(fill="x", padx=20, pady=(25, 0))
        
        self.title_lbl = tk.Label(self.header_frame, text="AUDITORÍA GREENMETRIC", font=("Arial", 13, "bold"), fg="#FFFFFF", bg=self.colors["PRIMARY"])
        self.title_lbl.pack(side="left")
        
        # Botón Toggle Modo Oscuro / Claro
        self.btn_theme = tk.Button(
            self.header_frame, 
            text=" 🌙 ", 
            font=("Arial", 9, "bold"), 
            fg="#FFFFFF", 
            bg=self.colors["SECONDARY"], 
            bd=0, 
            cursor="hand2",
            command=self.toggle_theme
        )
        self.btn_theme.pack(side="right", padx=(5, 0))
        
        # Botón de Ayuda [ ? ]
        self.btn_help = tk.Button(
            self.header_frame, 
            text=" ? ", 
            font=("Arial", 9, "bold"), 
            fg=self.colors["PRIMARY"], 
            bg=self.colors["ACCENT"], 
            bd=0, 
            cursor="hand2",
            command=self.show_help_window
        )
        self.btn_help.pack(side="right", padx=(5, 0))
        
        self.subtitle_lbl = tk.Label(self.left_panel, text="Sostenibilidad e Indicadores Globales", font=("Arial", 10, "italic"), fg=self.colors["ACCENT"], bg=self.colors["PRIMARY"])
        self.subtitle_lbl.pack(anchor="w", padx=20, pady=(2, 15))
        
        self.sep1 = tk.Frame(self.left_panel, height=2, bg=self.colors["SECONDARY"])
        self.sep1.pack(fill="x", padx=20, pady=5)
        
        # Selector
        self.filt_frame = tk.Frame(self.left_panel, bg=self.colors["PRIMARY"])
        self.filt_frame.pack(fill="x", padx=20, pady=10)
        
        self.lbl_campus = tk.Label(self.filt_frame, text="Campus Universitario:", font=("Arial", 10, "bold"), fg="#FFFFFF", bg=self.colors["PRIMARY"])
        self.lbl_campus.grid(row=0, column=0, sticky="w", pady=5)
        
        campuses = list(df_base['campus_name'].unique())
        self.campus_combo = ttk.Combobox(self.filt_frame, textvariable=self.selected_campus, values=campuses, state="readonly", width=32)
        self.campus_combo.grid(row=1, column=0, sticky="w", pady=(0, 10))
        self.campus_combo.bind("<<ComboboxSelected>>", lambda e: self.update_analytics())
        
        self.sep2 = tk.Frame(self.left_panel, height=2, bg=self.colors["SECONDARY"])
        self.sep2.pack(fill="x", padx=20, pady=10)
        
        # Indicadores
        self.res_frame = tk.Frame(self.left_panel, bg=self.colors["PRIMARY"])
        self.res_frame.pack(fill="both", expand=True, padx=20, pady=5)
        
        self.cons_lbl = tk.Label(self.res_frame, text="Consumo Total Campus: 0.00 MWh", font=("Arial", 10, "bold"), fg="#FFFFFF", bg=self.colors["PRIMARY"])
        self.cons_lbl.pack(anchor="w", pady=6)
        
        self.diff_lbl = tk.Label(self.res_frame, text="Desviación Energética: 0.00%", font=("Arial", 10, "bold"), fg=self.colors["ACCENT"], bg=self.colors["PRIMARY"])
        self.diff_lbl.pack(anchor="w", pady=6)
        
        self.co2_lbl = tk.Label(self.res_frame, text="Huella CO2 Emitida: 0.00 ton", font=("Arial", 10, "bold"), fg="#E0E0E0", bg=self.colors["PRIMARY"])
        self.co2_lbl.pack(anchor="w", pady=6)
        
        self.sep3 = tk.Frame(self.res_frame, height=2, bg=self.colors["SECONDARY"])
        self.sep3.pack(fill="x", pady=12)
        
        # Tarjeta de Puntuación
        self.score_card = tk.Frame(self.res_frame, bg=self.colors["CARD_SCORE"], bd=1, relief="solid")
        self.score_card.pack(fill="x", pady=5, ipady=8, ipadx=8)
        
        self.score_title = tk.Label(self.score_card, text="Puntaje Total GreenMetric:", font=("Arial", 10, "bold"), fg=self.colors["ACCENT"], bg=self.colors["CARD_SCORE"])
        self.score_title.pack(anchor="w", padx=10, pady=(5, 0))
        
        self.score_lbl = tk.Label(self.score_card, text="0 / 10,000 pts", font=("Arial", 16, "bold"), fg="#FFFFFF", bg=self.colors["CARD_SCORE"])
        self.score_lbl.pack(anchor="w", padx=10, pady=(0, 5))
        
        self.cat_summary_lbl = tk.Label(self.res_frame, text="", font=("Arial", 9), fg="#D0D0D0", bg=self.colors["PRIMARY"], justify="left")
        self.cat_summary_lbl.pack(anchor="w", pady=12)
        
        # Matplotlib Canvas
        self.fig = plt.Figure(figsize=(10, 6.5))
        self.ax_line = self.fig.add_subplot(121)
        self.ax_radar = self.fig.add_subplot(122, polar=True)
        self.fig.tight_layout(pad=3.5)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_panel)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    def toggle_theme(self):
        """Alterna entre Modo Claro y Modo Oscuro."""
        self.is_dark_mode = not self.is_dark_mode
        self.colors = PALETTE_DARK if self.is_dark_mode else PALETTE_LIGHT
        self.btn_theme.config(text=" 🌙 " if self.is_dark_mode else " ☀️ ")
        
        self.apply_theme()
        self.update_analytics()

    def apply_theme(self):
        """Aplica los colores seleccionados a los widgets de Tkinter."""
        self.root.configure(bg=self.colors["BG_MAIN"])
        self.left_panel.configure(bg=self.colors["PRIMARY"])
        self.right_panel.configure(bg=self.colors["BG_CARD"])
        self.header_frame.configure(bg=self.colors["PRIMARY"])
        self.title_lbl.configure(bg=self.colors["PRIMARY"])
        self.subtitle_lbl.configure(bg=self.colors["PRIMARY"], fg=self.colors["ACCENT"])
        
        self.sep1.configure(bg=self.colors["SECONDARY"])
        self.sep2.configure(bg=self.colors["SECONDARY"])
        self.sep3.configure(bg=self.colors["SECONDARY"])
        
        self.filt_frame.configure(bg=self.colors["PRIMARY"])
        self.lbl_campus.configure(bg=self.colors["PRIMARY"])
        
        self.res_frame.configure(bg=self.colors["PRIMARY"])
        self.cons_lbl.configure(bg=self.colors["PRIMARY"])
        self.diff_lbl.configure(bg=self.colors["PRIMARY"])
        self.co2_lbl.configure(bg=self.colors["PRIMARY"])
        
        self.score_card.configure(bg=self.colors["CARD_SCORE"])
        self.score_title.configure(bg=self.colors["CARD_SCORE"], fg=self.colors["ACCENT"])
        self.score_lbl.configure(bg=self.colors["CARD_SCORE"])
        self.cat_summary_lbl.configure(bg=self.colors["PRIMARY"])
        
        self.btn_help.configure(fg=self.colors["PRIMARY"], bg=self.colors["ACCENT"])

    def show_help_window(self):
        """Ventana Modal de Ayuda adaptada al tema actual."""
        help_win = tk.Toplevel(self.root)
        help_win.title("Guía del Análisis - UI GreenMetric")
        help_win.geometry("680x620")
        help_win.configure(bg=self.colors["BG_MAIN"])
        help_win.transient(self.root)
        help_win.grab_set()
        
        top_bar = tk.Frame(help_win, bg=self.colors["PRIMARY"], height=60)
        top_bar.pack(fill="x")
        
        title_help = tk.Label(top_bar, text=" Guía de Métricas e Indicadores", font=("Arial", 12, "bold"), fg="#FFFFFF", bg=self.colors["PRIMARY"])
        title_help.pack(side="left", padx=20, pady=15)
        
        container = tk.Frame(help_win, bg=self.colors["BG_MAIN"])
        container.pack(fill="both", expand=True, padx=20, pady=15)
        
        canvas = tk.Canvas(container, bg=self.colors["BG_MAIN"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg=self.colors["BG_MAIN"])
        
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        explicaciones = [
            ("⚡ Consumo Total Campus (MWh)", "Energía acumulada medida en el campus en Megavatios-hora."),
            ("📉 Desviación Energética (%)", "Porcentaje de diferencia respecto al modelo predicho. Negativo = Ahorro, Positivo = Sobreconsumo."),
            ("🌱 Huella CO2 Emitida (ton)", "Dióxido de carbono equivalente emitido por el consumo eléctrico y térmico."),
            ("🏆 Puntaje Total GreenMetric (sobre 10,000 pts)", "Suma global de los 7 ejes auditados según los lineamientos de UI GreenMetric."),
            ("📊 Categorías Evaluadas", "SI: Infraestructura | EC: Energía | WS: Residuos | WR: Agua | TR: Transporte | ED: Educación | GD: Gobernanza."),
            ("🕸️ Gráfico de Radar", "Muestra el porcentaje de cumplimiento (0-100%) por categoría para identificar fortalezas e ineficiencias.")
        ]
        
        for titulo, desc in explicaciones:
            card = tk.Frame(scroll_frame, bg=self.colors["BG_CARD"], bd=1, relief="solid", highlightthickness=0)
            card.pack(fill="x", pady=6, ipadx=10, ipady=8)
            
            lbl_title = tk.Label(card, text=titulo, font=("Arial", 10, "bold"), fg=self.colors["ACCENT"], bg=self.colors["BG_CARD"], anchor="w")
            lbl_title.pack(fill="x")
            
            lbl_desc = tk.Label(card, text=desc, font=("Arial", 9), fg=self.colors["TEXT_MAIN"], bg=self.colors["BG_CARD"], justify="left", anchor="w", wraplength=580)
            lbl_desc.pack(fill="x", pady=(4, 0))

        btn_close = tk.Button(help_win, text="Entendido / Cerrar", font=("Arial", 10, "bold"), fg="#FFFFFF", bg=self.colors["SECONDARY"], bd=0, cursor="hand2", pady=8, command=help_win.destroy)
        btn_close.pack(fill="x", padx=20, pady=(0, 15))

    def update_analytics(self):
        campus = self.selected_campus.get()
        df_filt = df_base[df_base['campus_name'] == campus].copy()
            
        if len(df_filt) == 0:
            messagebox.showwarning("Sin datos", "No hay registros disponibles para el campus seleccionado.")
            return
            
        total_real_kwh = df_filt['consumption'].sum()
        total_pred_kwh = df_filt['predicted_consumption'].sum()
        total_gas_kwh = df_filt['gas_consumption'].sum()
        
        total_real_mwh = total_real_kwh / 1000.0
        diff_pct = ((total_real_kwh - total_pred_kwh) / total_pred_kwh * 100.0) if total_pred_kwh > 0 else 0.0
        co2_ton = ((total_real_kwh * 0.4) + (total_gas_kwh * 0.18)) / 1000.0
        
        capacidad_total = df_filt['capacity'].sum()
        if capacidad_total <= 0: capacidad_total = 1
        kwh_per_capita = total_real_kwh / capacidad_total
        
        score_EC = int(np.clip(2100 - (kwh_per_capita * 0.8) - (diff_pct * 10), 400, 2100))
        score_SI, score_WS, score_WR, score_TR, score_ED, score_GD = 1150, 1350, 780, 1200, 1450, 350
        total_score = score_SI + score_EC + score_WS + score_WR + score_TR + score_ED + score_GD
        
        self.cons_lbl.config(text=f"Consumo Total Campus: {total_real_mwh:,.2f} MWh")
        diff_color = "#FF6B6B" if diff_pct > 5.0 else self.colors["ACCENT"]
        self.diff_lbl.config(text=f"Desviación Energética: {diff_pct:+.2f}%", fg=diff_color)
        self.co2_lbl.config(text=f"Huella CO2 Emitida: {co2_ton:,.2f} ton")
        self.score_lbl.config(text=f"{total_score:,} / 10,000 pts")
        
        detalles_txt = (
            f"• Infraestructura (SI): {score_SI} / 1500 pts\n"
            f"• Energía y Clima (EC): {score_EC} / 2100 pts\n"
            f"• Residuos (WS): {score_WS} / 1800 pts\n"
            f"• Agua (WR): {score_WR} / 1000 pts\n"
            f"• Transporte (TR): {score_TR} / 1800 pts\n"
            f"• Educación (ED): {score_ED} / 1800 pts\n"
            f"• Gobernanza (GD): {score_GD} pts"
        )
        self.cat_summary_lbl.config(text=detalles_txt)
        
        # Configuración de Colores de Matplotlib según el Modo Visual
        bg_fig = self.colors["BG_CARD"]
        text_fig = self.colors["TEXT_MAIN"] if not self.is_dark_mode else "#E2E8F0"
        grid_color = "#334155" if self.is_dark_mode else "#CBD5E1"
        
        self.fig.set_facecolor(self.colors["BG_MAIN"])
        self.ax_line.clear()
        self.ax_radar.clear()
        
        # Gráfico 1: Consumo Histórico
        df_monthly = df_filt.groupby('month_period').agg({'consumption': 'sum', 'predicted_consumption': 'sum'}) / 1000.0
        months_str = [str(x) for x in df_monthly.index]
        
        self.ax_line.set_facecolor(bg_fig)
        self.ax_line.plot(months_str, df_monthly['consumption'], color="#FF5252", marker="o", label="Consumo Real", linewidth=2)
        self.ax_line.plot(months_str, df_monthly['predicted_consumption'], color=self.colors["ACCENT"], marker="s", label="Esperado (ML)", linewidth=2, linestyle="--")
        
        self.ax_line.set_title("Auditoría Histórica de Energía (MWh)", fontsize=10, fontweight="bold", color=text_fig, pad=10)
        self.ax_line.set_xlabel("Periodo", fontsize=8, color=text_fig)
        self.ax_line.set_ylabel("MWh", fontsize=8, color=text_fig)
        self.ax_line.tick_params(colors=text_fig, labelsize=7)
        self.ax_line.legend(loc="upper right", fontsize=8, facecolor=bg_fig, edgecolor=grid_color, labelcolor=text_fig)
        self.ax_line.grid(True, linestyle="--", alpha=0.3, color=grid_color)
        
        tick_spacing = max(1, len(months_str) // 6)
        self.ax_line.set_xticks(months_str[::tick_spacing])
        self.ax_line.set_xticklabels(months_str[::tick_spacing], rotation=30)
        
        # Gráfico 2: Radar UI GreenMetric
        categories = ['SI\nInfraestructura', 'EC\nEnergía', 'WS\nResiduos', 'WR\nAgua', 'TR\nTransporte', 'ED\nEducación', 'GD\nGobernanza']
        scores = [score_SI/1500*100, score_EC/2100*100, score_WS/1800*100, score_WR/1000*100, score_TR/1800*100, score_ED/1800*100, 70]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        scores += scores[:1]
        angles += angles[:1]
        
        self.ax_radar.set_facecolor(bg_fig)
        self.ax_radar.plot(angles, scores, color=self.colors["ACCENT"], linewidth=2, linestyle='solid')
        self.ax_radar.fill(angles, scores, color=self.colors["ACCENT"], alpha=0.35)
        
        self.ax_radar.set_xticks(angles[:-1])
        self.ax_radar.set_xticklabels(categories, fontsize=7, fontweight="bold", color=text_fig)
        self.ax_radar.set_title("Cumplimiento % por Categoría GreenMetric", fontsize=10, fontweight="bold", color=text_fig, pad=15)
        self.ax_radar.tick_params(colors=text_fig)
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