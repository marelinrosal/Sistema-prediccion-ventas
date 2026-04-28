"""
Sistema de Predicción de Ventas para Empresas
Instituto Tecnológico de Toluca - Ingeniería en Sistemas Computacionales
Tópicos de Desarrollo de Sistemas

Autores:
  - Hernández Zepeda Rodrigo
  - Ortiz Gallegos Starenka Susana
  - Orozco Reyes Hiram
  - Salgado Rojas Marelin Iral

Dependencias:
  pip install pandas matplotlib scikit-learn numpy
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────── PALETA DE COLORES EMPRESARIAL ───────────────────
COLORS = {
    "bg":        "#F5F7FA",          # Fondo principal blanco grisáceo
    "panel":     "#FFFFFF",          # Paneles blancos
    "panel2":    "#EEF1F6",          # Paneles secundarios azul muy claro
    "accent":    "#1A56DB",          # Azul corporativo primario
    "accent2":   "#0E9F6E",          # Verde corporativo
    "success":   "#057A55",          # Verde oscuro
    "warning":   "#C27803",          # Ámbar corporativo
    "danger":    "#C81E1E",          # Rojo corporativo
    "text":      "#111928",          # Texto principal casi negro
    "text_dim":  "#6B7280",          # Texto secundario gris
    "border":    "#D1D5DB",          # Bordes gris claro
    "chart_bg":  "#FFFFFF",          # Fondo de gráficas blanco
    "header_bg": "#1A56DB",          # Header azul corporativo
    "kpi_border":"#DBEAFE",          # Borde tarjetas KPI azul suave
}

CHART_COLORS = ["#1A56DB", "#0E9F6E", "#C27803", "#9061F9", "#C81E1E",
                "#0694A2", "#E74694", "#057A55"]

# ─────────────────────────── LÓGICA DE DATOS ─────────────────────────────────
class SalesModel:
    def __init__(self):
        self.df = None
        self.df_monthly = None

    def load_csv(self, path: str) -> tuple[bool, str]:
        try:
            df = pd.read_csv(path)
            cols_req = {"fecha", "producto", "cantidad", "precio_unitario"}
            if not cols_req.issubset(set(df.columns.str.lower())):
                return False, f"El archivo debe tener las columnas: {cols_req}"
            df.columns = df.columns.str.lower()
            df["fecha"] = pd.to_datetime(df["fecha"])
            df["ingresos"] = df["cantidad"] * df["precio_unitario"]
            df.dropna(subset=["fecha", "cantidad", "precio_unitario"], inplace=True)
            df.drop_duplicates(inplace=True)
            self.df = df
            self._build_monthly()
            return True, f"✔ {len(df)} registros cargados correctamente."
        except Exception as e:
            return False, f"Error al leer el archivo: {e}"

    def _build_monthly(self):
        df = self.df.copy()
        df["mes"] = df["fecha"].dt.to_period("M")
        self.df_monthly = (
            df.groupby("mes")
              .agg(total_ingresos=("ingresos", "sum"),
                   total_cantidad=("cantidad", "sum"),
                   num_transacciones=("fecha", "count"))
              .reset_index()
              .sort_values("mes")
        )
        self.df_monthly["mes_idx"] = range(len(self.df_monthly))

    def summary(self) -> dict:
        df = self.df
        return {
            "total_ventas":   f"${df['ingresos'].sum():,.0f}",
            "promedio_mes":   f"${self.df_monthly['total_ingresos'].mean():,.0f}",
            "productos":      df["producto"].nunique(),
            "registros":      len(df),
            "periodo":        f"{df['fecha'].min().strftime('%b %Y')} – {df['fecha'].max().strftime('%b %Y')}",
        }

    def predict(self, n_months: int, degree: int = 2) -> dict:
        mon = self.df_monthly
        X = mon["mes_idx"].values.reshape(-1, 1)
        y = mon["total_ingresos"].values
        poly = PolynomialFeatures(degree=degree)
        Xp = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(Xp, y)
        y_pred_hist = model.predict(Xp)
        mae = mean_absolute_error(y, y_pred_hist)
        r2  = r2_score(y, y_pred_hist)
        last_idx = mon["mes_idx"].max()
        last_period = mon["mes"].iloc[-1]
        future_idx = np.arange(last_idx + 1, last_idx + 1 + n_months).reshape(-1, 1)
        future_Xp  = poly.transform(future_idx)
        future_vals = model.predict(future_Xp)
        future_periods = [last_period + i for i in range(1, n_months + 1)]
        return {
            "hist_x":    list(mon["mes"].astype(str)),
            "hist_y":    list(y),
            "fit_y":     list(y_pred_hist),
            "future_x":  [str(p) for p in future_periods],
            "future_y":  list(np.maximum(future_vals, 0)),
            "mae":       mae,
            "r2":        r2,
            "degree":    degree,
        }

    def top_products(self, n=5) -> pd.DataFrame:
        return (
            self.df.groupby("producto")["ingresos"]
            .sum().nlargest(n).reset_index()
        )

    def monthly_table(self) -> pd.DataFrame:
        m = self.df_monthly.copy()
        m["mes"] = m["mes"].astype(str)
        m["total_ingresos"] = m["total_ingresos"].apply(lambda x: f"${x:,.0f}")
        m = m.rename(columns={
            "mes": "Mes", "total_ingresos": "Ingresos",
            "total_cantidad": "Unidades", "num_transacciones": "Transacciones"
        })
        return m[["Mes", "Ingresos", "Unidades", "Transacciones"]]


# ─────────────────────────── APLICACIÓN PRINCIPAL ────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sistema de Predicción de Ventas")
        self.geometry("1280x780")
        self.configure(bg=COLORS["bg"])
        self.resizable(True, True)
        self.model = SalesModel()
        self._configure_styles()
        self._build_ui()

    def _configure_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background=COLORS["bg"], foreground=COLORS["text"],
                        font=("Segoe UI", 10))
        style.configure("TFrame",  background=COLORS["bg"])
        style.configure("Card.TFrame", background=COLORS["panel"],
                        relief="flat", borderwidth=1)
        style.configure("TLabel",  background=COLORS["bg"], foreground=COLORS["text"])
        style.configure("Dim.TLabel", background=COLORS["panel"],
                        foreground=COLORS["text_dim"], font=("Segoe UI", 9))
        style.configure("Title.TLabel", background=COLORS["panel"],
                        foreground=COLORS["text"], font=("Segoe UI", 13, "bold"))
        style.configure("Big.TLabel",  background=COLORS["panel"],
                        foreground=COLORS["accent"], font=("Segoe UI", 20, "bold"))
        style.configure("Header.TLabel", background=COLORS["bg"],
                        foreground=COLORS["text"], font=("Segoe UI", 11, "bold"))
        style.configure("TNotebook", background=COLORS["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", background=COLORS["panel2"],
                        foreground=COLORS["text_dim"], padding=[16, 8],
                        font=("Segoe UI", 10))
        style.map("TNotebook.Tab",
                  background=[("selected", COLORS["accent"])],
                  foreground=[("selected", "#FFFFFF")])
        style.configure("TButton", background=COLORS["accent"],
                        foreground="#FFFFFF", font=("Segoe UI", 10, "bold"),
                        borderwidth=0, focusthickness=0, padding=[14, 8])
        style.map("TButton",
                  background=[("active", "#1648C0"), ("pressed", "#1240A8")])
        style.configure("Sec.TButton", background=COLORS["panel2"],
                        foreground=COLORS["text"], font=("Segoe UI", 10),
                        borderwidth=0, padding=[12, 7])
        style.map("Sec.TButton", background=[("active", COLORS["border"])])
        style.configure("TCombobox", fieldbackground=COLORS["panel"],
                        background=COLORS["panel2"], foreground=COLORS["text"],
                        selectbackground=COLORS["accent"], arrowcolor=COLORS["text"])
        style.configure("TSpinbox", fieldbackground=COLORS["panel"],
                        background=COLORS["panel2"], foreground=COLORS["text"])
        style.configure("Treeview", background=COLORS["panel"],
                        fieldbackground=COLORS["panel"],
                        foreground=COLORS["text"], rowheight=26,
                        font=("Segoe UI", 9))
        style.configure("Treeview.Heading", background=COLORS["panel2"],
                        foreground=COLORS["text_dim"], font=("Segoe UI", 9, "bold"))
        style.map("Treeview", background=[("selected", COLORS["accent"])],
                  foreground=[("selected", "#FFFFFF")])
        style.configure("TScrollbar", background=COLORS["panel2"],
                        troughcolor=COLORS["panel"], arrowcolor=COLORS["text_dim"])

    def _build_ui(self):
        # ─ Header azul corporativo ─
        header = tk.Frame(self, bg=COLORS["header_bg"], height=64)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(header, text="📊 ", bg=COLORS["header_bg"],
                 fg="#FFFFFF", font=("Segoe UI", 22)).pack(side="left", padx=(20, 4), pady=10)
        tk.Label(header, text="Sistema de Predicción de Ventas",
                 bg=COLORS["header_bg"], fg="#FFFFFF",
                 font=("Segoe UI", 16, "bold")).pack(side="left", pady=10)
        tk.Label(header, text="IT Toluca · ISC · Tópicos de Desarrollo",
                 bg=COLORS["header_bg"], fg="#BFDBFE",
                 font=("Segoe UI", 9)).pack(side="right", padx=20)

        sep = tk.Frame(self, bg=COLORS["kpi_border"], height=2)
        sep.pack(fill="x")

        # ─ Toolbar ─
        toolbar = tk.Frame(self, bg=COLORS["bg"], pady=10)
        toolbar.pack(fill="x", padx=20)
        self.lbl_file = tk.Label(toolbar, text="Sin archivo cargado",
                                 bg=COLORS["panel"], fg=COLORS["text_dim"],
                                 font=("Segoe UI", 9), padx=12, pady=6,
                                 anchor="w", width=55, relief="flat",
                                 highlightbackground=COLORS["border"],
                                 highlightthickness=1)
        self.lbl_file.pack(side="left")
        ttk.Button(toolbar, text="📂  Cargar CSV",
                   command=self._load_file).pack(side="left", padx=(10, 5))
        ttk.Button(toolbar, text="🔄  Recargar",
                   command=self._reload, style="Sec.TButton").pack(side="left")
        self.lbl_status = tk.Label(toolbar, text="", bg=COLORS["bg"],
                                   fg=COLORS["success"], font=("Segoe UI", 9))
        self.lbl_status.pack(side="left", padx=14)

        # ─ Notebook ─
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        self.tab_dashboard  = ttk.Frame(self.nb)
        self.tab_tendencias = ttk.Frame(self.nb)
        self.tab_prediccion = ttk.Frame(self.nb)
        self.tab_productos  = ttk.Frame(self.nb)
        self.tab_tabla      = ttk.Frame(self.nb)
        self.nb.add(self.tab_dashboard,  text="  🏠  Resumen  ")
        self.nb.add(self.tab_tendencias, text="  📈  Tendencias  ")
        self.nb.add(self.tab_prediccion, text="  🔮  Predicción  ")
        self.nb.add(self.tab_productos,  text="  🏆  Productos  ")
        self.nb.add(self.tab_tabla,      text="  📋  Tabla mensual  ")
        self._build_dashboard()
        self._build_tendencias()
        self._build_prediccion()
        self._build_productos()
        self._build_tabla()

    # ── Pestaña 1: Resumen ────────────────────────────────────────────────────
    def _build_dashboard(self):
        frm = self.tab_dashboard
        frm.configure(style="TFrame")
        self.frm_hint = tk.Frame(frm, bg=COLORS["bg"])
        self.frm_hint.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(self.frm_hint, text="📂", bg=COLORS["bg"],
                 font=("Segoe UI", 48)).pack()
        tk.Label(self.frm_hint, text="Carga un archivo CSV para comenzar",
                 bg=COLORS["bg"], fg=COLORS["text_dim"],
                 font=("Segoe UI", 14)).pack(pady=6)
        tk.Label(self.frm_hint,
                 text="El archivo debe tener columnas:\nfecha, producto, cantidad, precio_unitario",
                 bg=COLORS["bg"], fg=COLORS["text_dim"],
                 font=("Segoe UI", 10), justify="center").pack()

        self.frm_kpis = tk.Frame(frm, bg=COLORS["bg"])
        kpi_defs = [
            ("💰", "Total Ingresos",    "total_ventas",  COLORS["accent"]),
            ("📅", "Promedio Mensual",  "promedio_mes",  COLORS["accent2"]),
            ("📦", "Productos Únicos",  "productos",     COLORS["success"]),
            ("🗂️", "Registros",         "registros",     COLORS["warning"]),
        ]
        self.kpi_labels = {}
        for icon, title, key, color in kpi_defs:
            card = tk.Frame(self.frm_kpis, bg=COLORS["panel"],
                            highlightbackground=color, highlightthickness=2,
                            width=200, height=100)
            card.pack(side="left", padx=12, pady=8)
            card.pack_propagate(False)
            tk.Label(card, text=icon, bg=COLORS["panel"],
                     font=("Segoe UI", 22)).place(x=12, y=10)
            tk.Label(card, text=title, bg=COLORS["panel"],
                     fg=COLORS["text_dim"], font=("Segoe UI", 9)).place(x=12, y=54)
            lv = tk.Label(card, text="—", bg=COLORS["panel"],
                          fg=color, font=("Segoe UI", 18, "bold"))
            lv.place(x=12, y=68)
            self.kpi_labels[key] = lv

        self.frm_chart_dash = tk.Frame(frm, bg=COLORS["bg"])
        self.canvas_dash = None
        lp = tk.Label(frm, text="Período: —", bg=COLORS["bg"],
                      fg=COLORS["text_dim"], font=("Segoe UI", 9))
        lp.pack(side="bottom", pady=4)
        self.lbl_periodo = lp

    def _update_dashboard(self):
        self.frm_hint.place_forget()
        sm = self.model.summary()
        for key, lbl in self.kpi_labels.items():
            lbl.config(text=str(sm[key]))
        self.lbl_periodo.config(text=f"Período: {sm['periodo']}")
        self.frm_kpis.pack(side="top", fill="x", padx=20, pady=(20, 0))
        self.frm_chart_dash.pack(side="top", fill="both", expand=True, padx=20, pady=10)
        if self.canvas_dash:
            self.canvas_dash.get_tk_widget().destroy()
        mon = self.model.df_monthly
        fig = Figure(figsize=(9, 3.5), facecolor=COLORS["chart_bg"])
        ax = fig.add_subplot(111, facecolor=COLORS["chart_bg"])
        xs = [str(m) for m in mon["mes"]]
        ys = mon["total_ingresos"].values / 1000
        ax.bar(xs, ys, color=COLORS["accent"], alpha=0.85, zorder=3)
        ax.plot(xs, ys, color=COLORS["accent2"], linewidth=2, marker="o",
                markersize=4, zorder=4)
        ax.set_title("Ingresos Mensuales (miles $)", color=COLORS["text"],
                     fontsize=11, pad=10)
        ax.tick_params(axis="x", rotation=45, labelsize=7, colors=COLORS["text_dim"])
        ax.tick_params(axis="y", labelsize=8, colors=COLORS["text_dim"])
        ax.set_facecolor(COLORS["chart_bg"])
        for spine in ax.spines.values():
            spine.set_edgecolor(COLORS["border"])
        ax.yaxis.grid(True, color=COLORS["border"], linestyle="--", alpha=0.5, zorder=0)
        fig.tight_layout()
        self.canvas_dash = FigureCanvasTkAgg(fig, master=self.frm_chart_dash)
        self.canvas_dash.draw()
        self.canvas_dash.get_tk_widget().pack(fill="both", expand=True)

    # ── Pestaña 2: Tendencias ─────────────────────────────────────────────────
    def _build_tendencias(self):
        frm = self.tab_tendencias
        ctrl = tk.Frame(frm, bg=COLORS["bg"])
        ctrl.pack(fill="x", padx=16, pady=10)
        tk.Label(ctrl, text="Agrupar por:", bg=COLORS["bg"],
                 fg=COLORS["text"], font=("Segoe UI", 10)).pack(side="left")
        self.cmb_group = ttk.Combobox(ctrl, values=["Mes", "Categoría", "Región"],
                                      state="readonly", width=14)
        self.cmb_group.set("Mes")
        self.cmb_group.pack(side="left", padx=8)
        ttk.Button(ctrl, text="Actualizar gráfico",
                   command=self._update_tendencias).pack(side="left")
        self.frm_tend = tk.Frame(frm, bg=COLORS["bg"])
        self.frm_tend.pack(fill="both", expand=True, padx=16, pady=(0, 10))
        self.canvas_tend = None
        self._placeholder(self.frm_tend, "Carga un archivo CSV para ver tendencias")

    def _update_tendencias(self):
        for w in self.frm_tend.winfo_children():
            w.destroy()
        if not self.model.df_monthly is not None and self.model.df is None:
            return
        group = self.cmb_group.get()
        df = self.model.df.copy()
        fig = Figure(figsize=(9, 4.5), facecolor=COLORS["chart_bg"])
        ax = fig.add_subplot(111, facecolor=COLORS["chart_bg"])
        if group == "Mes":
            mon = self.model.df_monthly
            xs = [str(m) for m in mon["mes"]]
            ax.fill_between(xs, mon["total_ingresos"] / 1000,
                            alpha=0.15, color=COLORS["accent"])
            ax.plot(xs, mon["total_ingresos"] / 1000,
                    color=COLORS["accent"], linewidth=2.5, marker="o", markersize=5)
            ax.set_title("Ingresos por Mes (miles $)", color=COLORS["text"], fontsize=12)
            ax.tick_params(axis="x", rotation=45, labelsize=7)
        elif group == "Categoría":
            if "categoria" not in df.columns:
                messagebox.showinfo("Info", "El archivo no tiene columna 'categoria'.")
                return
            cat_data = df.groupby("categoria")["ingresos"].sum()
            wedges, texts, autotexts = ax.pie(
                cat_data, labels=cat_data.index, autopct="%1.1f%%",
                colors=CHART_COLORS[:len(cat_data)], startangle=90,
                pctdistance=0.82, wedgeprops={"linewidth": 2, "edgecolor": COLORS["chart_bg"]})
            for t in texts + autotexts:
                t.set_color(COLORS["text"])
            ax.set_title("Ingresos por Categoría", color=COLORS["text"], fontsize=12)
        elif group == "Región":
            if "region" not in df.columns:
                messagebox.showinfo("Info", "El archivo no tiene columna 'region'.")
                return
            reg_data = df.groupby("region")["ingresos"].sum().sort_values(ascending=True)
            bars = ax.barh(reg_data.index, reg_data.values / 1000,
                           color=CHART_COLORS[:len(reg_data)])
            for bar, val in zip(bars, reg_data.values / 1000):
                ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                        f"${val:,.0f}k", va="center",
                        color=COLORS["text"], fontsize=9)
            ax.set_title("Ingresos por Región (miles $)", color=COLORS["text"], fontsize=12)
        for spine in ax.spines.values():
            spine.set_edgecolor(COLORS["border"])
        ax.tick_params(colors=COLORS["text_dim"])
        ax.yaxis.label.set_color(COLORS["text_dim"])
        ax.xaxis.label.set_color(COLORS["text_dim"])
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.frm_tend)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(canvas, self.frm_tend).pack(side="bottom", fill="x")

    # ── Pestaña 3: Predicción ─────────────────────────────────────────────────
    def _build_prediccion(self):
        frm = self.tab_prediccion
        ctrl = tk.Frame(frm, bg=COLORS["bg"])
        ctrl.pack(fill="x", padx=16, pady=12)
        tk.Label(ctrl, text="Meses a predecir:", bg=COLORS["bg"],
                 fg=COLORS["text"]).pack(side="left")
        self.spin_months = ttk.Spinbox(ctrl, from_=1, to=24, width=5)
        self.spin_months.set(6)
        self.spin_months.pack(side="left", padx=8)
        tk.Label(ctrl, text="Grado de regresión:", bg=COLORS["bg"],
                 fg=COLORS["text"]).pack(side="left", padx=(16, 0))
        self.cmb_degree = ttk.Combobox(ctrl, values=["1 - Lineal", "2 - Cuadrática", "3 - Cúbica"],
                                       state="readonly", width=18)
        self.cmb_degree.set("2 - Cuadrática")
        self.cmb_degree.pack(side="left", padx=8)
        ttk.Button(ctrl, text="🔮  Generar Predicción",
                   command=self._run_prediction).pack(side="left", padx=8)
        self.frm_metrics = tk.Frame(frm, bg=COLORS["bg"])
        self.frm_metrics.pack(fill="x", padx=16, pady=(0, 6))
        self.lbl_mae = tk.Label(self.frm_metrics, text="MAE: —", bg=COLORS["bg"],
                                fg=COLORS["text_dim"], font=("Segoe UI", 9))
        self.lbl_mae.pack(side="left", padx=4)
        self.lbl_r2 = tk.Label(self.frm_metrics, text="R²: —", bg=COLORS["bg"],
                               fg=COLORS["text_dim"], font=("Segoe UI", 9))
        self.lbl_r2.pack(side="left", padx=4)
        self.frm_pred_chart = tk.Frame(frm, bg=COLORS["bg"])
        self.frm_pred_chart.pack(fill="both", expand=True, padx=16, pady=(0, 10))
        self._placeholder(self.frm_pred_chart, "Configura los parámetros y presiona 'Generar Predicción'")
        self.frm_pred_tbl = tk.Frame(frm, bg=COLORS["bg"])
        self.frm_pred_tbl.pack(fill="x", padx=16, pady=(0, 12))

    def _run_prediction(self):
        if self.model.df is None:
            messagebox.showwarning("Aviso", "Primero carga un archivo CSV.")
            return
        try:
            n = int(self.spin_months.get())
        except ValueError:
            messagebox.showerror("Error", "Número de meses inválido.")
            return
        degree = int(self.cmb_degree.get().split(" ")[0])
        res = self.model.predict(n, degree)
        self.lbl_mae.config(text=f"MAE: ${res['mae']:,.0f}", fg=COLORS["warning"])
        self.lbl_r2.config(text=f"R²: {res['r2']:.4f}",
                           fg=COLORS["success"] if res["r2"] > 0.7 else COLORS["danger"])
        for w in self.frm_pred_chart.winfo_children():
            w.destroy()
        fig = Figure(figsize=(9, 4), facecolor=COLORS["chart_bg"])
        ax = fig.add_subplot(111, facecolor=COLORS["chart_bg"])
        all_x = res["hist_x"] + res["future_x"]
        n_hist = len(res["hist_x"])
        n_fut  = len(res["future_x"])
        ax.bar(range(n_hist), [v / 1000 for v in res["hist_y"]],
               color=COLORS["accent"], alpha=0.65, label="Histórico", zorder=3)
        ax.plot(range(n_hist), [v / 1000 for v in res["fit_y"]],
                color=COLORS["warning"], linewidth=2, linestyle="--",
                label=f"Ajuste (grado {degree})", zorder=4)
        future_start = n_hist - 1
        x_fut = list(range(future_start, n_hist + n_fut))
        y_fut_line = [res["fit_y"][-1] / 1000] + [v / 1000 for v in res["future_y"]]
        ax.bar(range(n_hist, n_hist + n_fut),
               [v / 1000 for v in res["future_y"]],
               color=COLORS["accent2"], alpha=0.65, label="Predicción", zorder=3)
        ax.plot(x_fut, y_fut_line, color=COLORS["accent2"],
                linewidth=2.5, marker="o", markersize=6, zorder=5)
        ax.set_xticks(range(len(all_x)))
        ax.set_xticklabels(all_x, rotation=45, ha="right", fontsize=7,
                           color=COLORS["text_dim"])
        ax.set_title("Predicción de Ingresos (miles $)", color=COLORS["text"], fontsize=12)
        ax.legend(facecolor=COLORS["panel"], edgecolor=COLORS["border"],
                  labelcolor=COLORS["text"], fontsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(COLORS["border"])
        ax.tick_params(axis="y", colors=COLORS["text_dim"])
        ax.yaxis.grid(True, color=COLORS["border"], linestyle="--", alpha=0.4)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.frm_pred_chart)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        for w in self.frm_pred_tbl.winfo_children():
            w.destroy()
        tk.Label(self.frm_pred_tbl, text="Valores predichos:",
                 bg=COLORS["bg"], fg=COLORS["text_dim"],
                 font=("Segoe UI", 9, "bold")).pack(side="left", padx=4)
        for mes, val in zip(res["future_x"], res["future_y"]):
            chip = tk.Label(self.frm_pred_tbl,
                            text=f"{mes}: ${val:,.0f}",
                            bg=COLORS["kpi_border"], fg=COLORS["accent"],
                            font=("Segoe UI", 8), padx=8, pady=3)
            chip.pack(side="left", padx=3)

    # ── Pestaña 4: Productos ──────────────────────────────────────────────────
    def _build_productos(self):
        frm = self.tab_productos
        ctrl = tk.Frame(frm, bg=COLORS["bg"])
        ctrl.pack(fill="x", padx=16, pady=10)
        tk.Label(ctrl, text="Top N productos:", bg=COLORS["bg"],
                 fg=COLORS["text"]).pack(side="left")
        self.spin_top = ttk.Spinbox(ctrl, from_=3, to=20, width=5)
        self.spin_top.set(5)
        self.spin_top.pack(side="left", padx=8)
        ttk.Button(ctrl, text="Actualizar",
                   command=self._update_productos).pack(side="left")
        self.frm_prod = tk.Frame(frm, bg=COLORS["bg"])
        self.frm_prod.pack(fill="both", expand=True, padx=16, pady=(0, 10))
        self._placeholder(self.frm_prod, "Carga un archivo CSV para ver el ranking de productos")

    def _update_productos(self):
        if self.model.df is None:
            return
        for w in self.frm_prod.winfo_children():
            w.destroy()
        try:
            n = int(self.spin_top.get())
        except ValueError:
            n = 5
        top = self.model.top_products(n)
        fig = Figure(figsize=(8, max(3, n * 0.55)), facecolor=COLORS["chart_bg"])
        ax = fig.add_subplot(111, facecolor=COLORS["chart_bg"])
        colors_bar = CHART_COLORS[:len(top)]
        bars = ax.barh(top["producto"], top["ingresos"] / 1000,
                       color=colors_bar, zorder=3)
        for bar, val in zip(bars, top["ingresos"].values / 1000):
            ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
                    f"${val:,.1f}k", va="center",
                    color=COLORS["text"], fontsize=9)
        ax.set_title(f"Top {n} Productos por Ingresos (miles $)",
                     color=COLORS["text"], fontsize=12)
        ax.invert_yaxis()
        for spine in ax.spines.values():
            spine.set_edgecolor(COLORS["border"])
        ax.tick_params(colors=COLORS["text_dim"])
        ax.xaxis.grid(True, color=COLORS["border"], linestyle="--", alpha=0.4)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.frm_prod)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # ── Pestaña 5: Tabla mensual ──────────────────────────────────────────────
    def _build_tabla(self):
        frm = self.tab_tabla
        ctrl = tk.Frame(frm, bg=COLORS["bg"])
        ctrl.pack(fill="x", padx=16, pady=10)
        ttk.Button(ctrl, text="📥  Exportar CSV",
                   command=self._export_monthly, style="Sec.TButton").pack(side="left")
        self.frm_tbl = tk.Frame(frm, bg=COLORS["bg"])
        self.frm_tbl.pack(fill="both", expand=True, padx=16, pady=(0, 14))
        self._placeholder(self.frm_tbl, "Carga un archivo CSV para ver la tabla mensual")

    def _update_tabla(self):
        for w in self.frm_tbl.winfo_children():
            w.destroy()
        df = self.model.monthly_table()
        cols = list(df.columns)
        tree = ttk.Treeview(self.frm_tbl, columns=cols, show="headings")
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, anchor="center", width=140)
        for _, row in df.iterrows():
            tree.insert("", "end", values=list(row))
        sb = ttk.Scrollbar(self.frm_tbl, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        tree.pack(fill="both", expand=True)

    def _export_monthly(self):
        if self.model.df is None:
            messagebox.showwarning("Aviso", "No hay datos para exportar.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")],
            title="Guardar tabla mensual")
        if path:
            self.model.df_monthly.to_csv(path, index=False)
            messagebox.showinfo("Exportado", f"Archivo guardado en:\n{path}")

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _placeholder(self, parent, msg="Sin datos"):
        f = tk.Frame(parent, bg=COLORS["bg"])
        f.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(f, text="📊", bg=COLORS["bg"],
                 font=("Segoe UI", 36)).pack()
        tk.Label(f, text=msg, bg=COLORS["bg"],
                 fg=COLORS["text_dim"], font=("Segoe UI", 12)).pack(pady=6)

    def _load_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Archivos CSV", "*.csv"), ("Todos", "*.*")],
            title="Seleccionar archivo de ventas")
        if not path:
            return
        self._apply_file(path)

    def _apply_file(self, path):
        ok, msg = self.model.load_csv(path)
        short = path.split("/")[-1].split("\\")[-1]
        self.lbl_file.config(text=f"📄 {short}")
        if ok:
            self.lbl_status.config(text=msg, fg=COLORS["success"])
            self._update_dashboard()
            self._update_tendencias()
            self._update_productos()
            self._update_tabla()
        else:
            self.lbl_status.config(text=msg, fg=COLORS["danger"])
            messagebox.showerror("Error al cargar", msg)

    def _reload(self):
        path = self.lbl_file.cget("text").replace("📄 ", "").strip()
        if not path or path == "Sin archivo cargado":
            messagebox.showinfo("Info", "No hay archivo cargado para recargar.")
            return
        self._apply_file(path)


# ─────────────────────────── PUNTO DE ENTRADA ────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()