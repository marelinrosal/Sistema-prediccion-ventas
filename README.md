# 📊 Sistema de Predicción de Ventas

**Instituto Tecnológico de Toluca — Ingeniería en Sistemas Computacionales**  
Tópicos de Desarrollo de Sistemas

## Autores
- Hernández Zepeda Rodrigo
- Ortiz Gallegos Starenka Susana
- Orozco Reyes Hiram
- Salgado Rojas Marelin Iral

---

## Descripción

Aplicación de escritorio en Python con interfaz gráfica (Tkinter) para analizar y predecir ventas empresariales a partir de datos históricos en CSV, usando regresión polinomial (scikit-learn).

## Características

- 🏠 **Resumen (KPIs):** Total de ingresos, promedio mensual, productos únicos y registros cargados, con gráfico de barras mensual.
- 📈 **Tendencias:** Visualización de ingresos por mes, categoría o región.
- 🔮 **Predicción:** Regresión polinomial (lineal, cuadrática o cúbica) con métricas MAE y R².
- 🏆 **Productos:** Ranking de los N productos más rentables.
- 📋 **Tabla mensual:** Vista tabular con exportación a CSV.

## Instalación

```bash
pip install pandas matplotlib scikit-learn numpy
python sales_predictor.py
```

## Formato del CSV

El archivo de ventas debe contener estas columnas:

| Columna          | Tipo      | Descripción                  |
|------------------|-----------|------------------------------|
| fecha            | fecha     | Fecha de la transacción      |
| producto         | texto     | Nombre del producto          |
| cantidad         | numérico  | Unidades vendidas            |
| precio_unitario  | numérico  | Precio por unidad            |

Columnas opcionales: `categoria`, `region`

## Ejemplo de CSV

```csv
fecha,producto,cantidad,precio_unitario
2024-01-05,Laptop Pro,10,15000
2024-01-12,Mouse Inalámbrico,50,350
2024-02-03,Monitor 27",8,8500
```

## Tecnologías

- Python 3.10+
- Tkinter (GUI)
- Pandas (procesamiento de datos)
- Matplotlib (gráficas)
- scikit-learn (regresión polinomial)
- NumPy
