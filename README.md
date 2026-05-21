## Curso

**Introducción a las Aplicaciones Digitales en Minería**
Universidad · Ingeniería de Minas · 2026

Docente:Ing. APAZA CHINO JULIAN

Alumno: CHURA QUISPE JHON DENILSON

# Mine-to-Mill · Análisis de Fragmentación por Voladura

**Versión 2.0** — Aplicativo web interactivo para la optimización integrada del proceso Mine-to-Mill en operaciones mineras a cielo abierto.

\---

## ¿Qué es Mine-to-Mill?

El concepto **Mine-to-Mill** trata la mina y la planta como un sistema integrado. Una fragmentación más fina en la voladura reduce el consumo de energía en molienda, aunque incrementa el costo de explosivos. Este aplicativo encuentra el punto óptimo de ese balance.

\---

## Módulos Implementados

|#|Tab|Módulo|Descripción|
|-|-|-|-|
|1|🔩 Diseño Malla|Ash + Vista 3D|Burden, Espaciamiento, Taco · Visualización 3D del banco|
|2|🪨 Fragmentación|Kuz-Ram + Swebrec|Curva granulométrica con dos modelos comparados|
|3|🎲 Riesgo MC|Monte Carlo|5 000 escenarios · P10 / P50 / P90|
|4|💰 Economía|Bond + Tornado|Costo voladura + molienda · Análisis de sensibilidad|
|5|⚙️ Optimizador|Trade-off + Pareto|Diámetro óptimo · Frontera de Pareto|
|6|🔊 Vibraciones|PPV|Modelo Holmberg-Persson · Semáforo DIN 4150|
|7|🗿 Zonas Geológicas|Multi-zona|F₈₀ ponderado de 2 zonas con distintas propiedades|
|8|⚖️ Comparador|Escenarios A vs B|Comparación directa de dos diseños de voladura|
|9|📂 Datos CSV|Carga de campo|Importar taladros reales · Histograma de F₈₀|
|10|📋 Reporte|Exportar|Descarga Excel (5 hojas) + PDF técnico + Historial|

\---

## Modelos Matemáticos

### Índice de Lilly

```
A = 0.06 × (RMD + JPS + JPO + SGI + H)
```

### Modelo de Ash

```
B = 30 × d    S = 1.15 × B    T = 0.80 × B
```

### Modelo Kuz-Ram (con sobreperforación J)

```
X₅₀ = A · (V/Q)^0.8 · Q^0.167 · (115/RWS)^0.633   \[cm]
F₈₀ = X₅₀ · 2.321^(1/n)                             \[cm]
```

### Distribución de Rosin-Rammler

```
P(x) = \[1 - exp(-0.693 · (x/X₅₀)^n)] × 100
```

### Distribución de Swebrec (Ouchterlony, 2005)

```
P(x) = 1 / (1 + (ln(xmax/x) / ln(xmax/X₅₀))^b) × 100
```

### 3ª Ley de Bond

```
W = 10 · Wi · (1/√P₈₀ - 1/√F₈₀)    \[kWh/t]
```

### Simulación Monte Carlo

```
A\_sim  \~ N(μ\_A,  σ\_A²)
Wi\_sim \~ N(μ\_Wi, σ\_Wi²)
C\_total = C\_voladura + C\_molienda
```

### Holmberg-Persson (PPV)

```
PPV = 1140 · (R / √Q)^(-1.6)    \[mm/s]
```

\---

## Instalación y Ejecución

### 1\. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/mine-to-mill.git
cd mine-to-mill
```

### 2\. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3\. Ejecutar la aplicación

```bash
streamlit run app.py
```

La aplicación se abre en `http://localhost:8501`

> ⚠️ \*\*No ejecutar con el botón ▶ de VS Code\*\* — siempre usar `streamlit run app.py` en la terminal.

\---

## Parámetros de Entrada

|Parámetro|Símbolo|Unidad|Rango típico|
|-|-|-|-|
|Diámetro de perforación|d|mm|89 – 311|
|Altura de banco|H|m|6 – 15|
|Sobreperforación|J|m|0.5 – 2.0|
|Densidad de roca|ρ\_r|t/m³|2.5 – 3.2|
|Resistencia uniaxial|UCS|MPa|40 – 200|
|Índice de resistencia|GSI|—|20 – 90|
|Índice de Bond|Wi|kWh/t|8 – 25|
|Objetivo de molienda|P₈₀|μm|75 – 250|
|Precio explosivo|Ce|$/kg|0.5 – 2.0|
|Tarifa eléctrica|Cn|$/kWh|0.04 – 0.12|

\---

## Base de Datos de Rocas

El aplicativo incluye 8 tipos de roca precargados:

|Roca|UCS (MPa)|GSI|Wi (kWh/t)|ρ (t/m³)|
|-|-|-|-|-|
|Granito|150|65|16.0|2.65|
|Andesita|90|55|14.0|2.70|
|Pórfido de cobre|80|60|13.0|2.75|
|Caliza|60|50|10.0|2.60|
|Cuarcita|200|70|20.0|2.65|
|Esquisto|40|35|8.0|2.55|
|Diorita|120|60|15.0|2.80|
|Basalto|200|65|17.0|2.90|

\---

## Formato del CSV de Campo

Para importar datos reales de taladros (Tab 9):

```csv
taladro\_id,diametro\_mm,altura\_banco\_m,burden\_m,espaciamiento\_m,taco\_m,UCS\_MPa,GSI
T001,165,10,4.95,5.69,3.96,85,58
T002,165,10,5.10,5.85,4.08,72,50
T003,165,10,4.80,5.52,3.84,92,60
```

Una plantilla descargable está disponible dentro del Tab 9 del aplicativo.

\---

## Estructura del Repositorio

```
mine-to-mill/
│
├── app.py              ← Aplicación principal (Streamlit v2.0)
├── requirements.txt    ← Dependencias Python
└── README.md           ← Este archivo
```

\---

## Dependencias

```
streamlit    → Framework web
plotly       → Gráficos interactivos (2D y 3D)
pandas       → Manejo de datos
numpy        → Cálculo científico y Monte Carlo
fpdf2        → Generación de reportes PDF
openpyxl     → Exportación a Excel
```

\---

## Referencias Bibliográficas

* Ash, R.L. (1963). *The mechanics of rock breakage*. Pit and Quarry, 56(2-5).
* Bond, F.C. (1952). *The third theory of comminution*. Trans. AIME, 193, 484-494.
* Cunningham, C.V.B. (1983). *The Kuz-Ram model for prediction of fragmentation from blasting*. 1st Int. Symp. Rock Fragmentation by Blasting, Luleå.
* Holmberg, R. \& Persson, P.A. (1979). *Design of tunnel perimeter blasthole patterns*. Tunnelling '79, London.
* Lilly, P.A. (1986). *An empirical method of assessing rock mass blastability*. AusIMM Bulletin, 291(3), 89-92.
* McKee, D.J. (2013). *Understanding Mine to Mill*. CRC ORE, Brisbane.
* Ouchterlony, F. (2005). *The Swebrec function: linking fragmentation by blasting and crushing*. Mining Technology, 114(1), 29-44.

\---

