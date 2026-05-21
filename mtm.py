# =============================================================================
#  MINE-TO-MILL · v2.0 — Aplicativo Web Avanzado
#  Nuevos módulos: Swebrec, Zonas Geológicas, 3D, Comparador, CSV, Reporte
#
#  Requisitos:  pip install -r requirements.txt
#  Ejecución:   streamlit run app.py
# =============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import io
from datetime import datetime

try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

st.set_page_config(
    page_title="Mine-to-Mill v2.0",
    page_icon="⛏",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# SECCIÓN 1 — MODELOS FÍSICOS
# =============================================================================

def lilly_a(UCS, GSI, H):
    RMD = 22 if GSI > 65 else 18 if GSI > 40 else 10
    JPS = 50 if GSI > 60 else 40 if GSI > 40 else 20
    return max(2.0, 0.06 * (RMD + JPS + 20 + min(50, UCS / 4) + min(H, 15)))


def ash_mesh(d_mm):
    B = 0.030 * d_mm
    return B, 1.15 * B, 0.80 * B


def kuz_ram(A, d_mm, B, S, H, T, rho_e, rws, J=0.0):
    """Kuz-Ram con parámetro de sobreperforación J (m)."""
    dm  = d_mm / 1000.0
    hc  = max(0.1, H - T + J)
    Q   = np.pi * (dm / 2) ** 2 * hc * rho_e
    V   = B * S * H
    PF  = Q / V
    denom_T = H + max(J, 0.01)
    nu  = float(np.clip(
        (2.2 - 14 * dm / B) * np.sqrt((1 + S / B) / 2) * (1 - T / denom_T),
        0.5, 3.0
    ))
    X50 = A * (V / Q) ** 0.8 * Q ** 0.167 * (115 / rws) ** 0.633
    F80 = X50 * 2.321 ** (1.0 / nu)
    return Q, V, PF, nu, X50, F80


def bond_w(Wi, F80_cm, P80_um):
    return max(0.0, 10 * Wi * (1 / np.sqrt(P80_um) - 1 / np.sqrt(F80_cm * 1e4)))


def ppv_calc(Q, R):
    return 1140 * (max(10.0, R) / np.sqrt(max(1.0, Q))) ** -1.6


def rosin_r(X50_mm, n, pts=120):
    lo = max(1.0, X50_mm * 0.02)
    hi = X50_mm * 12
    x  = np.geomspace(lo, hi, pts)
    P  = (1 - np.exp(-0.693 * (x / X50_mm) ** n)) * 100
    return x, P


def swebrec(X50_mm, xmax_mm, b=2.5, pts=120):
    """Distribución de Swebrec (Ouchterlony, 2005). Más precisa en los extremos."""
    lo = max(1.0, X50_mm * 0.02)
    hi = min(xmax_mm * 0.999, xmax_mm - 1)
    x  = np.geomspace(lo, hi, pts)
    ratio = np.log(xmax_mm / x) / np.log(xmax_mm / X50_mm)
    P = 1 / (1 + ratio ** b) * 100
    return x, P


@st.cache_data
def run_mc(A, sA, Wi, sWi, d, B, S, H, T, rho_e, rws, P80, Ce, Cn, rhoR, J=0.0, N=5000):
    rng  = np.random.default_rng(42)
    As   = np.maximum(1.5, A  + sA  * rng.standard_normal(N))
    Ws   = np.maximum(1.0, Wi + sWi * rng.standard_normal(N))
    dm   = d / 1000.0
    hc   = max(0.1, H - T + J)
    Q_   = np.pi * (dm / 2) ** 2 * hc * rho_e
    V_   = B * S * H
    PF_  = Q_ / V_
    dT   = H + max(J, 0.01)
    nu_  = float(np.clip(
        (2.2 - 14 * dm / B) * np.sqrt((1 + S / B) / 2) * (1 - T / dT), 0.5, 3.0
    ))
    X50s = As * (V_ / Q_) ** 0.8 * Q_ ** 0.167 * (115 / rws) ** 0.633
    F80s = X50s * 2.321 ** (1.0 / nu_)
    Ws_e = np.maximum(0, 10 * Ws * (1 / np.sqrt(P80) - 1 / np.sqrt(F80s * 1e4)))
    cv   = np.full(N, PF_ / rhoR * Ce)
    cm   = Ws_e * Cn
    return pd.DataFrame({
        "X50_mm": X50s * 10, "F80_mm": F80s * 10, "W": Ws_e,
        "cv": cv, "cm": cm, "ct": cv + cm,
    })


def calc_opt(A, H, rhoR, Wi, P80, Ce, Cn, rho_e, rws, factor, J=0.0):
    diams = [89, 102, 115, 127, 140, 152, 165, 178, 200, 229, 251, 311]
    rows  = []
    for d_ in diams:
        B_, S_, T_ = ash_mesh(d_)
        _, _, PF_, _, _, F80_ = kuz_ram(A, d_, B_, S_, H, T_, rho_e, rws, J)
        W_  = bond_w(Wi, F80_, P80)
        cv_ = PF_ / rhoR * Ce * factor
        cm_ = W_ * Cn
        rows.append({"d": d_, "C_vol": cv_, "C_mol": cm_, "C_total": cv_ + cm_})
    return pd.DataFrame(rows)


def kuz_ram_vec(A_arr, d_arr, B_arr, S_arr, H_arr, T_arr, rho_e, rws):
    """Kuz-Ram vectorizado para CSV de campo (arrays numpy)."""
    dm   = d_arr / 1000.0
    hc   = np.maximum(0.1, H_arr - T_arr)
    Q    = np.pi * (dm / 2) ** 2 * hc * rho_e
    V    = B_arr * S_arr * H_arr
    PF   = Q / V
    nu   = np.clip(
        (2.2 - 14 * dm / B_arr) * np.sqrt((1 + S_arr / B_arr) / 2) * (1 - T_arr / H_arr),
        0.5, 3.0
    )
    X50  = A_arr * (V / Q) ** 0.8 * Q ** 0.167 * (115 / rws) ** 0.633
    F80  = X50 * 2.321 ** (1.0 / nu)
    return Q, PF, X50 * 10, F80 * 10   # X50 y F80 en mm


# =============================================================================
# SECCIÓN 2 — BASE DE DATOS DE ROCAS Y EXPLOSIVOS
# =============================================================================

ROCK_DB = {
    "── Seleccionar roca ──": None,
    "Granito":               {"UCS": 150, "GSI": 65, "Wi": 16.0, "rhoR": 2.65},
    "Andesita":              {"UCS":  90, "GSI": 55, "Wi": 14.0, "rhoR": 2.70},
    "Pórfido de cobre":      {"UCS":  80, "GSI": 60, "Wi": 13.0, "rhoR": 2.75},
    "Caliza":                {"UCS":  60, "GSI": 50, "Wi": 10.0, "rhoR": 2.60},
    "Cuarcita":              {"UCS": 200, "GSI": 70, "Wi": 20.0, "rhoR": 2.65},
    "Esquisto":              {"UCS":  40, "GSI": 35, "Wi":  8.0, "rhoR": 2.55},
    "Diorita":               {"UCS": 120, "GSI": 60, "Wi": 15.0, "rhoR": 2.80},
    "Basalto":               {"UCS": 200, "GSI": 65, "Wi": 17.0, "rhoR": 2.90},
}

EXPL = {
    # ── Mezclas ANFO ─────────────────────────────────────────────────────────
    "ANFO (a granel)":      {"rws": 100, "rho": 850,  "factor": 1.00, "tipo": "ANFO",
                              "desc": "Estándar industria. Sensible al agua. Solo taladros secos."},
    "ANFO Pesado 50/50":    {"rws": 110, "rho": 1050, "factor": 1.15, "tipo": "Heavy ANFO",
                              "desc": "50% ANFO + 50% emulsión. Mayor energía y resistencia al agua."},
    "ANFO Pesado 70/30":    {"rws": 118, "rho": 1150, "factor": 1.30, "tipo": "Heavy ANFO",
                              "desc": "70% ANFO + 30% emulsión. Alta energía. Roca dura-media."},
    "ANFO Pesado 80/20":    {"rws": 122, "rho": 1200, "factor": 1.40, "tipo": "Heavy ANFO",
                              "desc": "80% ANFO + 20% emulsión. Máxima energía en mezclas ANFO."},
    # ── Emulsiones ───────────────────────────────────────────────────────────
    "Emulsión Gasificada":  {"rws": 105, "rho": 1100, "factor": 1.45, "tipo": "Emulsión",
                              "desc": "Microesferas + emulsión. Alta resistencia al agua. Densidad ajustable."},
    "Emulsión Estándar":    {"rws": 115, "rho": 1250, "factor": 1.60, "tipo": "Emulsión",
                              "desc": "Alta energía y resistencia al agua. Ideal taladros inundados."},
    "Emulsión Encartuchada":{"rws": 110, "rho": 1200, "factor": 1.75, "tipo": "Emulsión",
                              "desc": "Cartuchos para taladros pequeños o voladuras controladas."},
    # ── Slurry / Watergel ────────────────────────────────────────────────────
    "Slurry / Watergel":    {"rws":  80, "rho": 1200, "factor": 1.50, "tipo": "Slurry",
                              "desc": "Resistente al agua. Menor energía que emulsión."},
    # ── Explosivos especiales ─────────────────────────────────────────────────
    "Dinamita 60%":         {"rws": 115, "rho": 1300, "factor": 3.20, "tipo": "Dinamita",
                              "desc": "Alta potencia. Minería subterránea o roca muy dura. Alto costo."},
    "Pentolita (booster)":  {"rws": 125, "rho": 1600, "factor": 8.00, "tipo": "Especial",
                              "desc": "Iniciador/booster. No usar como carga principal de columna."},
}


# =============================================================================
# SECCIÓN 3 — COLORES Y LAYOUT
# =============================================================================

AMBER = "#e8a000"
GREEN = "#3a8a46"
RED   = "#8a3a3a"
BLUE  = "#5060a0"
DIM   = "#64584e"

BASE  = dict(
    template="plotly_dark",
    paper_bgcolor="#121416",
    plot_bgcolor="#0b0c0e",
    font=dict(family="monospace", color="#d2c8bc", size=11),
    margin=dict(l=55, r=25, t=40, b=50),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)
TT = {"contentStyle": {"background": "#121416", "border": "0.5px solid #1e2022",
                       "borderRadius": "4px", "fontSize": "11px"}}


# =============================================================================
# SECCIÓN 4 — SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("## ⛏ Mine-to-Mill v2.0")
    st.caption("Kuz-Ram · Swebrec · Bond · Monte Carlo")
    st.divider()

    # Base de datos de rocas
    st.markdown("### 🗄️ Base de Datos de Rocas")
    roca_sel = st.selectbox("Tipo de roca", list(ROCK_DB.keys()))
    roca = ROCK_DB[roca_sel]

    st.markdown("### ⛏ Geometría de Mina")
    d     = st.number_input("Diámetro d (mm)", 75, 500,
                             int(roca["UCS"] // 5 * 5) if roca else 165, step=5)
    H     = st.number_input("Altura de banco H (m)", 3.0, 30.0, 10.0, step=0.5)
    J     = st.number_input("Sobreperforación J (m)", 0.0, 5.0, 1.0, step=0.1,
                             help="Longitud perforada bajo el piso del banco (subdrill)")
    rho_r = st.number_input("Densidad ρ_r (t/m³)", 1.0, 4.0,
                             float(roca["rhoR"]) if roca else 2.7, step=0.05, format="%.2f")

    st.markdown("### 🪨 Macizo Rocoso")
    UCS = st.number_input("UCS (MPa)", 10, 300,
                           int(roca["UCS"]) if roca else 80, step=5)
    GSI = st.slider("GSI", 10, 100, int(roca["GSI"]) if roca else 55)

    st.markdown("### 💥 Explosivo")
    # Agrupar por tipo para el selectbox
    tipos = sorted(set(e["tipo"] for e in EXPL.values()))
    tipo_sel = st.selectbox("Categoría", tipos)
    opciones_filtradas = [k for k,v in EXPL.items() if v["tipo"] == tipo_sel]
    expl_name = st.selectbox("Explosivo", opciones_filtradas)
    E = EXPL[expl_name]
    st.info(
        f"**RWS:** {E['rws']}  ·  **ρ:** {E['rho']/1000:.2f} g/cc  ·  **Factor precio:** ×{E['factor']:.2f}\n\n"
        f"{E['desc']}"
    )

    st.markdown("### 🏭 Planta")
    Wi  = st.number_input("Wi (kWh/t)", 1.0, 50.0,
                           float(roca["Wi"]) if roca else 14.0, step=0.5)
    sWi = st.number_input("σ(Wi)", 0.1, 10.0, 2.0, step=0.1, format="%.1f")
    P80 = st.number_input("P₈₀ (μm)", 10, 1000, 150, step=10)

    st.markdown("### 💰 Economía")
    Ce = st.number_input("Explosivo ($/kg)", 0.10, 5.0, 0.80, step=0.05, format="%.2f")
    Cn = st.number_input("Electricidad ($/kWh)", 0.005, 0.5, 0.060, step=0.005, format="%.3f")

    st.markdown("### 🎲 Monte Carlo")
    sA       = st.number_input("σ(A)", 0.1, 5.0, 1.5, step=0.1, format="%.1f")
    R_struct = st.number_input("Distancia estructura (m)", 10, 2000, 150, step=10)

    # Parámetro Swebrec
    st.markdown("### 🌊 Swebrec")
    b_sw  = st.slider("Parámetro b", 1.0, 5.0, 2.5, step=0.1)
    xmax_factor = st.slider("xmax = X₅₀ ×", 4, 15, 8)


# =============================================================================
# SECCIÓN 5 — CÁLCULOS BASE
# =============================================================================

B, S, T  = ash_mesh(d)
A_val    = lilly_a(UCS, GSI, H)
Q, V, PF, nu, X50, F80 = kuz_ram(A_val, d, B, S, H, T, E["rho"], E["rws"], J)
X50mm    = X50 * 10
F80mm    = F80 * 10
W_base   = bond_w(Wi, F80, P80)
Ce_adj   = Ce * E["factor"]
ppv_cur  = ppv_calc(Q, R_struct)
xmax_sw  = X50mm * xmax_factor

mc = run_mc(A_val, sA, Wi, sWi, d, B, S, H, T, E["rho"], E["rws"],
            P80, Ce_adj, Cn, rho_r, J)
P10, P50, P90 = np.percentile(mc["ct"], [10, 50, 90])

x_rr, P_rr = rosin_r(X50mm, nu)
x_sw, P_sw = swebrec(X50mm, xmax_sw, b_sw)

df_opt = calc_opt(A_val, H, rho_r, Wi, P80, Ce, Cn, E["rho"], E["rws"], E["factor"], J)
opt_row = df_opt.loc[df_opt["C_total"].idxmin()]


# =============================================================================
# SECCIÓN 6 — ENCABEZADO
# =============================================================================

st.title("⛏ Mine-to-Mill v2.0")
st.caption(f"Explosivo: **{expl_name}** · J = {J} m · Roca: **{roca_sel if roca else 'Manual'}**")

ppv_icon = "🔴" if ppv_cur > 100 else "🟡" if ppv_cur > 50 else "🟢"
for col, (lbl, val) in zip(st.columns(7), [
    ("d (mm)", str(d)), ("A", f"{A_val:.2f}"),
    ("X₅₀ (mm)", f"{X50mm:.0f}"), ("F₈₀ (mm)", f"{F80mm:.0f}"),
    ("W (kWh/t)", f"{W_base:.2f}"), ("P50 ($/t)", f"{P50:.4f}"),
    (f"PPV {ppv_icon}", f"{ppv_cur:.0f}"),
]):
    col.metric(lbl, val)

st.divider()


# =============================================================================
# SECCIÓN 7 — TABS
# =============================================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "🔩 Diseño Malla", "🪨 Fragmentación", "🎲 Riesgo MC",
    "💰 Economía",     "⚙️ Optimizador",   "🔊 Vibraciones",
    "🗿 Zonas Geológicas", "⚖️ Comparador",
    "📂 Datos CSV",    "📋 Reporte",
])


# ════════════════════════════════════════════════════════════════════
# TAB 1 — DISEÑO DE MALLA + VISTA 3D
# ════════════════════════════════════════════════════════════════════
with tab1:
    for col, (lbl, val, tip) in zip(st.columns(4), [
        ("Burden B (m)", f"{B:.2f}", "= 30 × d"),
        ("Espaciamiento S (m)", f"{S:.2f}", "= 1.15 × B"),
        ("Taco T (m)", f"{T:.2f}", "= 0.80 × B"),
        ("Sobreperforación J (m)", f"{J:.2f}", "Subdrill"),
    ]):
        col.metric(lbl, val, help=tip)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Parámetros de Diseño")
        st.dataframe(pd.DataFrame({
            "Parámetro": [
                "Longitud carga útil", "Masa explosivo/hole (Q)",
                "Volumen roca/hole (V)", "PF volumétrico",
                "PF másico", "Índice uniformidad n",
            ],
            "Valor": [
                f"{H - T + J:.2f} m", f"{Q:.1f} kg",
                f"{V:.1f} m³", f"{PF:.3f} kg/m³",
                f"{PF/rho_r*1000:.1f} g/t", f"{nu:.3f}",
            ],
        }), hide_index=True, use_container_width=True)

    with col_right:
        st.markdown("#### Vista en Planta (2D)")
        cols_h, rows_h = 5, 4
        x_h = [(c + 0.5) * S for r in range(rows_h) for c in range(cols_h)]
        y_h = [(r + 0.5) * B for r in range(rows_h) for c in range(cols_h)]
        fig2d = go.Figure()
        fig2d.add_trace(go.Scatter(
            x=x_h, y=y_h, mode="markers",
            marker=dict(size=max(8, d / 9), color=AMBER, opacity=0.85),
            name=f"ø{d}mm",
        ))
        for i in range(cols_h + 1):
            fig2d.add_vline(x=i*S, line_dash="dash", line_color="#2a2c2e", line_width=0.8)
        for i in range(rows_h + 1):
            fig2d.add_hline(y=i*B, line_dash="dash", line_color="#2a2c2e", line_width=0.8)
        fig2d.update_layout(**BASE, height=280,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"))
        st.plotly_chart(fig2d, use_container_width=True)

    # ── VISTA 3D ─────────────────────────────────────────────────────────────
    st.markdown("#### Modelo 3D del Banco de Perforación")
    fig3d = go.Figure()
    for row in range(rows_h):
        for col in range(cols_h):
            xp, yp = (col + 0.5) * S, (row + 0.5) * B
            # Taco (stemming)
            fig3d.add_trace(go.Scatter3d(
                x=[xp, xp], y=[yp, yp], z=[0, -T],
                mode="lines", line=dict(color="#888", width=6),
                showlegend=(row == 0 and col == 0),
                name="Taco",
            ))
            # Carga explosiva
            fig3d.add_trace(go.Scatter3d(
                x=[xp, xp], y=[yp, yp], z=[-T, -(H - 0.1)],
                mode="lines", line=dict(color=AMBER, width=6),
                showlegend=(row == 0 and col == 0),
                name="Explosivo",
            ))
            # Sobreperforación
            if J > 0:
                fig3d.add_trace(go.Scatter3d(
                    x=[xp, xp], y=[yp, yp], z=[-(H - 0.1), -(H + J)],
                    mode="lines", line=dict(color=GREEN, width=4, dash="dot"),
                    showlegend=(row == 0 and col == 0),
                    name="Sobreperf.",
                ))
    # Superficie del banco
    xs = np.array([0, cols_h * S, cols_h * S, 0])
    ys = np.array([0, 0, rows_h * B, rows_h * B])
    zs = np.zeros(4)
    fig3d.add_trace(go.Mesh3d(
        x=xs, y=ys, z=zs,
        color="#2a2c2e", opacity=0.4, showlegend=False,
    ))
    fig3d.update_layout(
        **BASE, height=420,
        scene=dict(
            xaxis=dict(title="Espaciamiento (m)", color=DIM, gridcolor="#1e2022"),
            yaxis=dict(title="Burden (m)", color=DIM, gridcolor="#1e2022"),
            zaxis=dict(title="Profundidad (m)", color=DIM, gridcolor="#1e2022"),
            bgcolor="#0b0c0e",
            camera=dict(eye=dict(x=1.5, y=-1.8, z=1.2)),
        ),
        title=f"Malla {cols_h}×{rows_h} · B={B:.1f}m · S={S:.1f}m · H={H}m",
    )
    st.plotly_chart(fig3d, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# TAB 2 — FRAGMENTACIÓN + SWEBREC
# ════════════════════════════════════════════════════════════════════
with tab2:
    P300 = (1 - np.exp(-0.693 * (300 / X50mm) ** nu)) * 100
    for col, (lbl, val) in zip(st.columns(4), [
        ("X₅₀ (mm)", f"{X50mm:.0f}"),
        ("F₈₀ (mm)", f"{F80mm:.0f}"),
        ("n uniformidad", f"{nu:.2f}"),
        ("Sobresize >300mm", f"{100-P300:.1f}%"),
    ]):
        col.metric(lbl, val)

    # Curva comparativa Rosin-Rammler vs Swebrec
    fig_frag = go.Figure()
    fig_frag.add_trace(go.Scatter(
        x=x_rr, y=P_rr, mode="lines", name="Rosin-Rammler",
        line=dict(color=AMBER, width=2.5),
        fill="tozeroy", fillcolor="rgba(232,160,0,0.10)",
    ))
    fig_frag.add_trace(go.Scatter(
        x=x_sw, y=P_sw, mode="lines", name=f"Swebrec (b={b_sw})",
        line=dict(color=BLUE, width=2.5, dash="dot"),
    ))
    fig_frag.add_vline(x=X50mm, line_dash="dash", line_color="#7a5000",
                       annotation_text="X₅₀", annotation_font_color="#7a5000")
    fig_frag.add_vline(x=F80mm, line_dash="dash", line_color=AMBER,
                       annotation_text="F₈₀", annotation_font_color=AMBER)
    fig_frag.add_vline(x=xmax_sw, line_dash="dot", line_color=DIM,
                       annotation_text="xmax", annotation_font_color=DIM)
    fig_frag.add_hline(y=80, line_dash="dot", line_color=DIM,
                       annotation_text="80%", annotation_position="right")
    fig_frag.update_layout(
        **BASE, height=360,
        title="Curva Granulométrica — Rosin-Rammler vs Swebrec (Ouchterlony 2005)",
        xaxis=dict(type="log", title="Tamaño (mm)", gridcolor="#1e2022"),
        yaxis=dict(title="% Pasante", range=[0, 100], gridcolor="#1e2022"),
    )
    st.plotly_chart(fig_frag, use_container_width=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### Energía de Molienda — 3ª Ley de Bond")
        st.metric("W", f"{W_base:.3f} kWh/t")
        st.code(f"W = 10·{Wi}·(1/√{P80} − 1/√{F80*1e4:.0f}) = {W_base:.3f} kWh/t")
    with col_right:
        st.markdown("#### Diferencia Rosin-Rammler vs Swebrec")
        # F80 de Swebrec: interpolación numérica directa
        idx80 = np.searchsorted(P_sw, 80)
        idx80 = min(idx80, len(x_sw) - 1)
        f80_sw_val = float(x_sw[idx80])
        st.metric("F₈₀ Rosin-Rammler", f"{F80mm:.0f} mm")
        st.metric("F₈₀ Swebrec (aprox.)", f"{f80_sw_val:.0f} mm",
                  delta=f"{f80_sw_val - F80mm:+.0f} mm vs R-R")
        if abs(f80_sw_val - F80mm) / F80mm > 0.15:
            st.warning("Diferencia > 15% entre modelos. El parámetro b de Swebrec requiere calibración con datos de campo.")
        else:
            st.success("Buena concordancia entre modelos.")


# ════════════════════════════════════════════════════════════════════
# TAB 3 — RIESGO MC
# ════════════════════════════════════════════════════════════════════
with tab3:
    c1, c2, c3 = st.columns(3)
    c1.metric("P10 — Optimista ($/t)", f"{P10:.4f}")
    c2.metric("P50 — Esperado ($/t)",  f"{P50:.4f}")
    c3.metric("P90 — Pesimista ($/t)", f"{P90:.4f}")

    ct_arr = mc["ct"].values
    counts, bins = np.histogram(ct_arr, bins=30)
    bin_mid = (bins[:-1] + bins[1:]) / 2
    bar_colors = [RED if b >= P90 else GREEN if b <= P10 else AMBER for b in bin_mid]

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Bar(x=bin_mid, y=counts, marker_color=bar_colors,
                              marker_opacity=0.7, showlegend=False))
    for xv, lbl, c in [(P10,"P10",GREEN),(P50,"P50",AMBER),(P90,"P90",RED)]:
        fig_hist.add_vline(x=xv, line_dash="dash", line_color=c,
                           annotation_text=lbl, annotation_font_color=c)
    fig_hist.update_layout(**BASE, height=280,
        title="Distribución Monte Carlo — 5 000 Escenarios",
        xaxis=dict(title="Costo Total ($/t)", gridcolor="#1e2022"),
        yaxis=dict(title="N° Escenarios", gridcolor="#1e2022"))
    st.plotly_chart(fig_hist, use_container_width=True)

    col_left, col_right = st.columns(2)
    with col_left:
        for pp, lbl in [(10,"Optimista"),(50,"Esperado"),(90,"Pesimista")]:
            st.metric(f"F₈₀ P{pp} — {lbl}", f"{np.percentile(mc['F80_mm'],pp):.0f} mm")
    with col_right:
        idx = (P90-P10)/P50*100
        st.metric("Índice de incertidumbre", f"{idx:.1f}%")
        if idx > 40:
            st.warning("⚠ Alta incertidumbre. Aumentar muestreo geomecánico.")
        else:
            st.success("✓ Diseño robusto.")


# ════════════════════════════════════════════════════════════════════
# TAB 4 — ECONOMÍA
# ════════════════════════════════════════════════════════════════════
with tab4:
    cv_p50 = float(np.percentile(mc["cv"], 50))
    cm_p50 = float(np.percentile(mc["cm"], 50))

    col_left, col_right = st.columns(2)
    with col_left:
        pcts = [10, 50, 90]
        df_ec = pd.DataFrame({
            "Percentil": [f"P{p}" for p in pcts],
            "Voladura":  [float(np.percentile(mc["cv"],p)) for p in pcts],
            "Molienda":  [float(np.percentile(mc["cm"],p)) for p in pcts],
        })
        fig_ec = go.Figure()
        fig_ec.add_trace(go.Bar(name="Voladura", x=df_ec["Percentil"], y=df_ec["Voladura"], marker_color=AMBER))
        fig_ec.add_trace(go.Bar(name="Molienda", x=df_ec["Percentil"], y=df_ec["Molienda"], marker_color=BLUE))
        fig_ec.update_layout(**BASE, barmode="stack", height=260,
            yaxis=dict(title="$/t", gridcolor="#1e2022"), title="Desglose por Percentil")
        st.plotly_chart(fig_ec, use_container_width=True)

    with col_right:
        def _ct(p):
            B_,S_,T_ = ash_mesh(p["d"])
            aa = lilly_a(p["UCS"],p["GSI"],p["H"])
            _,_,pf_,_,_,f80_ = kuz_ram(aa,p["d"],B_,S_,p["H"],T_,E["rho"],E["rws"])
            return (pf_/p["rhoR"])*p["Ce"]*E["factor"] + bond_w(p["Wi"],f80_,p["P80"])*Cn

        base_p = dict(d=d,H=H,UCS=UCS,GSI=GSI,Wi=Wi,Ce=Ce,rhoR=rho_r,P80=P80)
        tor_r = []
        for nm,key,lo,hi in [("Wi","Wi",0.75,1.25),("d","d",0.80,1.20),
                               ("UCS","UCS",0.70,1.30),("GSI","GSI",0.80,1.20),
                               ("Ce","Ce",0.80,1.20)]:
            tor_r.append({"Variable":nm,"Rango":abs(_ct({**base_p,key:base_p[key]*hi})-_ct({**base_p,key:base_p[key]*lo}))})
        df_tor = pd.DataFrame(tor_r).sort_values("Rango",ascending=True)
        fig_tor = go.Figure(go.Bar(x=df_tor["Rango"],y=df_tor["Variable"],
                                   orientation="h",marker_color=AMBER,marker_opacity=0.8))
        fig_tor.update_layout(**BASE,height=260,
            xaxis=dict(title="Rango ($/t)",gridcolor="#1e2022"),title="Diagrama Tornado")
        st.plotly_chart(fig_tor,use_container_width=True)

    for col,(lbl,val) in zip(st.columns(4),[
        ("C.Voladura P50",f"{cv_p50:.4f} $/t"),
        ("C.Molienda P50",f"{cm_p50:.4f} $/t"),
        ("C.Total P50",f"{P50:.4f} $/t"),
        ("Ratio Mol/Vol",f"{cm_p50/cv_p50:.2f}×"),
    ]):
        col.metric(lbl,val)


# ════════════════════════════════════════════════════════════════════
# TAB 5 — OPTIMIZADOR + FRONTERA DE PARETO
# ════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown(f"**Diámetro óptimo: {int(opt_row['d'])} mm · C_total mínimo: {opt_row['C_total']:.4f} $/t**")

    fig_opt = go.Figure()
    for col_key, name, color, width in [
        ("C_vol","Voladura",AMBER,2),("C_mol","Molienda",BLUE,2),("C_total","Total","#e0e0e0",2.5)
    ]:
        fig_opt.add_trace(go.Scatter(x=df_opt["d"],y=df_opt[col_key],name=name,
            line=dict(color=color,width=width),mode="lines+markers"))
    fig_opt.add_vline(x=opt_row["d"],line_dash="dash",line_color=AMBER,
        annotation_text=f"Óptimo {int(opt_row['d'])}mm",annotation_font_color=AMBER)
    if d != int(opt_row["d"]):
        fig_opt.add_vline(x=d,line_dash="dot",line_color=DIM,
            annotation_text="Actual",annotation_font_color=DIM)
    fig_opt.update_layout(**BASE,height=320,
        title="Curva Trade-off Mine-to-Mill",
        xaxis=dict(title="Diámetro (mm)",gridcolor="#1e2022"),
        yaxis=dict(title="Costo ($/t)",gridcolor="#1e2022"))
    st.plotly_chart(fig_opt,use_container_width=True)

    # Frontera de Pareto (scatter C_vol vs C_mol)
    fig_pareto = go.Figure()
    fig_pareto.add_trace(go.Scatter(
        x=df_opt["C_vol"], y=df_opt["C_mol"],
        mode="markers+text",
        marker=dict(size=12, color=df_opt["C_total"],
                    colorscale="YlOrRd_r", showscale=True,
                    colorbar=dict(title="C_total $/t", thickness=12)),
        text=[str(d_) for d_ in df_opt["d"]],
        textposition="top center",
        textfont=dict(size=9, color="#d2c8bc"),
        name="Diámetros",
    ))
    opt_x = float(df_opt.loc[df_opt["C_total"].idxmin(), "C_vol"])
    opt_y = float(df_opt.loc[df_opt["C_total"].idxmin(), "C_mol"])
    fig_pareto.add_trace(go.Scatter(
        x=[opt_x], y=[opt_y], mode="markers",
        marker=dict(size=18, color=AMBER, symbol="star"),
        name="Óptimo",
    ))
    fig_pareto.update_layout(**BASE,height=320,
        title="Frontera de Pareto — Costo Voladura vs Molienda",
        xaxis=dict(title="C. Voladura ($/t)",gridcolor="#1e2022"),
        yaxis=dict(title="C. Molienda ($/t)",gridcolor="#1e2022"))
    st.plotly_chart(fig_pareto,use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# TAB 6 — VIBRACIONES
# ════════════════════════════════════════════════════════════════════
with tab6:
    ppv_status = "⚠ PELIGRO" if ppv_cur>100 else "◉ PRECAUCIÓN" if ppv_cur>50 else "✓ SEGURO"
    c1,c2,c3 = st.columns(3)
    c1.metric("PPV (mm/s)",f"{ppv_cur:.1f}")
    c2.metric("Q/hole (kg)",f"{Q:.1f}")
    c3.metric("Distancia (m)",R_struct)
    if ppv_cur>100: st.error(ppv_status)
    elif ppv_cur>50: st.warning(ppv_status)
    else: st.success(ppv_status)

    Rs = np.linspace(30,600,150)
    PPVs = [ppv_calc(Q,r) for r in Rs]
    fig_ppv = go.Figure()
    fig_ppv.add_trace(go.Scatter(x=Rs,y=PPVs,mode="lines",name="PPV",
        line=dict(color=AMBER,width=2.5),fill="tozeroy",fillcolor="rgba(232,160,0,0.10)"))
    for yl,lbl,c in [(25,"25",GREEN),(50,"50",AMBER),(100,"100",RED)]:
        fig_ppv.add_hline(y=yl,line_dash="dash",line_color=c,
            annotation_text=f"{lbl} mm/s",annotation_position="right",annotation_font_color=c)
    ppv_lc = RED if ppv_cur>100 else AMBER if ppv_cur>50 else GREEN
    fig_ppv.add_vline(x=R_struct,line_dash="dot",line_color=ppv_lc,
        annotation_text=f"{ppv_cur:.0f}mm/s",annotation_font_color=ppv_lc)
    fig_ppv.update_layout(**BASE,height=320,
        title="PPV vs Distancia — Holmberg-Persson",
        xaxis=dict(title="Distancia (m)",gridcolor="#1e2022"),
        yaxis=dict(title="PPV (mm/s)",gridcolor="#1e2022"))
    st.plotly_chart(fig_ppv,use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# TAB 7 — ZONAS GEOLÓGICAS
# ════════════════════════════════════════════════════════════════════
with tab7:
    st.markdown("#### Fragmentación por Zonas Geológicas")
    st.caption("Divide el banco en 2 zonas con distintas propiedades. Calcula el F₈₀ ponderado real que llega a planta.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Zona 1")
        ucs1 = st.number_input("UCS Zona 1 (MPa)", 10, 300, UCS, step=5, key="ucs1")
        gsi1 = st.slider("GSI Zona 1", 10, 100, GSI, key="gsi1")
        wi1  = st.number_input("Wi Zona 1 (kWh/t)", 1.0, 50.0, Wi, step=0.5, key="wi1")
        vol1 = st.slider("Volumen Zona 1 (%)", 10, 90, 60, key="vol1")
    with col2:
        st.markdown("##### Zona 2")
        ucs2 = st.number_input("UCS Zona 2 (MPa)", 10, 300, max(10,UCS-30), step=5, key="ucs2")
        gsi2 = st.slider("GSI Zona 2", 10, 100, max(10,GSI-15), key="gsi2")
        wi2  = st.number_input("Wi Zona 2 (kWh/t)", 1.0, 50.0, max(1.0,Wi-3.0), step=0.5, key="wi2")
        vol2 = 100 - vol1
        st.metric("Volumen Zona 2 (%)", vol2)

    A1 = lilly_a(ucs1, gsi1, H)
    A2 = lilly_a(ucs2, gsi2, H)
    _, _, _, nu1, X50_1, F80_1 = kuz_ram(A1, d, B, S, H, T, E["rho"], E["rws"], J)
    _, _, _, nu2, X50_2, F80_2 = kuz_ram(A2, d, B, S, H, T, E["rho"], E["rws"], J)

    X50_1mm, F80_1mm = X50_1*10, F80_1*10
    X50_2mm, F80_2mm = X50_2*10, F80_2*10
    f1, f2 = vol1/100, vol2/100
    X50_pond = f1*X50_1mm + f2*X50_2mm
    F80_pond  = f1*F80_1mm  + f2*F80_2mm
    W1 = bond_w(wi1, F80_1, P80)
    W2 = bond_w(wi2, F80_2, P80)
    W_pond = f1*W1 + f2*W2

    # Métricas
    c1,c2,c3 = st.columns(3)
    c1.metric("F₈₀ Zona 1", f"{F80_1mm:.0f} mm", delta=f"A={A1:.2f}")
    c2.metric("F₈₀ Zona 2", f"{F80_2mm:.0f} mm", delta=f"A={A2:.2f}")
    c3.metric("F₈₀ Ponderado", f"{F80_pond:.0f} mm",
              delta=f"{F80_pond-F80mm:+.0f} mm vs una zona")

    # Gráfico comparativo de curvas
    x1_rr, P1_rr = rosin_r(X50_1mm, nu1)
    x2_rr, P2_rr = rosin_r(X50_2mm, nu2)
    xp_rr, Pp_rr = rosin_r(X50_pond, (nu1+nu2)/2)

    fig_z = go.Figure()
    fig_z.add_trace(go.Scatter(x=x1_rr,y=P1_rr,mode="lines",name=f"Zona 1 ({vol1}%)",
        line=dict(color=AMBER,width=2)))
    fig_z.add_trace(go.Scatter(x=x2_rr,y=P2_rr,mode="lines",name=f"Zona 2 ({vol2}%)",
        line=dict(color=BLUE,width=2)))
    fig_z.add_trace(go.Scatter(x=xp_rr,y=Pp_rr,mode="lines",name="Ponderado (alimento real)",
        line=dict(color="#e0e0e0",width=2.5,dash="dot")))
    fig_z.add_hline(y=80,line_dash="dot",line_color=DIM,annotation_text="80%",annotation_position="right")
    fig_z.update_layout(**BASE,height=320,
        title="Curvas Granulométricas por Zona",
        xaxis=dict(type="log",title="Tamaño (mm)",gridcolor="#1e2022"),
        yaxis=dict(title="% Pasante",range=[0,100],gridcolor="#1e2022"))
    st.plotly_chart(fig_z,use_container_width=True)

    # Tabla resumen zonas
    st.dataframe(pd.DataFrame({
        "": ["Zona 1","Zona 2","Ponderado"],
        "Volumen (%)": [vol1, vol2, 100],
        "A (Lilly)": [f"{A1:.2f}",f"{A2:.2f}","—"],
        "X₅₀ (mm)": [f"{X50_1mm:.0f}",f"{X50_2mm:.0f}",f"{X50_pond:.0f}"],
        "F₈₀ (mm)": [f"{F80_1mm:.0f}",f"{F80_2mm:.0f}",f"{F80_pond:.0f}"],
        "W (kWh/t)": [f"{W1:.2f}",f"{W2:.2f}",f"{W_pond:.2f}"],
    }), hide_index=True, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# TAB 8 — COMPARADOR DE ESCENARIOS
# ════════════════════════════════════════════════════════════════════
with tab8:
    st.markdown("#### Comparador de Escenarios de Diseño")
    st.caption("Compara el escenario actual (Sidebar) contra un escenario alternativo.")

    col_a, col_b = st.columns(2)
    with col_b:
        st.markdown("##### Escenario B — Alternativo")
        d_b    = st.number_input("Diámetro d_B (mm)", 75, 500, max(75, d+25), step=5, key="db")
        ucs_b  = st.number_input("UCS_B (MPa)", 10, 300, UCS, step=5, key="ucsb")
        gsi_b  = st.slider("GSI_B", 10, 100, GSI, key="gsib")
        wi_b   = st.number_input("Wi_B (kWh/t)", 1.0, 50.0, Wi, step=0.5, key="wib")
        expl_b = st.selectbox("Explosivo B", list(EXPL.keys()), key="explb")
        E_b    = EXPL[expl_b]

    # Calcular escenario B
    B_b, S_b, T_b = ash_mesh(d_b)
    A_b = lilly_a(ucs_b, gsi_b, H)
    Q_b, V_b, PF_b, nu_b, X50_b, F80_b = kuz_ram(A_b, d_b, B_b, S_b, H, T_b, E_b["rho"], E_b["rws"], J)
    W_b = bond_w(wi_b, F80_b, P80)
    mc_b = run_mc(A_b, sA, wi_b, sWi, d_b, B_b, S_b, H, T_b, E_b["rho"], E_b["rws"],
                  P80, Ce*E_b["factor"], Cn, rho_r, J)
    P50_b = float(np.percentile(mc_b["ct"], 50))

    with col_a:
        st.markdown("##### Escenario A — Actual (desde Sidebar)")
        st.dataframe(pd.DataFrame({
            "Parámetro": ["d (mm)","A (Lilly)","Explosivo","B (m)","X₅₀ (mm)","F₈₀ (mm)","W (kWh/t)","P50 ($/t)"],
            "Valor A":   [d,f"{A_val:.2f}",expl_name,f"{B:.2f}",f"{X50mm:.0f}",f"{F80mm:.0f}",f"{W_base:.2f}",f"{P50:.4f}"],
        }), hide_index=True, use_container_width=True)

    # Tabla comparativa
    st.markdown("#### Comparación Directa")
    delta_F80 = (F80_b*10) - F80mm
    delta_W   = W_b - W_base
    delta_P50 = P50_b - P50
    comp_df = pd.DataFrame({
        "Indicador":    ["Diámetro","Factor A","X₅₀","F₈₀","W molienda","C_vol P50","C_mol P50","C_total P50"],
        "Escenario A":  [f"{d}mm",f"{A_val:.2f}",f"{X50mm:.0f}mm",f"{F80mm:.0f}mm",
                         f"{W_base:.2f}kWh/t",f"{float(np.percentile(mc['cv'],50)):.4f}$/t",
                         f"{float(np.percentile(mc['cm'],50)):.4f}$/t",f"{P50:.4f}$/t"],
        "Escenario B":  [f"{d_b}mm",f"{A_b:.2f}",f"{X50_b*10:.0f}mm",f"{F80_b*10:.0f}mm",
                         f"{W_b:.2f}kWh/t",f"{float(np.percentile(mc_b['cv'],50)):.4f}$/t",
                         f"{float(np.percentile(mc_b['cm'],50)):.4f}$/t",f"{P50_b:.4f}$/t"],
        "Diferencia B-A":["—","—",f"{X50_b*10-X50mm:+.0f}mm",f"{delta_F80:+.0f}mm",
                          f"{delta_W:+.2f}kWh/t","—","—",f"{delta_P50:+.4f}$/t"],
    })
    st.dataframe(comp_df, hide_index=True, use_container_width=True)

    # Curvas superpuestas
    xb_rr, Pb_rr = rosin_r(X50_b*10, nu_b)
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=x_rr,y=P_rr,mode="lines",name="Escenario A",
        line=dict(color=AMBER,width=2.5),fill="tozeroy",fillcolor="rgba(232,160,0,0.08)"))
    fig_comp.add_trace(go.Scatter(x=xb_rr,y=Pb_rr,mode="lines",name="Escenario B",
        line=dict(color=BLUE,width=2.5),fill="tozeroy",fillcolor="rgba(80,96,160,0.08)"))
    fig_comp.add_hline(y=80,line_dash="dot",line_color=DIM,annotation_text="80%",annotation_position="right")
    fig_comp.update_layout(**BASE,height=300,
        title="Curvas Granulométricas — Comparación de Escenarios",
        xaxis=dict(type="log",title="Tamaño (mm)",gridcolor="#1e2022"),
        yaxis=dict(title="% Pasante",range=[0,100],gridcolor="#1e2022"))
    st.plotly_chart(fig_comp,use_container_width=True)

    winner = "A" if P50 < P50_b else "B"
    saving = abs(P50_b - P50)
    if winner == "A":
        st.success(f"✓ Escenario A es más económico por **{saving:.4f} $/t**")
    else:
        st.success(f"✓ Escenario B es más económico por **{saving:.4f} $/t**")


# ════════════════════════════════════════════════════════════════════
# TAB 9 — CARGA DE DATOS CSV
# ════════════════════════════════════════════════════════════════════
with tab9:
    st.markdown("#### Importar Datos de Campo (CSV)")

    # Plantilla descargable
    template_df = pd.DataFrame({
        "taladro_id":    ["T001","T002","T003"],
        "diametro_mm":   [165,   165,   165],
        "altura_banco_m":[10.0,  10.0,  10.0],
        "burden_m":      [4.95,  5.10,  4.80],
        "espaciamiento_m":[5.69, 5.85,  5.52],
        "taco_m":        [3.96,  4.08,  3.84],
        "UCS_MPa":       [80,    72,    92],
        "GSI":           [55,    50,    60],
    })
    csv_template = template_df.to_csv(index=False).encode()
    st.download_button("📥 Descargar plantilla CSV", csv_template,
                       "plantilla_campo.csv", "text/csv")

    uploaded = st.file_uploader("Subir CSV de campo", type=["csv"])

    if uploaded:
        df_campo = pd.read_csv(uploaded)
        st.success(f"✓ {len(df_campo)} taladros cargados.")
        st.dataframe(df_campo.head(10), use_container_width=True, hide_index=True)

        required = ["diametro_mm","altura_banco_m","burden_m","espaciamiento_m","taco_m","UCS_MPa","GSI"]
        if all(c in df_campo.columns for c in required):
            A_arr = np.array([lilly_a(u, g, h) for u, g, h in
                              zip(df_campo["UCS_MPa"], df_campo["GSI"], df_campo["altura_banco_m"])])
            Q_arr, PF_arr, X50_arr, F80_arr = kuz_ram_vec(
                A_arr, df_campo["diametro_mm"].values,
                df_campo["burden_m"].values, df_campo["espaciamiento_m"].values,
                df_campo["altura_banco_m"].values, df_campo["taco_m"].values,
                E["rho"], E["rws"],
            )
            W_arr = np.array([bond_w(Wi, f/10, P80) for f in F80_arr])

            df_campo["X50_mm"] = X50_arr.round(0)
            df_campo["F80_mm"] = F80_arr.round(0)
            df_campo["W_kWh_t"] = W_arr.round(2)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("F₈₀ promedio", f"{F80_arr.mean():.0f} mm")
            col2.metric("F₈₀ std dev",  f"{F80_arr.std():.0f} mm")
            col3.metric("F₈₀ mínimo",   f"{F80_arr.min():.0f} mm")
            col4.metric("F₈₀ máximo",   f"{F80_arr.max():.0f} mm")

            fig_csv = go.Figure()
            fig_csv.add_trace(go.Histogram(x=F80_arr, nbinsx=15,
                marker_color=AMBER, marker_opacity=0.75, name="F₈₀"))
            fig_csv.add_vline(x=F80_arr.mean(), line_dash="dash", line_color=GREEN,
                              annotation_text="Promedio", annotation_font_color=GREEN)
            fig_csv.update_layout(**BASE, height=270,
                title="Distribución de F₈₀ — Datos de Campo",
                xaxis=dict(title="F₈₀ (mm)", gridcolor="#1e2022"),
                yaxis=dict(title="N° Taladros", gridcolor="#1e2022"))
            st.plotly_chart(fig_csv, use_container_width=True)

            # Descarga de resultados
            csv_result = df_campo.to_csv(index=False).encode()
            st.download_button("📤 Descargar resultados CSV", csv_result,
                               "resultados_campo.csv", "text/csv")
        else:
            st.error(f"Faltan columnas. Usa la plantilla descargable.")
    else:
        st.info("Sube un archivo CSV con datos de taladros de campo para análisis masivo.")


# ════════════════════════════════════════════════════════════════════
# TAB 10 — REPORTE Y EXPORTACIÓN
# ════════════════════════════════════════════════════════════════════
with tab10:
    st.markdown("#### Reporte de Resultados")
    ts = datetime.now().strftime("%d/%m/%Y %H:%M")

    # ── Tabla resumen ─────────────────────────────────────────────────────────
    resumen = pd.DataFrame({
        "Módulo": [
            "Geometría","Geometría","Geometría",
            "Lilly","Kuz-Ram","Kuz-Ram","Kuz-Ram",
            "Bond","Monte Carlo","Monte Carlo","Monte Carlo",
            "Economía","Economía","Optimizador","Vibraciones",
        ],
        "Parámetro": [
            "Diámetro d","Burden B","Espaciamiento S",
            "Factor A","X₅₀","F₈₀","n (uniformidad)",
            "W energía","P10","P50","P90",
            "C_vol P50","C_mol P50","d_óptimo","PPV",
        ],
        "Valor": [
            f"{d} mm", f"{B:.2f} m", f"{S:.2f} m",
            f"{A_val:.2f}", f"{X50mm:.0f} mm", f"{F80mm:.0f} mm", f"{nu:.3f}",
            f"{W_base:.3f} kWh/t",
            f"{P10:.4f} $/t", f"{P50:.4f} $/t", f"{P90:.4f} $/t",
            f"{float(np.percentile(mc['cv'],50)):.4f} $/t",
            f"{float(np.percentile(mc['cm'],50)):.4f} $/t",
            f"{int(opt_row['d'])} mm",
            f"{ppv_cur:.1f} mm/s",
        ],
    })
    st.dataframe(resumen, hide_index=True, use_container_width=True)

    # ── Exportar Excel ─────────────────────────────────────────────────────────
    st.markdown("---")
    col_xl, col_pdf = st.columns(2)

    with col_xl:
        st.markdown("##### 📊 Exportar Excel")
        if st.button("Generar Excel"):
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                resumen.to_excel(writer, sheet_name="Resumen", index=False)
                mc.round(4).to_excel(writer, sheet_name="Monte_Carlo", index=False)
                df_opt.round(4).to_excel(writer, sheet_name="Optimizador", index=False)
                pd.DataFrame({"x_mm": x_rr.round(1), "RR_%": P_rr.round(1),
                               "SW_%": np.interp(x_rr, x_sw, P_sw).round(1)}
                              ).to_excel(writer, sheet_name="Granulometria", index=False)
                pd.DataFrame({
                    "Parametro":["d","H","J","UCS","GSI","Wi","P80","Ce","Cn","Explosivo"],
                    "Valor":    [d,H,J,UCS,GSI,Wi,P80,Ce,Cn,expl_name],
                }).to_excel(writer, sheet_name="Parametros", index=False)
            buf.seek(0)
            st.download_button(
                "📥 Descargar .xlsx", buf,
                f"mine_to_mill_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    # ── Exportar PDF ──────────────────────────────────────────────────────────
    with col_pdf:
        st.markdown("##### 📄 Exportar PDF")
        if not HAS_FPDF:
            st.warning("Instalar: `pip install fpdf2`")
        else:
            if st.button("Generar PDF"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_margins(20, 20, 20)

                # Título
                pdf.set_font("Helvetica", "B", 14)
                pdf.set_fill_color(213, 232, 240)
                pdf.cell(0, 10, "REPORTE MINE-TO-MILL", align="C", fill=True, ln=True)
                pdf.set_font("Helvetica", "", 9)
                pdf.cell(0, 6, f"Generado: {ts}  |  Explosivo: {expl_name}  |  d = {d} mm", align="C", ln=True)
                pdf.ln(4)

                # Secciones
                sections = [
                    ("1. PARAMETROS DE ENTRADA", [
                        ("Diametro d",f"{d} mm"), ("Altura banco H",f"{H} m"),
                        ("Sobreperforacion J",f"{J} m"), ("UCS",f"{UCS} MPa"),
                        ("GSI",str(GSI)), ("Wi",f"{Wi} kWh/t"),
                        ("P80 objetivo",f"{P80} um"), ("Precio explosivo",f"{Ce:.2f} $/kg"),
                        ("Tarifa electrica",f"{Cn:.3f} $/kWh"),
                    ]),
                    ("2. DISENO DE MALLA", [
                        ("Burden B",f"{B:.2f} m"), ("Espaciamiento S",f"{S:.2f} m"),
                        ("Taco T",f"{T:.2f} m"), ("Factor Lilly A",f"{A_val:.2f}"),
                        ("Q explosivo/hole",f"{Q:.1f} kg"),
                        ("Factor potencia",f"{PF:.3f} kg/m3"),
                    ]),
                    ("3. FRAGMENTACION (KUZ-RAM)", [
                        ("X50 tamano medio",f"{X50mm:.0f} mm"),
                        ("F80 alimento planta",f"{F80mm:.0f} mm"),
                        ("n uniformidad",f"{nu:.3f}"),
                    ]),
                    ("4. ENERGIA DE MOLIENDA (BOND)", [
                        ("Energia especifica W",f"{W_base:.3f} kWh/t"),
                        ("Reduccion de tamano",f"{F80mm*1000/P80:.0f} x"),
                    ]),
                    ("5. ANALISIS DE RIESGO (MONTE CARLO)", [
                        ("P10 Optimista",f"{P10:.4f} $/t"),
                        ("P50 Esperado",f"{P50:.4f} $/t"),
                        ("P90 Pesimista",f"{P90:.4f} $/t"),
                        ("Indice incertidumbre",f"{(P90-P10)/P50*100:.1f} %"),
                    ]),
                    ("6. OPTIMIZADOR", [
                        ("Diametro optimo",f"{int(opt_row['d'])} mm"),
                        ("Costo minimo",f"{opt_row['C_total']:.4f} $/t"),
                    ]),
                    ("7. VIBRACIONES (PPV)", [
                        ("PPV en estructura",f"{ppv_cur:.1f} mm/s"),
                        ("Estado",ppv_status),
                    ]),
                ]

                for title, rows in sections:
                    pdf.set_font("Helvetica", "B", 10)
                    pdf.set_fill_color(213, 232, 240)
                    pdf.cell(0, 7, title, fill=True, ln=True)
                    pdf.set_font("Helvetica", "", 9)
                    for lbl, val in rows:
                        pdf.cell(100, 5.5, lbl)
                        pdf.cell(0, 5.5, val, ln=True)
                    pdf.ln(2)

                pdf_buf = io.BytesIO(bytes(pdf.output()))
                st.download_button(
                    "📥 Descargar .pdf", pdf_buf,
                    f"mine_to_mill_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    "application/pdf",
                )

    # ── Historial de sesión ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("##### 📝 Historial de Cálculos (sesión actual)")
    if "historial" not in st.session_state:
        st.session_state["historial"] = []

    if st.button("💾 Guardar diseño actual en historial"):
        st.session_state["historial"].append({
            "Hora":       datetime.now().strftime("%H:%M:%S"),
            "d (mm)":     d,
            "Explosivo":  expl_name,
            "F₈₀ (mm)":   f"{F80mm:.0f}",
            "W (kWh/t)":  f"{W_base:.2f}",
            "P50 ($/t)":  f"{P50:.4f}",
            "d_óptimo":   int(opt_row["d"]),
        })
        st.success("Guardado.")

    if st.session_state["historial"]:
        df_hist = pd.DataFrame(st.session_state["historial"])
        st.dataframe(df_hist, hide_index=True, use_container_width=True)
        hist_csv = df_hist.to_csv(index=False).encode()
        st.download_button("📥 Exportar historial CSV", hist_csv, "historial.csv", "text/csv")
        if st.button("🗑️ Limpiar historial"):
            st.session_state["historial"] = []
            st.rerun()
    else:
        st.info("Aún no hay diseños guardados en esta sesión.")
