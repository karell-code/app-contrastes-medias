# ============================================================
# APP WEB: CONTRASTES DE HIPÓTESIS PARA LA MEDIA
# Streamlit - lista para desplegar
# ============================================================
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(
    page_title="Contrastes de Hipótesis para la Media",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

COL_MAIN, COL_ALT = '#0984e3', '#e84393'
COL_NEG, COL_POS  = '#d63031', '#00b894'
BG, TXT, SUBTXT   = '#f7f7f9', '#2d3436', '#636e72'

plt.rcParams['figure.facecolor'] = BG
plt.rcParams['axes.facecolor']   = BG
plt.rcParams['font.family']      = 'DejaVu Sans'

st.markdown("""
<style>
    .main-title { color: #0984e3; font-size: 2.2rem; font-weight: 700; }
    .subtitle  { color: #636e72; font-size: 1rem; margin-bottom: 1.5rem; }
    .result-box { padding: 1.2rem; border-radius: 8px; margin: 1rem 0; font-weight: 600; }
    .reject    { background: #ffeaea; border-left: 6px solid #d63031; }
    .accept    { background: #e8f8f0; border-left: 6px solid #00b894; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Funciones de cálculo
# ============================================================
def parsear_datos(texto):
    if not texto or not texto.strip():
        return np.array([])
    texto = texto.replace(';', ',').replace('\n', ',').replace('\t', ',')
    partes = [p.strip() for p in texto.split(',') if p.strip()]
    valores = []
    for p in partes:
        try:
            valores.append(float(p))
        except ValueError:
            pass
    return np.array(valores)


def contraste_una_muestra(x, mu0, alpha, cola, sigma_conocida=None):
    n = len(x)
    media = np.mean(x)
    s = np.std(x, ddof=1)

    if sigma_conocida is not None and sigma_conocida > 0:
        se = sigma_conocida / np.sqrt(n)
        estad = (media - mu0) / se
        if cola == 'two-sided':
            p_val = 2 * (1 - stats.norm.cdf(abs(estad)))
        elif cola == 'greater':
            p_val = 1 - stats.norm.cdf(estad)
        else:
            p_val = stats.norm.cdf(estad)
        distribucion = 'z (normal)'
        gl = None
        crit = stats.norm.ppf(1 - alpha/2) if cola == 'two-sided' else stats.norm.ppf(1 - alpha)
    else:
        se = s / np.sqrt(n)
        estad = (media - mu0) / se
        gl = n - 1
        if cola == 'two-sided':
            p_val = 2 * (1 - stats.t.cdf(abs(estad), gl))
        elif cola == 'greater':
            p_val = 1 - stats.t.cdf(estad, gl)
        else:
            p_val = stats.t.cdf(estad, gl)
        distribucion = 't de Student'
        crit = stats.t.ppf(1 - alpha/2, gl) if cola == 'two-sided' else stats.t.ppf(1 - alpha, gl)

    if cola == 'two-sided':
        ic = (media - crit * se, media + crit * se)
    elif cola == 'greater':
        ic = (media - crit * se, np.inf)
    else:
        ic = (-np.inf, media + crit * se)

    return {
        'n': n, 'media': media, 's': s, 'se': se,
        'estadistico': estad, 'p_valor': p_val, 'gl': gl,
        'distribucion': distribucion, 'critico': crit, 'ic': ic,
        'mu0': mu0, 'alpha': alpha, 'cola': cola
    }


def contraste_dos_muestras(x, y, alpha, cola, varianzas_iguales=False):
    nx, ny = len(x), len(y)
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)

    if varianzas_iguales:
        sp2 = ((nx-1)*sx**2 + (ny-1)*sy**2) / (nx+ny-2)
        se = np.sqrt(sp2 * (1/nx + 1/ny))
        gl = nx + ny - 2
    else:
        se = np.sqrt(sx**2/nx + sy**2/ny)
        num = (sx**2/nx + sy**2/ny)**2
        den = (sx**2/nx)**2/(nx-1) + (sy**2/ny)**2/(ny-1)
        gl = num/den

    estad = (mx - my) / se
    if cola == 'two-sided':
        p_val = 2 * (1 - stats.t.cdf(abs(estad), gl))
    elif cola == 'greater':
        p_val = 1 - stats.t.cdf(estad, gl)
    else:
        p_val = stats.t.cdf(estad, gl)
    crit = stats.t.ppf(1 - alpha/2, gl) if cola == 'two-sided' else stats.t.ppf(1 - alpha, gl)

    if cola == 'two-sided':
        ic = (mx - my - crit*se, mx - my + crit*se)
    elif cola == 'greater':
        ic = (mx - my - crit*se, np.inf)
    else:
        ic = (-np.inf, mx - my + crit*se)

    return {
        'nx': nx, 'ny': ny, 'mx': mx, 'my': my, 'sx': sx, 'sy': sy,
        'se': se, 'estadistico': estad, 'p_valor': p_val, 'gl': gl,
        'critico': crit, 'ic': ic, 'alpha': alpha, 'cola': cola,
        'varianzas_iguales': varianzas_iguales
    }


def contraste_pareado(x, y, alpha, cola):
    d = np.array(x) - np.array(y)
    n = len(d)
    media_d = np.mean(d)
    s_d = np.std(d, ddof=1)
    se = s_d / np.sqrt(n)
    estad = media_d / se
    gl = n - 1
    if cola == 'two-sided':
        p_val = 2 * (1 - stats.t.cdf(abs(estad), gl))
    elif cola == 'greater':
        p_val = 1 - stats.t.cdf(estad, gl)
    else:
        p_val = stats.t.cdf(estad, gl)
    crit = stats.t.ppf(1 - alpha/2, gl) if cola == 'two-sided' else stats.t.ppf(1 - alpha, gl)
    if cola == 'two-sided':
        ic = (media_d - crit*se, media_d + crit*se)
    elif cola == 'greater':
        ic = (media_d - crit*se, np.inf)
    else:
        ic = (-np.inf, media_d + crit*se)
    return {
        'n': n, 'media_d': media_d, 's_d': s_d, 'se': se,
        'estadistico': estad, 'p_valor': p_val, 'gl': gl,
        'critico': crit, 'ic': ic, 'alpha': alpha, 'cola': cola
    }


# ============================================================
# Gráficos
# ============================================================
def grafico_contraste(res):
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=110)
    estad = res['estadistico']
    gl = res['gl']
    cola = res['cola']
    alpha = res['alpha']

    dist = stats.norm if gl is None else stats.t(gl)
    x = np.linspace(-4, 4, 800)
    y = dist.pdf(x)
    ax.plot(x, y, color=COL_MAIN, lw=2, label=f'Distribución {res["distribucion"]}')

    if cola == 'two-sided':
        c1 = dist.ppf(alpha/2); c2 = dist.ppf(1 - alpha/2)
        xa1 = x[x <= c1]; xa2 = x[x >= c2]
        ax.fill_between(xa1, dist.pdf(xa1), color=COL_NEG, alpha=0.35, label=f'Región crítica (α={alpha})')
        ax.fill_between(xa2, dist.pdf(xa2), color=COL_NEG, alpha=0.35)
        ax.axvline(c1, color=COL_NEG, ls='--', lw=1)
        ax.axvline(c2, color=COL_NEG, ls='--', lw=1)
    elif cola == 'greater':
        c = dist.ppf(1 - alpha)
        xa = x[x >= c]
        ax.fill_between(xa, dist.pdf(xa), color=COL_NEG, alpha=0.35, label=f'Región crítica (α={alpha})')
        ax.axvline(c, color=COL_NEG, ls='--', lw=1)
    else:
        c = dist.ppf(alpha)
        xa = x[x <= c]
        ax.fill_between(xa, dist.pdf(xa), color=COL_NEG, alpha=0.35, label=f'Región crítica (α={alpha})')
        ax.axvline(c, color=COL_NEG, ls='--', lw=1)

    ax.axvline(estad, color=COL_ALT, ls='-', lw=2.5, label=f'Estadístico observado = {estad:.3f}')
    ax.set_title('Distribución del estadístico y región crítica', fontsize=12, fontweight='bold')
    ax.set_xlabel('Valor del estadístico', fontsize=10, color=SUBTXT)
    ax.set_ylabel('Densidad', fontsize=10, color=SUBTXT)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.25)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    return fig


def grafico_datos(x, y=None, tipo='una_muestra', mu0=None):
    fig, ax = plt.subplots(figsize=(10, 3.8), dpi=110)
    if tipo == 'una_muestra':
        ax.hist(x, bins='auto', color=COL_MAIN, edgecolor='white', alpha=0.85)
        ax.axvline(np.mean(x), color=COL_ALT, lw=2, label=f'Media muestral = {np.mean(x):.3f}')
        if mu0 is not None:
            ax.axvline(mu0, color=COL_NEG, lw=2, ls='--', label=f'μ₀ = {mu0}')
    elif tipo == 'dos_muestras':
        ax.hist(x, bins='auto', color=COL_MAIN, edgecolor='white', alpha=0.7, label='Muestra 1')
        ax.hist(y, bins='auto', color=COL_ALT, edgecolor='white', alpha=0.7, label='Muestra 2')
        ax.axvline(np.mean(x), color=COL_MAIN, lw=2, ls='--')
        ax.axvline(np.mean(y), color=COL_ALT, lw=2, ls='--')
    else:
        d = np.array(x) - np.array(y)
        ax.hist(d, bins='auto', color=COL_MAIN, edgecolor='white', alpha=0.85)
        ax.axvline(np.mean(d), color=COL_ALT, lw=2, label=f'Media de diferencias = {np.mean(d):.3f}')
        ax.axvline(0, color=COL_NEG, lw=2, ls='--', label='Diferencia nula (0)')
    ax.set_xlabel('Valor', fontsize=10, color=SUBTXT)
    ax.set_ylabel('Frecuencia', fontsize=10, color=SUBTXT)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.25)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    return fig


# ============================================================
# Interfaz
# ============================================================
st.markdown('<p class="main-title">🧪 Contrastes de Hipótesis para la Media</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">App interactiva para realizar pruebas de hipótesis '
            'sobre medias poblacionales.</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Configuración")

    tipo = st.radio("Tipo de contraste:",
                    ["Una muestra", "Dos muestras independientes", "Muestras pareadas"],
                    index=0)

    alpha = st.select_slider("Nivel de significación (α):",
                             options=[0.001, 0.01, 0.05, 0.10],
                             value=0.05, format_func=lambda x: f"{x:.3f}")

    cola_opciones = {
        "Bilateral (≠)": "two-sided",
        "Cola derecha (>)": "greater",
        "Cola izquierda (<)": "less",
    }
    cola_label = st.radio("Hipótesis alternativa:", list(cola_opciones.keys()))
    cola = cola_opciones[cola_label]

    if tipo == "Una muestra":
        mu0 = st.number_input("Media poblacional bajo H₀ (μ₀):", value=0.0, step=0.1)
        usar_sigma = st.checkbox("Usar σ conocida (test z)")
        sigma = st.number_input("σ:", value=1.0, step=0.1, min_value=0.001) if usar_sigma else None
        var_iguales = False
    elif tipo == "Dos muestras independientes":
        mu0 = 0.0; sigma = None
        var_iguales = st.checkbox("Asumir varianzas iguales")
    else:
        mu0 = 0.0; sigma = None; var_iguales = False

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Muestra 1")
    datos_1_txt = st.text_area("Pega los datos (separados por comas, espacios o saltos de línea):",
                                height=150, key="datos1")
    archivo_1 = st.file_uploader("O sube un archivo .csv / .txt", type=['csv', 'txt'], key="f1")
    if archivo_1 is not None:
        try:
            datos_1_txt = archivo_1.read().decode('utf-8')
            st.success(f"Archivo cargado: {archivo_1.name}")
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")

with col2:
    if tipo != "Una muestra":
        st.subheader("📊 Muestra 2")
        datos_2_txt = st.text_area("Pega los datos (separados por comas, espacios o saltos de línea):",
                                    height=150, key="datos2")
        archivo_2 = st.file_uploader("O sube un archivo .csv / .txt", type=['csv', 'txt'], key="f2")
        if archivo_2 is not None:
            try:
                datos_2_txt = archivo_2.read().decode('utf-8')
                st.success(f"Archivo cargado: {archivo_2.name}")
            except Exception as e:
                st.error(f"Error al leer el archivo: {e}")
    else:
        datos_2_txt = ""

st.markdown("---")
if st.button("▶ Ejecutar contraste", type="primary", use_container_width=True):

    x = parsear_datos(datos_1_txt)
    y = parsear_datos(datos_2_txt)

    if len(x) == 0:
        st.error("⚠ Introduce los datos de la muestra 1.")
        st.stop()
    if tipo != "Una muestra" and len(y) == 0:
        st.error("⚠ Introduce los datos de la muestra 2.")
        st.stop()
    if tipo == "Muestras pareadas" and len(x) != len(y):
        st.error(f"⚠ En muestras pareadas las longitudes deben coincidir. M1={len(x)}, M2={len(y)}")
        st.stop()

    if tipo == "Una muestra":
        res = contraste_una_muestra(x, mu0, alpha, cola, sigma)
        encabezado = "Contraste de una muestra"
        hipotesis = f"H₀: μ = {mu0}  vs  H₁: μ {'≠' if cola=='two-sided' else '>' if cola=='greater' else '<'} {mu0}"
    elif tipo == "Dos muestras independientes":
        res = contraste_dos_muestras(x, y, alpha, cola, var_iguales)
        encabezado = "Contraste de dos muestras independientes"
        hipotesis = f"H₀: μ₁ - μ₂ = 0  vs  H₁: μ₁ - μ₂ {'≠' if cola=='two-sided' else '>' if cola=='greater' else '<'} 0"
    else:
        res = contraste_pareado(x, y, alpha, cola)
        encabezado = "Contraste de muestras pareadas"
        hipotesis = f"H₀: μ_d = 0  vs  H₁: μ_d {'≠' if cola=='two-sided' else '>' if cola=='greater' else '<'} 0"

    rechaza = res['p_valor'] < alpha

    st.markdown(f"### {encabezado}")
    st.markdown(f"`{hipotesis}`")
    st.markdown(f"Nivel de significación: **α = {alpha}**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("n", res.get('n', res.get('nx', '-')))
    c2.metric("Media", f"{res.get('media', res.get('mx', res.get('media_d', 0))):.4f}")
    c3.metric("Estadístico", f"{res['estadistico']:.4f}")
    c4.metric("p-valor", f"{res['p_valor']:.6f}")

    with st.expander("📋 Ver tabla detallada de resultados", expanded=True):
        tabla = {
            'Estadístico': ['n', 'Media muestral', 'Desviación estándar', 'Error estándar',
                            f'Estadístico ({res["distribucion"]})', 'Grados de libertad',
                            'Valor crítico', 'p-valor', 'Intervalo de confianza'],
            'Valor': [
                str(res.get('n', res.get('nx', '-'))),
                f"{res.get('media', res.get('mx', res.get('media_d', 0))):.4f}",
                f"{res.get('s', res.get('sx', res.get('s_d', 0))):.4f}",
                f"{res['se']:.4f}",
                f"{res['estadistico']:.4f}",
                str(res['gl']) if res['gl'] is not None else '—',
                f"±{res['critico']:.4f}",
                f"{res['p_valor']:.6f}",
                f"({res['ic'][0]:.4f}, {res['ic'][1]:.4f})",
            ]
        }
        st.dataframe(pd.DataFrame(tabla), use_container_width=True, hide_index=True)

    if rechaza:
        st.markdown(f"""
        <div class="result-box reject">
            ✅ Se <b>RECHAZA</b> H₀ con α = {alpha} (p = {res['p_valor']:.6f}).
            Existe evidencia estadísticamente significativa.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box accept">
            ❌ <b>NO</b> se rechaza H₀ con α = {alpha} (p = {res['p_valor']:.6f}).
            No hay evidencia suficiente para rechazar la hipótesis nula.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📈 Visualizaciones")

    if tipo == "Una muestra":
        fig1 = grafico_datos(x, tipo='una_muestra', mu0=mu0)
    elif tipo == "Dos muestras independientes":
        fig1 = grafico_datos(x, y=y, tipo='dos_muestras')
    else:
        fig1 = grafico_datos(x, y=y, tipo='pareado')
    st.pyplot(fig1)

    fig2 = grafico_contraste(res)
    st.pyplot(fig2)

st.markdown("---")
st.caption("App desarrollada con Streamlit · Universidad de Ciencias Médicas de Las Tunas. Dr. Karell Rafael Vazquez Argote. Profesor Asistente")