# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Detección de ciberataques mediante aprendizaje automático sobre CICIDS2017
# **Notebook del experimento del TFM** · Pablo Claudio Baragaño · Máster Universitario en Inteligencia Artificial (UAX)
#
# Este cuaderno implementa, fase a fase, el diseño experimental del **Capítulo 4** (Ilustración 7) y genera todas las
# tablas y figuras del **Capítulo 5**:
#
# | Fase | Apartado del Cap. 4 | Qué hace |
# |---|---|---|
# | 1. Obtención y limpieza | 4.4.1 | Carga los 5 CSV, normaliza nombres, quita la columna duplicada, infinitos/nulos, duplicados y negativos |
# | 2. Partición | 4.4.2 | 70 / 15 / 15 estratificado |
# | 3. Preprocesamiento | 4.4.3 | Etiquetas, características constantes y correlacionadas (>0,95), estandarización |
# | 4. Balanceo | 4.4.4 | SMOTE (k = 5) con un mínimo de 20.000 registros por categoría (solo multiclase) |
# | 5. Entrenamiento | 4.4.5 | Búsqueda aleatoria + validación cruzada para RF, XGBoost y SVM; búsqueda sobre validación para la DNN |
# | 6. Evaluación | 4.5 | Métricas, matrices de confusión, curvas ROC, prueba de McNemar y selección del mejor modelo |
# | 7. Interpretabilidad | 4.4.6 | SHAP global y local |
# | 8. Prototipo | 4.4.7 | Esquema de la arquitectura y simulación por lotes de 1.000 flujos |
#
# ## Antes de ejecutar
# 1. **Datos.** Descarga CICIDS2017 en su versión de flujos *MachineLearningCVE* (archivo `MachineLearningCSV.zip`,
#    desde la página oficial del Canadian Institute for Cybersecurity: https://www.unb.ca/cic/datasets/ids-2017.html)
#    y copia estos cinco CSV en la carpeta `datos/`, junto a este notebook (pueden estar en subcarpetas):
#    - `Monday-WorkingHours.pcap_ISCX.csv`
#    - `Wednesday-workingHours.pcap_ISCX.csv`
#    - `Friday-WorkingHours-Morning.pcap_ISCX.csv`
#    - `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
#    - `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
# 2. **Entorno.** Lo más sencillo es hacer doble clic en `ejecutar_experimento.bat`, que crea el entorno, hace la
#    prueba y lanza la ejecución real. Si lo haces a mano, usa Python 3.12 y crea el entorno en una ruta sin tildes
#    (TensorFlow puede fallar si la ruta tiene caracteres como «á» o «º»):
#    ```
#    py -3.12 -m venv %USERPROFILE%\tfm_cicids2017_venv
#    %USERPROFILE%\tfm_cicids2017_venv\Scripts\activate
#    pip install -r requirements.txt
#    ```
#    En Google Colab basta con ejecutar la celda de instalación que aparece más abajo.
# 3. **Primero en modo prueba.** Deja `MODO_PRUEBA = True` y ejecuta todo (*Run All*). Tarda unos minutos y usa solo
#    el 2 % de los datos: sirve para comprobar que el entorno funciona. Sus resultados se guardan aparte, en
#    `resultados_prueba/`, y **no deben usarse en el TFM**.
# 4. **Después, la ejecución real.** Cambia a `MODO_PRUEBA = False` y vuelve a ejecutar todo. Los resultados quedan en
#    `resultados/`: tablas en CSV y en un único Excel, figuras en PNG a 300 ppp, los modelos entrenados y un resumen en
#    JSON. Si la ejecución se corta, al volver a lanzarla se reutiliza todo lo que ya estaba calculado.

# %% tags=["parameters"]
# ======================= PARÁMETROS PRINCIPALES =======================
MODO_PRUEBA = True     # True: ejecución rápida de comprobación (2 % de los datos). False: experimento completo.
DIR_BASE = None        # None = la carpeta donde está este notebook (o /content/drive/MyDrive/TFM en Colab)
FORZAR = False         # True = recalcular todo aunque ya existan resultados guardados
AJUSTES_EXTRA = {}     # Para cambiar cualquier parámetro de la configuración sin tocar el código

# Al lanzar el experimento con ejecutar_experimento.bat (sin abrir el notebook), el modo se elige así:
import os as _os
if _os.environ.get('TFM_MODO_PRUEBA') in ('0', '1'):
    MODO_PRUEBA = _os.environ['TFM_MODO_PRUEBA'] == '1'
if _os.environ.get('TFM_DIR_BASE'):
    DIR_BASE = _os.environ['TFM_DIR_BASE']

# %%
# Solo en Google Colab: instala las librerías que no vienen de serie y monta Google Drive.
import sys
EN_COLAB = 'google.colab' in sys.modules
if EN_COLAB:
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'imbalanced-learn', 'shap', 'xgboost'], check=True)
    from google.colab import drive
    drive.mount('/content/drive')

# %%
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import json, time, math, platform, warnings, itertools, re
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve,
                             f1_score, accuracy_score)
from xgboost import XGBClassifier
from imblearn import FunctionSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from statsmodels.stats.contingency_tables import mcnemar
import shap

import tensorflow as tf
import keras

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
pd.set_option('display.max_columns', 60)
pd.set_option('display.width', 200)

try:
    from IPython.display import display
except ImportError:          # ejecución fuera de Jupyter
    display = print

T_INICIO_GLOBAL = time.perf_counter()

# %%
# ======================= CONFIGURACIÓN (valores del Capítulo 4) =======================
if DIR_BASE is None:
    BASE = Path('/content/drive/MyDrive/TFM') if EN_COLAB else Path.cwd()
else:
    BASE = Path(DIR_BASE)

CFG = dict(
    SEMILLA=42,
    PARTICION=(0.70, 0.15, 0.15),        # entrenamiento, validación, prueba (4.4.2)
    UMBRAL_CORRELACION=0.95,             # 4.4.3
    SMOTE_MINIMO=20000,                  # registros mínimos por categoría en entrenamiento (4.4.4)
    SMOTE_K=5,
    N_ITER=20,                           # combinaciones por algoritmo y escenario (4.4.5)
    CV_PARTICIONES=3,
    FRACCION_BUSQUEDA=0.20,              # submuestra estratificada para la búsqueda
    SVM_N=50000,                         # submuestra para la SVM (búsqueda y entrenamiento final)
    DNN_FRACCION_BUSQUEDA=0.20,          # submuestra de entrenamiento para comparar configuraciones de la DNN
    DNN_EPOCAS_MAX=50,
    DNN_PACIENCIA=5,
    SHAP_N=5000,                         # flujos del conjunto de prueba explicados con SHAP (4.4.6)
    SHAP_MAX_MINUTOS=30,                 # si SHAP fuese a tardar más, se reduce la muestra (se informa)
    LOTE_PROTOTIPO=1000,                 # 4.4.7
    ALFA=0.05,                           # nivel de significación (4.5)
    N_JOBS=-1,
    FRACCION_MODO_PRUEBA=0.02,
)
if MODO_PRUEBA:
    CFG.update(N_ITER=2, CV_PARTICIONES=2, SVM_N=3000, DNN_EPOCAS_MAX=3, SHAP_N=300, SHAP_MAX_MINUTOS=5,
               SMOTE_MINIMO=400)
CFG.update(AJUSTES_EXTRA)
SEMILLA = CFG['SEMILLA']

DIR_DATOS = BASE / 'datos'
DIR_RES = BASE / ('resultados_prueba' if MODO_PRUEBA else 'resultados')
DIR_TABLAS, DIR_FIG, DIR_MOD, DIR_CACHE = (DIR_RES / 'tablas', DIR_RES / 'figuras', DIR_RES / 'modelos', DIR_RES / 'cache')
for d in (DIR_TABLAS, DIR_FIG, DIR_MOD, DIR_CACHE):
    d.mkdir(parents=True, exist_ok=True)

np.random.seed(SEMILLA)
keras.utils.set_random_seed(SEMILLA)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print('Carpeta base     :', BASE)
print('Datos            :', DIR_DATOS)
print('Resultados       :', DIR_RES)
print('MODO_PRUEBA      :', MODO_PRUEBA)
if MODO_PRUEBA:
    print('\n*** MODO PRUEBA: resultados solo para comprobar que todo funciona. NO usar en el TFM. ***')

# %%
# ======================= ESTILO DE LAS FIGURAS =======================
# Colores fijos por modelo y por escenario (paleta categórica validada para daltonismo).
COLOR_MODELO = OrderedDict([('Random Forest', '#2a78d6'), ('XGBoost', '#eb6834'), ('SVM', '#1baf7a'), ('DNN', '#eda100')])
COLOR_ESCENARIO = OrderedDict([('E1', '#2a78d6'), ('E2', '#eb6834'), ('E3', '#1baf7a')])
AZUL, AZUL_CLARO = '#1c5cab', '#86b6ef'
TINTA, TINTA_2, REJILLA, EJE = '#0b0b0b', '#52514e', '#e1e0d9', '#c3c2b7'
CMAP_SECUENCIAL = LinearSegmentedColormap.from_list(
    'azules', ['#fcfcfb', '#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#104281'])

plt.rcParams.update({
    'font.family': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'axes.edgecolor': EJE, 'axes.labelcolor': TINTA, 'axes.titlecolor': TINTA,
    'xtick.color': TINTA_2, 'ytick.color': TINTA_2, 'axes.grid': False,
    'grid.color': REJILLA, 'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'figure.dpi': 110, 'savefig.dpi': 300,
})

def coma(x, dec=2):
    """Número con coma decimal (formato español)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '–'
    return f'{x:,.{dec}f}'.replace(',', 'X').replace('.', ',').replace('X', '.')

def formato_es(ax, ejes='xy'):
    """Números de los ejes con coma decimal."""
    f = matplotlib.ticker.FuncFormatter(lambda v, p: f'{v:g}'.replace('.', ','))
    if 'x' in ejes and ax.get_xscale() == 'linear':
        ax.xaxis.set_major_formatter(f)
    if 'y' in ejes and ax.get_yscale() == 'linear':
        ax.yaxis.set_major_formatter(f)

TRADUCCIONES_SHAP = [
    (r'Sum of (\d+) other features', r'Suma de otras \1 características'),
    (r'(\d+) other features', r'\1 características más'),
    (r'mean\(\|SHAP value\|\)', 'media(|valor SHAP|)'),
    (r'SHAP value \(impact on model output\)', 'Valor SHAP (impacto en la salida del modelo)'),
    (r'^SHAP value$', 'Valor SHAP'),
    (r'Feature value', 'Valor de la característica'),
    (r'^High$', 'Alto'), (r'^Low$', 'Bajo'),
    (r'(\d)\.(\d)', r'\1,\2'),
]
def _es(txt):
    for patron, sust in TRADUCCIONES_SHAP:
        txt = re.sub(patron, sust, txt)
    return txt

def traducir_shap(fig):
    """Traduce al español los textos que añade la librería SHAP y usa coma decimal."""
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_text():
            t.set_text(_es(t.get_text()))
    for ax in fig.axes:
        for eje in (ax.xaxis, ax.yaxis):
            f = eje.get_major_formatter()
            if isinstance(f, matplotlib.ticker.FixedFormatter):
                f.seq = [_es(x) for x in f.seq]
            elif isinstance(f, matplotlib.ticker.FuncFormatter):      # etiquetas de texto (nombres, "High", "Low"...)
                eje.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p, _f=f: _es(str(_f(v, p)))))
            elif isinstance(f, matplotlib.ticker.ScalarFormatter):
                eje.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f'{v:g}'.replace('.', ',')))

TABLAS = OrderedDict()
FIGURAS = OrderedDict()
RESUMEN = OrderedDict(modo_prueba=MODO_PRUEBA)

def guardar_tabla(df, nombre, descripcion):
    df = df.copy()
    df.to_csv(DIR_TABLAS / f'{nombre}.csv', index=False, encoding='utf-8')
    TABLAS[nombre] = (descripcion, df)
    print(f'[tabla] {nombre}.csv  ->  {descripcion}')
    display(df)

def guardar_figura(fig, nombre, descripcion):
    if MODO_PRUEBA:
        fig.text(0.5, 0.5, 'MODO PRUEBA', fontsize=40, color='red', alpha=0.15,
                 ha='center', va='center', rotation=25, transform=fig.transFigure)
    ruta = DIR_FIG / f'{nombre}.png'
    fig.savefig(ruta, bbox_inches='tight', facecolor='white')
    FIGURAS[nombre] = descripcion
    print(f'[figura] {nombre}.png  ->  {descripcion}')
    plt.show()
    plt.close(fig)

def cronometro(etiqueta, t0):
    print(f'   ({etiqueta}: {time.perf_counter() - t0:,.1f} s)')

# %% [markdown]
# ## Fase 1 · Obtención y limpieza de los datos (apartado 4.4.1)
# Se cargan los cinco archivos del lunes, el miércoles y el viernes. Las columnas numéricas se convierten a `float32`
# para reducir a la mitad la memoria. La primera tabla reproduce la Tabla 3 del Capítulo 4 y la compara con los valores
# que figuran en ella, para detectar cualquier diferencia con los archivos descargados.

# %%
ARCHIVOS = OrderedDict([
    ('Monday-WorkingHours.pcap_ISCX.csv', 'Lunes'),
    ('Wednesday-workingHours.pcap_ISCX.csv', 'Miércoles'),
    ('Friday-WorkingHours-Morning.pcap_ISCX.csv', 'Viernes (mañana)'),
    ('Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv', 'Viernes (tarde, PortScan)'),
    ('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Viernes (tarde, DDoS)'),
])
# Valores de la Tabla 3 del Capítulo 4 (para comprobarlos con los archivos reales)
ESPERADO_T1 = {
    ('Monday-WorkingHours.pcap_ISCX.csv', 'BENIGN'): 529918,
    ('Wednesday-workingHours.pcap_ISCX.csv', 'BENIGN'): 440031,
    ('Wednesday-workingHours.pcap_ISCX.csv', 'DoS Hulk'): 231073,
    ('Wednesday-workingHours.pcap_ISCX.csv', 'DoS GoldenEye'): 10293,
    ('Wednesday-workingHours.pcap_ISCX.csv', 'DoS slowloris'): 5796,
    ('Wednesday-workingHours.pcap_ISCX.csv', 'DoS Slowhttptest'): 5499,
    ('Wednesday-workingHours.pcap_ISCX.csv', 'Heartbleed'): 11,
    ('Friday-WorkingHours-Morning.pcap_ISCX.csv', 'BENIGN'): 189067,
    ('Friday-WorkingHours-Morning.pcap_ISCX.csv', 'Bot'): 1966,
    ('Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv', 'BENIGN'): 127537,
    ('Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv', 'PortScan'): 158930,
    ('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'BENIGN'): 97718,
    ('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'DDoS'): 128027,
}

def localizar_archivos():
    encontrados = {}
    if DIR_DATOS.exists():
        for ruta in DIR_DATOS.rglob('*.csv'):
            for esperado in ARCHIVOS:
                if ruta.name.lower() == esperado.lower():
                    encontrados[esperado] = ruta
    faltan = [a for a in ARCHIVOS if a not in encontrados]
    if faltan:
        raise FileNotFoundError(
            'No se encuentran estos archivos en ' + str(DIR_DATOS) + ':\n  - ' + '\n  - '.join(faltan) +
            '\nDescarga MachineLearningCSV.zip (CICIDS2017) y copia los CSV en la carpeta datos/.')
    return encontrados

def leer_csv(ruta, archivo):
    df = pd.read_csv(ruta, encoding='latin-1', low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]          # paso 1: normalización de nombres
    if 'Label' not in df.columns:
        raise ValueError(f'{archivo}: no se encuentra la columna Label')
    df['Label'] = df['Label'].astype(str).str.strip()
    cols = [c for c in df.columns if c != 'Label']
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
    df['Archivo'] = archivo
    return df

t0 = time.perf_counter()
cache_bruto = DIR_RES.parent / 'cache_datos_brutos.parquet'      # compartido por modo prueba y modo real
if cache_bruto.exists() and not FORZAR:
    datos = pd.read_parquet(cache_bruto)
    print('Datos brutos leídos de la caché:', cache_bruto.name)
else:
    rutas = localizar_archivos()
    partes = []
    for archivo in ARCHIVOS:
        print('Leyendo', archivo, '...')
        partes.append(leer_csv(rutas[archivo], archivo))
    datos = pd.concat(partes, ignore_index=True)
    del partes
    datos.to_parquet(cache_bruto, index=False)
datos['Label'] = datos['Label'].astype(str)
cronometro('carga', t0)
print('Dimensiones:', datos.shape)

# Tabla de composición original (equivale a la Tabla 3 del Capítulo 4)
t01 = (datos.groupby(['Archivo', 'Label'], sort=False).size().rename('Registros').reset_index())
t01['Jornada'] = t01['Archivo'].map(ARCHIVOS)
t01['Tabla 3 (Cap. 4)'] = [ESPERADO_T1.get((a, l), np.nan) for a, l in zip(t01['Archivo'], t01['Label'])]
t01['¿Coincide?'] = np.where(t01['Registros'] == t01['Tabla 3 (Cap. 4)'], 'Sí', 'NO')
t01 = t01[['Jornada', 'Archivo', 'Label', 'Registros', 'Tabla 3 (Cap. 4)', '¿Coincide?']]
guardar_tabla(t01, 'T01_composicion_original', 'Registros por archivo y etiqueta antes de la limpieza (comprobación de la Tabla 3)')
if (t01['¿Coincide?'] == 'NO').any():
    print('\n>>> ATENCIÓN: hay diferencias con la Tabla 3 del Capítulo 4. Actualiza la tabla con los valores de T01.')
else:
    print('\nLos recuentos coinciden con la Tabla 3 del Capítulo 4.')
RESUMEN['registros_originales'] = int(len(datos))

# %%
# --- Limpieza en cinco pasos ---
t0 = time.perf_counter()
CARACT = [c for c in datos.columns if c not in ('Label', 'Archivo')]
pasos = [('Registros de partida', 0, len(datos))]
eliminados_por_paso = {}

# Paso 1: normalización de los nombres de las columnas (ya hecha al leer)
pasos.append(('1. Normalización de los nombres de las columnas', 0, len(datos)))

# Paso 2: columna Fwd Header Length duplicada (pandas renombra la segunda como "Fwd Header Length.1")
duplicadas = []
for c in list(CARACT):
    m = re.match(r'^(.*)\.(\d+)$', c)
    if m and m.group(1) in CARACT:
        iguales = np.array_equal(datos[c].to_numpy(), datos[m.group(1)].to_numpy(), equal_nan=True)
        print(f'Columna duplicada {c!r} -> idéntica a {m.group(1)!r}: {iguales}')
        if iguales:
            duplicadas.append(c)
datos = datos.drop(columns=duplicadas)
CARACT = [c for c in CARACT if c not in duplicadas]
pasos.append(('2. Eliminación de la columna duplicada (' + ', '.join(duplicadas) + ')' if duplicadas
              else '2. Eliminación de la columna duplicada (no encontrada)', 0, len(datos)))
RESUMEN['columnas_duplicadas_eliminadas'] = duplicadas
RESUMEN['n_caracteristicas_tras_limpieza'] = len(CARACT)

# Paso 3: infinitos y nulos
X_tmp = datos[CARACT].to_numpy()
mask_inf = np.isinf(X_tmp).any(axis=1)
mask_nan = np.isnan(X_tmp).any(axis=1)
mask_paso3 = mask_inf | mask_nan
inf_por_col = pd.Series(np.isinf(X_tmp).sum(axis=0), index=CARACT)
nan_por_col = pd.Series(np.isnan(X_tmp).sum(axis=0), index=CARACT)
del X_tmp
print('Columnas con infinitos:', inf_por_col[inf_por_col > 0].to_dict())
print('Columnas con nulos    :', nan_por_col[nan_por_col > 0].to_dict())
eliminados_por_paso['3. Valores infinitos o nulos'] = datos.loc[mask_paso3, 'Label'].value_counts()
datos = datos.loc[~mask_paso3].reset_index(drop=True)
pasos.append(('3. Valores infinitos o nulos', int(mask_paso3.sum()), len(datos)))

# Paso 4: registros duplicados (mismas características y misma etiqueta)
mask_dup = datos.duplicated(subset=CARACT + ['Label'], keep='first').to_numpy()
eliminados_por_paso['4. Registros duplicados'] = datos.loc[mask_dup, 'Label'].value_counts()
datos = datos.loc[~mask_dup].reset_index(drop=True)
pasos.append(('4. Registros duplicados', int(mask_dup.sum()), len(datos)))
# Información adicional: flujos con las mismas características pero distinta etiqueta (no se eliminan)
conflictos = datos.duplicated(subset=CARACT, keep=False)
RESUMEN['flujos_con_etiqueta_contradictoria'] = int(conflictos.sum())
print('Flujos idénticos con etiquetas distintas (se mantienen, solo informativo):', int(conflictos.sum()))

# Paso 5: valores negativos en características que no pueden ser negativas
PERMITIDAS_NEGATIVAS = ['Init_Win_bytes_forward', 'Init_Win_bytes_backward']
cols_neg = [c for c in CARACT if c not in PERMITIDAS_NEGATIVAS]
negativos = (datos[cols_neg] < 0)
neg_por_col = negativos.sum()
mask_neg = negativos.any(axis=1).to_numpy()
print('Columnas con negativos:', neg_por_col[neg_por_col > 0].to_dict())
eliminados_por_paso['5. Valores negativos no válidos'] = datos.loc[mask_neg, 'Label'].value_counts()
datos = datos.loc[~mask_neg].reset_index(drop=True)
pasos.append(('5. Valores negativos no válidos', int(mask_neg.sum()), len(datos)))
del negativos

t02 = pd.DataFrame(pasos, columns=['Paso', 'Registros eliminados', 'Registros restantes'])
t02['% eliminado'] = (t02['Registros eliminados'] / pasos[0][2] * 100).round(2)
guardar_tabla(t02, 'T02_limpieza', 'Registros eliminados en cada paso de la limpieza')

t02b = pd.DataFrame(eliminados_por_paso).fillna(0).astype(int)
t02b.index.name = 'Etiqueta'
t02b = t02b.reset_index()
guardar_tabla(t02b, 'T02b_limpieza_por_etiqueta', 'Registros eliminados en cada paso, por etiqueta original')

t02c = pd.DataFrame({'Característica': CARACT, 'Infinitos': inf_por_col.reindex(CARACT).to_numpy(),
                     'Nulos': nan_por_col.reindex(CARACT).to_numpy()})
t02c['Negativos'] = [int(neg_por_col.get(c, 0)) for c in CARACT]
t02c = t02c[(t02c[['Infinitos', 'Nulos', 'Negativos']].sum(axis=1)) > 0]
guardar_tabla(t02c, 'T02c_valores_problematicos', 'Características con valores infinitos, nulos o negativos')
RESUMEN['registros_tras_limpieza'] = int(len(datos))
cronometro('limpieza', t0)

# %% [markdown]
# ### Etiquetas: tarea binaria y multiclase (apartado 4.1)
# Binaria: BENIGN = 0 y el resto = 1. Multiclase: Benigno, DoS (Hulk, GoldenEye, slowloris, Slowhttptest), DDoS,
# PortScan y Bot. Heartbleed solo participa en la tarea binaria.

# %%
MAPA_CATEGORIA = {
    'BENIGN': 'Benigno', 'DoS Hulk': 'DoS', 'DoS GoldenEye': 'DoS', 'DoS slowloris': 'DoS',
    'DoS Slowhttptest': 'DoS', 'DDoS': 'DDoS', 'PortScan': 'PortScan', 'Bot': 'Bot', 'Heartbleed': 'Heartbleed',
}
CLASES_MULTI = ['Benigno', 'DoS', 'DDoS', 'PortScan', 'Bot']
CATEGORIAS = CLASES_MULTI + ['Heartbleed']
desconocidas = sorted(set(datos['Label']) - set(MAPA_CATEGORIA))
if desconocidas:
    raise ValueError(f'Etiquetas no previstas: {desconocidas}')
datos['Categoria'] = datos['Label'].map(MAPA_CATEGORIA)

if MODO_PRUEBA:
    # Submuestra estratificada (con un mínimo por clase) solo para comprobar el funcionamiento
    rng = np.random.default_rng(SEMILLA)
    idx_sel = []
    for cat, idx in datos.groupby('Categoria').groups.items():
        idx = np.asarray(idx)
        n = max(int(round(len(idx) * CFG['FRACCION_MODO_PRUEBA'])), min(len(idx), 60))
        idx_sel.append(rng.choice(idx, size=min(n, len(idx)), replace=False))
    datos = datos.loc[np.sort(np.concatenate(idx_sel))].reset_index(drop=True)
    print('MODO PRUEBA: submuestra de', len(datos), 'registros')

t03 = (datos.groupby(['Categoria', 'Label']).size().rename('Registros').reset_index())
t03['Categoria'] = pd.Categorical(t03['Categoria'], CATEGORIAS, ordered=True)
t03 = t03.sort_values(['Categoria', 'Registros'], ascending=[True, False])
t03['% del total'] = (t03['Registros'] / len(datos) * 100).round(3)
guardar_tabla(t03, 'T03_distribucion_tras_limpieza', 'Registros por etiqueta y categoría tras la limpieza')

# Figura: distribución por categoría antes y después de la limpieza
antes = (pd.read_parquet(cache_bruto, columns=['Label'])['Label'].astype(str).map(MAPA_CATEGORIA)
         .value_counts().reindex(CATEGORIAS).fillna(0))
despues = datos['Categoria'].value_counts().reindex(CATEGORIAS).fillna(0)
fig, ax = plt.subplots(figsize=(6.3, 3.0))
y = np.arange(len(CATEGORIAS))
h = 0.38
ax.barh(y - h / 2, antes.values, height=h, color=AZUL_CLARO, label='Antes de la limpieza')
ax.barh(y + h / 2, despues.values, height=h, color=AZUL, label='Después de la limpieza')
for yi, v in zip(y + h / 2, despues.values):
    ax.text(v * 1.15, yi, f'{int(v):,}'.replace(',', '.'), va='center', fontsize=7.5, color=TINTA_2)
ax.set_xscale('log')
ax.set_yticks(y, CATEGORIAS)
ax.invert_yaxis()
ax.set_xlabel('Número de registros (escala logarítmica)')
ax.grid(axis='x', which='major')
ax.set_axisbelow(True)
ax.legend(loc='lower right', fontsize=8)
ax.set_xlim(right=max(antes.max(), despues.max()) * 8)
guardar_figura(fig, 'F01_distribucion_clases', 'Distribución de registros por categoría antes y después de la limpieza')
RESUMEN['distribucion_categoria'] = {k: int(v) for k, v in despues.items()}

# %% [markdown]
# ## Fase 2 · Partición estratificada 70 / 15 / 15 (apartado 4.4.2)

# %%
def clave_estrato(categorias, minimo=8):
    """Categorías para estratificar; las que tienen muy pocos registros se agrupan para que la división sea posible."""
    s = pd.Series(np.asarray(categorias, dtype=object)).astype(str)
    cuenta = s.value_counts()
    pequenas = cuenta[cuenta < minimo].index
    if len(pequenas):
        s = s.where(~s.isin(pequenas), '__otros__')
        if (s == '__otros__').sum() < minimo:          # si aun así son muy pocos, se unen a la clase más numerosa
            s = s.where(s != '__otros__', cuenta.index[0])
    return s.to_numpy()

p_ent, p_val, p_pru = CFG['PARTICION']
indices = np.arange(len(datos))
estrato = clave_estrato(datos['Categoria'])
idx_ent, idx_tmp = train_test_split(indices, test_size=p_val + p_pru, stratify=estrato, random_state=SEMILLA)
idx_val, idx_pru = train_test_split(idx_tmp, test_size=p_pru / (p_val + p_pru),
                                    stratify=clave_estrato(datos['Categoria'].to_numpy()[idx_tmp], minimo=2),
                                    random_state=SEMILLA)
idx_ent, idx_val, idx_pru = np.sort(idx_ent), np.sort(idx_val), np.sort(idx_pru)

cat = datos['Categoria'].to_numpy()
t04 = pd.DataFrame({
    'Categoría': CATEGORIAS,
    'Entrenamiento': [int((cat[idx_ent] == c).sum()) for c in CATEGORIAS],
    'Validación': [int((cat[idx_val] == c).sum()) for c in CATEGORIAS],
    'Prueba': [int((cat[idx_pru] == c).sum()) for c in CATEGORIAS],
})
t04['Total'] = t04[['Entrenamiento', 'Validación', 'Prueba']].sum(axis=1)
t04.loc[len(t04)] = ['Total'] + [int(t04[c].sum()) for c in ['Entrenamiento', 'Validación', 'Prueba', 'Total']]
guardar_tabla(t04, 'T04_particion', 'Registros por categoría en los conjuntos de entrenamiento, validación y prueba')
RESUMEN['tamanos_particion'] = {'entrenamiento': len(idx_ent), 'validacion': len(idx_val), 'prueba': len(idx_pru)}

# %% [markdown]
# ## Fase 3 · Preprocesamiento y selección de características (apartado 4.4.3)
# Todo se ajusta solo con el conjunto de entrenamiento y después se aplica a validación y prueba.

# %%
t0 = time.perf_counter()
X_ent_bruto = datos.loc[idx_ent, CARACT]

# Características constantes en entrenamiento
constantes = [c for c in CARACT if X_ent_bruto[c].max() == X_ent_bruto[c].min()]

# Correlación de Pearson calculada por bloques (ahorra memoria) sobre el conjunto de entrenamiento
def matriz_correlacion(df, columnas, bloque=200_000):
    n = len(df)
    medias = df[columnas].mean().to_numpy(np.float64)
    std = df[columnas].std(ddof=0).to_numpy(np.float64)
    acum = np.zeros((len(columnas), len(columnas)))
    for i in range(0, n, bloque):
        z = (df[columnas].iloc[i:i + bloque].to_numpy(np.float64) - medias) / std
        acum += z.T @ z
    return pd.DataFrame(acum / n, index=columnas, columns=columnas)

no_constantes = [c for c in CARACT if c not in constantes]
corr = matriz_correlacion(X_ent_bruto, no_constantes)
umbral = CFG['UMBRAL_CORRELACION']
conservadas, correlacionadas = [], []
for c in no_constantes:                       # recorrido en el orden original: se conserva la primera de cada par
    if conservadas:
        r = corr.loc[c, conservadas].abs()
        if r.max() > umbral:
            correlacionadas.append((c, r.idxmax(), float(corr.loc[c, r.idxmax()])))
            continue
    conservadas.append(c)
CARACT_SEL = conservadas

t05 = pd.DataFrame(
    [(c, 'Constante en entrenamiento', '') for c in constantes] +
    [(c, f'Correlación > {coma(umbral)}', f'{o} (r = {coma(r, 3)})') for c, o, r in correlacionadas],
    columns=['Característica eliminada', 'Motivo', 'Correlacionada con'])
guardar_tabla(t05, 'T05_caracteristicas_eliminadas', 'Características eliminadas en el preprocesamiento')
print(f'Características: {len(CARACT)} tras la limpieza -> {len(CARACT_SEL)} seleccionadas '
      f'({len(constantes)} constantes, {len(correlacionadas)} correlacionadas)')
RESUMEN['n_caracteristicas_seleccionadas'] = len(CARACT_SEL)
RESUMEN['caracteristicas_seleccionadas'] = CARACT_SEL

# Estandarización
escalador = StandardScaler().fit(X_ent_bruto[CARACT_SEL].to_numpy(np.float32))
def preparar(idx):
    return escalador.transform(datos.loc[idx, CARACT_SEL].to_numpy(np.float32)).astype(np.float32)
X_ent, X_val, X_pru = preparar(idx_ent), preparar(idx_val), preparar(idx_pru)
del X_ent_bruto

# Etiquetas
y_bin = (datos['Label'] != 'BENIGN').astype(np.int8).to_numpy()
cod_multi = {c: i for i, c in enumerate(CLASES_MULTI)}
y_multi = datos['Categoria'].map(cod_multi).fillna(-1).astype(np.int8).to_numpy()    # -1 = Heartbleed

conj = {}
for nombre, idx, X in (('ent', idx_ent, X_ent), ('val', idx_val, X_val), ('pru', idx_pru, X_pru)):
    m = y_multi[idx] >= 0
    conj[nombre] = dict(idx=idx, X=X, y_bin=y_bin[idx], cat=cat[idx],
                        X_m=X[m], y_m=y_multi[idx][m], idx_m=idx[m])
joblib.dump({'caracteristicas': CARACT_SEL, 'escalador': escalador}, DIR_MOD / 'preprocesamiento.joblib')
cronometro('preprocesamiento', t0)

# Estadísticos descriptivos de algunas características relevantes, por categoría (datos limpios, sin escalar)
DESCRIPTIVAS = ['Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
                'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Fwd Packet Length Max', 'Init_Win_bytes_forward']
DESCRIPTIVAS = [c for c in DESCRIPTIVAS if c in CARACT]
filas = []
for c in DESCRIPTIVAS:
    g = datos.groupby('Categoria')[c]
    for est, serie in (('Mediana', g.median()), ('Media', g.mean()), ('Desv. típica', g.std())):
        filas.append([c, est] + [float(serie.get(k, np.nan)) for k in CATEGORIAS])
t06 = pd.DataFrame(filas, columns=['Característica', 'Estadístico'] + CATEGORIAS)
guardar_tabla(t06, 'T06_estadisticos_descriptivos', 'Estadísticos descriptivos de características relevantes por categoría')

# %% [markdown]
# ## Fase 4 · Tratamiento del desbalanceo con SMOTE (apartado 4.4.4)
# Solo en la tarea multiclase (escenario E3). Las categorías con menos de 20.000 registros en entrenamiento se
# sobremuestrean hasta 20.000. Cuando SMOTE se aplica sobre una submuestra (búsqueda de hiperparámetros o SVM), el
# mínimo se escala en la misma proporción, para que la clase sobremuestreada pese lo mismo que en el entrenamiento completo.

# %%
N_REF_MULTI = len(conj['ent']['y_m'])

def smote_proporcional(X, y, minimo, n_ref, k, semilla):
    """SMOTE con un mínimo de registros por clase proporcional al tamaño del conjunto."""
    clases, cuentas = np.unique(y, return_counts=True)
    objetivo = int(round(minimo * len(y) / n_ref))
    destino = {int(c): objetivo for c, n in zip(clases, cuentas) if 2 <= n < objetivo}
    if not destino:
        return X, y
    k_real = int(min(k, min(cuentas[np.isin(clases, list(destino))]) - 1))
    sm = SMOTE(sampling_strategy=destino, k_neighbors=k_real, random_state=semilla)
    return sm.fit_resample(X, y)

def muestreador_smote():
    return FunctionSampler(func=smote_proporcional, validate=False,
                           kw_args=dict(minimo=CFG['SMOTE_MINIMO'], n_ref=N_REF_MULTI, k=CFG['SMOTE_K'], semilla=SEMILLA))

t0 = time.perf_counter()
X_ent_smote, y_ent_smote = smote_proporcional(conj['ent']['X_m'], conj['ent']['y_m'],
                                              CFG['SMOTE_MINIMO'], N_REF_MULTI, CFG['SMOTE_K'], SEMILLA)
X_ent_smote = X_ent_smote.astype(np.float32)
antes_s = np.bincount(conj['ent']['y_m'], minlength=5)
despues_s = np.bincount(y_ent_smote, minlength=5)
t07 = pd.DataFrame({
    'Categoría': CLASES_MULTI,
    'Registros en entrenamiento': antes_s,
    'Estrategia': ['SMOTE' if d > a else 'Sin modificar' for a, d in zip(antes_s, despues_s)],
    'Registros tras SMOTE': despues_s,
    'Factor': np.round(despues_s / np.maximum(antes_s, 1), 2),
})
guardar_tabla(t07, 'T07_smote', 'Estrategia de sobremuestreo con SMOTE en el conjunto de entrenamiento (cifras reales)')
RESUMEN['smote'] = {c: [int(a), int(d)] for c, a, d in zip(CLASES_MULTI, antes_s, despues_s)}
cronometro('SMOTE', t0)

# %% [markdown]
# ## Fase 5 · Entrenamiento y ajuste de hiperparámetros (apartado 4.4.5)
# Tres escenarios × cuatro algoritmos = doce modelos:
# - **E1**: binaria, distribución original · **E2**: multiclase, distribución original · **E3**: multiclase con SMOTE.
#
# RF, XGBoost y SVM: búsqueda aleatoria (20 combinaciones, o todas si hay menos) con validación cruzada estratificada
# de 3 particiones sobre una submuestra del 20 % del entrenamiento; la SVM usa una submuestra de 50.000 registros.
# DNN: las 24 configuraciones de la Tabla 7 se comparan sobre el conjunto de validación (entrenando con la misma
# submuestra del 20 %) y la mejor se reentrena con el entrenamiento completo.

# %%
ESPACIOS = {
    'Random Forest': {'n_estimators': [100, 200, 300], 'max_depth': [None, 20, 30],
                      'min_samples_split': [2, 5, 10], 'max_features': ['sqrt', 'log2']},
    'XGBoost': {'n_estimators': [100, 300, 500], 'max_depth': [4, 6, 8, 10], 'learning_rate': [0.05, 0.1, 0.2],
                'subsample': [0.8, 1.0], 'colsample_bytree': [0.8, 1.0]},
    'SVM': {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 0.01, 0.1]},
}
ESPACIO_DNN = {'capas': [(128, 64), (128, 64, 32), (256, 128, 64)], 'dropout': [0.2, 0.3],
               'lr': [0.001, 0.0001], 'batch': [512, 1024]}
if MODO_PRUEBA:
    ESPACIO_DNN = {'capas': [(128, 64, 32)], 'dropout': [0.3], 'lr': [0.001], 'batch': [512, 1024]}
MODELOS = list(COLOR_MODELO)
ESCENARIOS = OrderedDict([
    ('E1', dict(tarea='binaria', smote=False, titulo='E1 · Binaria (distribución original)')),
    ('E2', dict(tarea='multiclase', smote=False, titulo='E2 · Multiclase (distribución original)')),
    ('E3', dict(tarea='multiclase', smote=True, titulo='E3 · Multiclase con SMOTE')),
])

def dispositivo_xgb():
    """'cuda' solo si XGBoost tiene soporte CUDA y hay una GPU NVIDIA visible; si no, 'cpu'."""
    import xgboost
    if not xgboost.build_info().get('USE_CUDA', False):
        return 'cpu'
    try:
        with warnings.catch_warnings(record=True) as avisos_xgb:
            warnings.simplefilter('always')
            XGBClassifier(n_estimators=1, device='cuda', tree_method='hist').fit(np.zeros((4, 2)), [0, 1, 0, 1])
        if any('GPU' in str(a.message) or 'CPU' in str(a.message) or 'cuda' in str(a.message).lower() for a in avisos_xgb):
            return 'cpu'
        return 'cuda'
    except Exception:
        return 'cpu'
DISPOSITIVO_XGB = dispositivo_xgb()
print('XGBoost usará:', DISPOSITIVO_XGB, '| GPU para TensorFlow:', tf.config.list_physical_devices('GPU'))

def crear_modelo(nombre, tarea, en_busqueda):
    nj = 1 if en_busqueda else CFG['N_JOBS']
    if nombre == 'Random Forest':
        return RandomForestClassifier(random_state=SEMILLA, n_jobs=nj)
    if nombre == 'XGBoost':
        extra = dict(objective='binary:logistic', eval_metric='logloss') if tarea == 'binaria' else \
                dict(objective='multi:softprob', eval_metric='mlogloss')
        return XGBClassifier(tree_method='hist', device=DISPOSITIVO_XGB, random_state=SEMILLA,
                             n_jobs=(nj if DISPOSITIVO_XGB == 'cpu' else 1), **extra)
    if nombre == 'SVM':
        return SVC(kernel='rbf', cache_size=1000, random_state=SEMILLA)
    raise ValueError(nombre)

def submuestra(y_estrato, n=None, fraccion=None, semilla=SEMILLA):
    """Índices de una submuestra estratificada."""
    total = len(y_estrato)
    n = int(round(total * fraccion)) if n is None else int(n)
    if n >= total:
        return np.arange(total)
    idx, _ = train_test_split(np.arange(total), train_size=n, stratify=clave_estrato(y_estrato, minimo=2),
                              random_state=semilla)
    return np.sort(idx)

def datos_escenario(esc):
    """Conjuntos de entrenamiento, validación y prueba de cada escenario."""
    if ESCENARIOS[esc]['tarea'] == 'binaria':
        e, v, p = conj['ent'], conj['val'], conj['pru']
        return dict(X_ent=e['X'], y_ent=e['y_bin'], estrato_ent=e['cat'], X_val=v['X'], y_val=v['y_bin'],
                    X_pru=p['X'], y_pru=p['y_bin'], idx_pru=p['idx'])
    e, v, p = conj['ent'], conj['val'], conj['pru']
    return dict(X_ent=e['X_m'], y_ent=e['y_m'], estrato_ent=e['y_m'], X_val=v['X_m'], y_val=v['y_m'],
                X_pru=p['X_m'], y_pru=p['y_m'], idx_pru=p['idx_m'])

def ruta_modelo(esc, nombre):
    base = DIR_MOD / f'{esc}_{nombre.replace(" ", "_")}'
    return base.with_suffix('.keras') if nombre == 'DNN' else base.with_suffix('.joblib')

def ruta_info(esc, nombre):
    return DIR_MOD / f'{esc}_{nombre.replace(" ", "_")}_info.json'

def busqueda_clasica(esc, nombre):
    """Búsqueda aleatoria + validación cruzada estratificada y entrenamiento final."""
    d = datos_escenario(esc)
    tarea, usa_smote = ESCENARIOS[esc]['tarea'], ESCENARIOS[esc]['smote']
    if nombre == 'SVM':
        sel = submuestra(d['estrato_ent'], n=min(CFG['SVM_N'], len(d['y_ent'])))
    else:
        sel = submuestra(d['estrato_ent'], fraccion=CFG['FRACCION_BUSQUEDA'])
    X_b, y_b = d['X_ent'][sel], d['y_ent'][sel]
    estimador = crear_modelo(nombre, tarea, en_busqueda=True)
    espacio = dict(ESPACIOS[nombre])
    if usa_smote:
        estimador = ImbPipeline([('smote', muestreador_smote()), ('clf', estimador)])
        espacio = {f'clf__{k}': v for k, v in espacio.items()}
    n_comb = math.prod(len(v) for v in espacio.values())
    busqueda = RandomizedSearchCV(
        estimador, espacio, n_iter=min(CFG['N_ITER'], n_comb),
        scoring='f1' if tarea == 'binaria' else 'f1_macro',
        cv=StratifiedKFold(CFG['CV_PARTICIONES'], shuffle=True, random_state=SEMILLA),
        n_jobs=(CFG['N_JOBS'] if not (nombre == 'XGBoost' and DISPOSITIVO_XGB == 'cuda') else 1),
        random_state=SEMILLA, refit=False, error_score='raise')
    t0 = time.perf_counter()
    busqueda.fit(X_b, y_b)
    t_busqueda = time.perf_counter() - t0
    i = busqueda.best_index_
    mejores = {k.replace('clf__', ''): v for k, v in busqueda.best_params_.items()}
    info = dict(escenario=esc, modelo=nombre, mejores_parametros=mejores, n_combinaciones=int(min(CFG['N_ITER'], n_comb)),
                n_busqueda=int(len(y_b)), f1_cv_media=float(busqueda.cv_results_['mean_test_score'][i]),
                f1_cv_dt=float(busqueda.cv_results_['std_test_score'][i]), t_busqueda_s=t_busqueda)
    # Entrenamiento final con la mejor configuración
    if nombre == 'SVM':
        X_f, y_f = X_b, y_b
        if usa_smote:
            X_f, y_f = smote_proporcional(X_f, y_f, CFG['SMOTE_MINIMO'], N_REF_MULTI, CFG['SMOTE_K'], SEMILLA)
    elif usa_smote:
        X_f, y_f = X_ent_smote, y_ent_smote
    else:
        X_f, y_f = d['X_ent'], d['y_ent']
    modelo = crear_modelo(nombre, tarea, en_busqueda=False).set_params(**mejores)
    t0 = time.perf_counter()
    modelo.fit(X_f, y_f)
    info.update(t_entrenamiento_s=time.perf_counter() - t0, n_entrenamiento=int(len(y_f)))
    return modelo, info

def construir_dnn(n_entrada, n_salida, capas, dropout, lr):
    keras.utils.set_random_seed(SEMILLA)
    modelo = keras.Sequential([keras.Input(shape=(n_entrada,))])
    for i, n in enumerate(capas):
        modelo.add(keras.layers.Dense(n, activation='relu'))
        if i < 2:                                     # dropout tras las dos primeras capas ocultas
            modelo.add(keras.layers.Dropout(dropout))
    if n_salida == 1:
        modelo.add(keras.layers.Dense(1, activation='sigmoid'))
        perdida = 'binary_crossentropy'
    else:
        modelo.add(keras.layers.Dense(n_salida, activation='softmax'))
        perdida = 'sparse_categorical_crossentropy'
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=perdida)
    return modelo

def entrenar_dnn(modelo, X, y, X_val, y_val, batch):
    parada = keras.callbacks.EarlyStopping(monitor='val_loss', patience=CFG['DNN_PACIENCIA'], restore_best_weights=True)
    hist = modelo.fit(X, y, validation_data=(X_val, y_val), epochs=CFG['DNN_EPOCAS_MAX'], batch_size=batch,
                      callbacks=[parada], verbose=0, shuffle=True)
    return hist.history

def predecir_dnn(modelo, X, tarea):
    p = modelo.predict(X, batch_size=8192, verbose=0)
    return (p.ravel() >= 0.5).astype(np.int8) if tarea == 'binaria' else p.argmax(axis=1).astype(np.int8)

def busqueda_dnn(esc):
    d = datos_escenario(esc)
    tarea, usa_smote = ESCENARIOS[esc]['tarea'], ESCENARIOS[esc]['smote']
    n_salida = 1 if tarea == 'binaria' else len(CLASES_MULTI)
    sel = submuestra(d['estrato_ent'], fraccion=CFG['DNN_FRACCION_BUSQUEDA'])
    X_b, y_b = d['X_ent'][sel], d['y_ent'][sel]
    if usa_smote:
        X_b, y_b = smote_proporcional(X_b, y_b, CFG['SMOTE_MINIMO'], N_REF_MULTI, CFG['SMOTE_K'], SEMILLA)
    registros = []
    t0 = time.perf_counter()
    claves = list(ESPACIO_DNN)
    for valores in itertools.product(*ESPACIO_DNN.values()):
        p = dict(zip(claves, valores))
        modelo = construir_dnn(X_b.shape[1], n_salida, p['capas'], p['dropout'], p['lr'])
        hist = entrenar_dnn(modelo, X_b, y_b, d['X_val'], d['y_val'], p['batch'])
        pred = predecir_dnn(modelo, d['X_val'], tarea)
        f1v = f1_score(d['y_val'], pred, average='binary' if tarea == 'binaria' else 'macro', zero_division=0)
        registros.append(dict(p, f1_validacion=f1v, epocas=len(hist['loss'])))
        print(f'   {esc} DNN {p} -> F1 validación = {f1v:.4f} ({len(hist["loss"])} épocas)')
        keras.backend.clear_session()
    t_busqueda = time.perf_counter() - t0
    tabla = pd.DataFrame(registros)
    mejor = tabla.sort_values('f1_validacion', ascending=False).iloc[0]
    mejores = dict(capas=tuple(mejor['capas']), dropout=float(mejor['dropout']), lr=float(mejor['lr']), batch=int(mejor['batch']))
    # Entrenamiento final con todo el conjunto de entrenamiento
    X_f, y_f = (X_ent_smote, y_ent_smote) if usa_smote else (d['X_ent'], d['y_ent'])
    modelo = construir_dnn(X_f.shape[1], n_salida, mejores['capas'], mejores['dropout'], mejores['lr'])
    t0 = time.perf_counter()
    hist = entrenar_dnn(modelo, X_f, y_f, d['X_val'], d['y_val'], mejores['batch'])
    t_ent = time.perf_counter() - t0
    pred = predecir_dnn(modelo, d['X_val'], tarea)
    f1v = f1_score(d['y_val'], pred, average='binary' if tarea == 'binaria' else 'macro', zero_division=0)
    info = dict(escenario=esc, modelo='DNN', mejores_parametros={k: (list(v) if isinstance(v, tuple) else v) for k, v in mejores.items()},
                n_combinaciones=len(registros), n_busqueda=int(len(y_b)), f1_cv_media=np.nan, f1_cv_dt=np.nan,
                f1_validacion=float(f1v), t_busqueda_s=t_busqueda, t_entrenamiento_s=t_ent, n_entrenamiento=int(len(y_f)),
                epocas=len(hist['loss']), historial={k: [float(x) for x in v] for k, v in hist.items()},
                busqueda=[{k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()} for r in registros])
    return modelo, info

# %%
MODELOS_ENTRENADOS, INFO = {}, {}
for esc in ESCENARIOS:
    for nombre in MODELOS:
        rm, ri = ruta_modelo(esc, nombre), ruta_info(esc, nombre)
        if rm.exists() and ri.exists() and not FORZAR:
            modelo = keras.models.load_model(rm) if nombre == 'DNN' else joblib.load(rm)
            info = json.loads(ri.read_text(encoding='utf-8'))
            print(f'{esc} · {nombre}: cargado de {rm.name}')
        else:
            print(f'\n>>> {esc} · {nombre}: búsqueda de hiperparámetros y entrenamiento ...')
            t0 = time.perf_counter()
            modelo, info = busqueda_dnn(esc) if nombre == 'DNN' else busqueda_clasica(esc, nombre)
            if nombre == 'DNN':
                modelo.save(rm)
            else:
                joblib.dump(modelo, rm, compress=3)
            ri.write_text(json.dumps(info, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
            cronometro(f'{esc} {nombre}', t0)
        MODELOS_ENTRENADOS[(esc, nombre)] = modelo
        INFO[(esc, nombre)] = info
        print(f'    mejores parámetros: {info["mejores_parametros"]}')

filas = []
for (esc, nombre), info in INFO.items():
    filas.append([esc, nombre, ', '.join(f'{k} = {v}' for k, v in info['mejores_parametros'].items()),
                  info['n_combinaciones'], info['n_busqueda'], info['n_entrenamiento'],
                  info['f1_cv_media'], info['f1_cv_dt'], info.get('f1_validacion', np.nan),
                  info['t_busqueda_s'], info['t_entrenamiento_s']])
t08 = pd.DataFrame(filas, columns=['Escenario', 'Modelo', 'Mejores hiperparámetros', 'Combinaciones probadas',
                                   'Registros en la búsqueda', 'Registros en el entrenamiento final',
                                   'F1 VC (media)', 'F1 VC (desv. típica)', 'F1 validación (DNN)',
                                   'Tiempo de búsqueda (s)', 'Tiempo de entrenamiento final (s)'])
guardar_tabla(t08, 'T08_hiperparametros', 'Hiperparámetros óptimos, tamaño de los conjuntos y tiempos de cada modelo')

# Curvas de aprendizaje de la DNN (pérdida en entrenamiento y validación)
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=False)
for ax, esc in zip(axes, ESCENARIOS):
    h = INFO[(esc, 'DNN')]['historial']
    ep = np.arange(1, len(h['loss']) + 1)
    ax.plot(ep, h['loss'], color=AZUL, lw=1.6, label='Entrenamiento')
    ax.plot(ep, h['val_loss'], color='#eb6834', lw=1.6, ls='--', label='Validación')
    ax.set_title(esc)
    ax.set_xlabel('Época')
    ax.grid(axis='y')
    formato_es(ax)
axes[0].set_ylabel('Pérdida (entropía cruzada)')
axes[0].legend(fontsize=7.5)
guardar_figura(fig, 'F02_dnn_curvas_aprendizaje', 'Evolución de la pérdida de la DNN en entrenamiento y validación')

# %% [markdown]
# ## Fase 6 · Evaluación sobre el conjunto de prueba (apartado 4.5)

# %%
def predecir_y_medir(esc, nombre, X):
    """Predicción de etiquetas (cronometrada) y puntuaciones para el AUC."""
    tarea = ESCENARIOS[esc]['tarea']
    modelo = MODELOS_ENTRENADOS[(esc, nombre)]
    t0 = time.perf_counter()
    pred = predecir_dnn(modelo, X, tarea) if nombre == 'DNN' else modelo.predict(X).astype(np.int8)
    t_inf = time.perf_counter() - t0
    if nombre == 'DNN':
        p = modelo.predict(X, batch_size=8192, verbose=0)
        punt = p.ravel() if tarea == 'binaria' else p
    elif nombre == 'SVM':
        punt = modelo.decision_function(X)
    else:
        p = modelo.predict_proba(X)
        punt = p[:, 1] if tarea == 'binaria' else p
    return pred, punt, t_inf

def auc_ovr_macro(y, punt, k):
    return float(np.mean([roc_auc_score(y == c, punt[:, c]) for c in range(k) if 0 < (y == c).sum() < len(y)]))

RES = {}
for esc in ESCENARIOS:
    d = datos_escenario(esc)
    tarea = ESCENARIOS[esc]['tarea']
    for nombre in MODELOS:
        pred, punt, t_inf = predecir_y_medir(esc, nombre, d['X_pru'])
        y = d['y_pru']
        if tarea == 'binaria':
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            p_, r_, f_, _ = precision_recall_fscore_support(y, pred, average='binary', zero_division=0)
            auc = roc_auc_score(y, punt)
            fpr = fp / (fp + tn)
            extra = dict(VP=int(tp), FP=int(fp), FN=int(fn), VN=int(tn))
        else:
            p_, r_, f_, _ = precision_recall_fscore_support(y, pred, average='macro', zero_division=0)
            auc = auc_ovr_macro(y, punt, len(CLASES_MULTI))
            fpr = float(((y == 0) & (pred != 0)).sum() / max((y == 0).sum(), 1))
            pc, rc, fc, sc = precision_recall_fscore_support(y, pred, labels=range(len(CLASES_MULTI)), zero_division=0)
            extra = dict(por_clase=pd.DataFrame({'Categoría': CLASES_MULTI, 'Precisión': pc, 'Sensibilidad': rc,
                                                 'F1': fc, 'Soporte': sc}))
        RES[(esc, nombre)] = dict(pred=pred, punt=punt, y=y, precision=p_, sensibilidad=r_, f1=f_, auc=auc, fpr=fpr,
                                  exactitud=accuracy_score(y, pred), t_inf_ms_1000=t_inf / len(y) * 1000 * 1000, **extra)
        print(f'{esc} · {nombre:13s} F1 = {f_:.4f} · FPR = {fpr:.5f} · AUC = {auc:.4f}')

def tabla_metricas(esc):
    filas = []
    for nombre in MODELOS:
        r, info = RES[(esc, nombre)], INFO[(esc, nombre)]
        filas.append([nombre, r['precision'], r['sensibilidad'], r['f1'], r['auc'], r['fpr'], r['exactitud'],
                      info['f1_cv_media'], info['f1_cv_dt'], info['t_entrenamiento_s'], r['t_inf_ms_1000']])
    sufijo = '' if ESCENARIOS[esc]['tarea'] == 'binaria' else ' (macro)'
    return pd.DataFrame(filas, columns=['Modelo', 'Precisión' + sufijo, 'Sensibilidad' + sufijo, 'F1' + sufijo,
                                        'AUC-ROC', 'FPR', 'Exactitud', 'F1 VC (media)', 'F1 VC (desv. típica)',
                                        'Tiempo de entrenamiento (s)', 'Inferencia (ms / 1.000 flujos)'])

guardar_tabla(tabla_metricas('E1'), 'T09_E1_metricas', 'Escenario E1 (binaria): métricas sobre el conjunto de prueba')
guardar_tabla(tabla_metricas('E2'), 'T10_E2_metricas', 'Escenario E2 (multiclase): métricas macro sobre el conjunto de prueba')
guardar_tabla(tabla_metricas('E3'), 'T11_E3_metricas', 'Escenario E3 (multiclase con SMOTE): métricas macro sobre el conjunto de prueba')

t09b = pd.DataFrame([[n, RES[('E1', n)]['VP'], RES[('E1', n)]['FP'], RES[('E1', n)]['FN'], RES[('E1', n)]['VN']]
                     for n in MODELOS], columns=['Modelo', 'VP', 'FP', 'FN', 'VN'])
guardar_tabla(t09b, 'T09b_E1_matriz_confusion', 'Escenario E1: verdaderos/falsos positivos y negativos en prueba')

# Métricas por categoría en E2 y E3
for esc, nombre_t in (('E2', 'T12_E2_por_clase'), ('E3', 'T13_E3_por_clase')):
    partes = []
    for nombre in MODELOS:
        pc = RES[(esc, nombre)]['por_clase'].copy()
        pc.insert(0, 'Modelo', nombre)
        partes.append(pc)
    guardar_tabla(pd.concat(partes, ignore_index=True), nombre_t, f'Escenario {esc}: precisión, sensibilidad y F1 por categoría')

# Efecto de SMOTE (E3 - E2) por categoría (PI2)
filas = []
for nombre in MODELOS:
    a, b = RES[('E2', nombre)]['por_clase'], RES[('E3', nombre)]['por_clase']
    for i, c in enumerate(CLASES_MULTI):
        filas.append([nombre, c, a['Sensibilidad'][i], b['Sensibilidad'][i], b['Sensibilidad'][i] - a['Sensibilidad'][i],
                      a['F1'][i], b['F1'][i], b['F1'][i] - a['F1'][i]])
    filas.append([nombre, 'FPR (benigno)', np.nan, np.nan, np.nan, RES[('E2', nombre)]['fpr'], RES[('E3', nombre)]['fpr'],
                  RES[('E3', nombre)]['fpr'] - RES[('E2', nombre)]['fpr']])
t14 = pd.DataFrame(filas, columns=['Modelo', 'Categoría', 'Sensibilidad E2', 'Sensibilidad E3', 'Δ sensibilidad',
                                   'F1 E2 (o FPR)', 'F1 E3 (o FPR)', 'Δ'])
guardar_tabla(t14, 'T14_efecto_smote', 'Efecto de SMOTE: diferencia E3 − E2 por categoría y en la tasa de falsos positivos')

# %%
# --- Figuras: matrices de confusión y curvas ROC por escenario ---
def figura_matrices(esc, nombre_fig):
    tarea = ESCENARIOS[esc]['tarea']
    etiquetas = ['Benigno', 'Ataque'] if tarea == 'binaria' else CLASES_MULTI
    k = len(etiquetas)
    tam = (7.0, 6.4) if k > 2 else (6.0, 5.2)
    fig, axes = plt.subplots(2, 2, figsize=tam)
    for ax, nombre in zip(axes.ravel(), MODELOS):
        r = RES[(esc, nombre)]
        cm = confusion_matrix(r['y'], r['pred'], labels=range(k))
        cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(cmn, cmap=CMAP_SECUENCIAL, vmin=0, vmax=1)
        for i in range(k):
            for j in range(k):
                n_txt = f'{cm[i, j]:,}'.replace(',', '.')
                if cm[i, j] == 0:
                    txt = '0'
                elif k == 2:
                    txt = f'{coma(cmn[i, j] * 100, 2)} %\n({n_txt})'
                else:
                    txt = f'{coma(cmn[i, j] * 100, 1)}\n{n_txt}'
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=(8 if k == 2 else 5.6), color=('white' if cmn[i, j] > 0.55 else TINTA))
        ax.set_xticks(range(k), etiquetas, fontsize=(8 if k == 2 else 6.5), rotation=(0 if k == 2 else 35))
        ax.set_yticks(range(k), etiquetas, fontsize=(8 if k == 2 else 6.5))
        ax.set_title(nombre, color=COLOR_MODELO[nombre], fontweight='bold')
        ax.tick_params(length=0)
        ax.set_xlabel('Clase predicha', fontsize=8)
        ax.set_ylabel('Clase real', fontsize=8)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    nota = '' if k == 2 else ' (porcentaje por fila y número de flujos)'
    guardar_figura(fig, nombre_fig, f'Matrices de confusión del escenario {esc} en el conjunto de prueba{nota}')

def curva_roc_macro(y, punt, k):
    rejilla = np.concatenate([[0], np.logspace(-7, 0, 400)])
    tprs = []
    for c in range(k):
        if 0 < (y == c).sum() < len(y):
            f, t, _ = roc_curve(y == c, punt[:, c])
            tprs.append(np.interp(rejilla, f, t))
    return rejilla, np.mean(tprs, axis=0)

def figura_roc(esc, nombre_fig):
    tarea = ESCENARIOS[esc]['tarea']
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.0), sharex=True, sharey=True)
    for ax, nombre in zip(axes.ravel(), MODELOS):
        r = RES[(esc, nombre)]
        if tarea == 'binaria':
            f, t, _ = roc_curve(r['y'], r['punt'])
        else:
            f, t = curva_roc_macro(r['y'], r['punt'], len(CLASES_MULTI))
        f = np.clip(f, 1e-6, 1)
        ax.plot(f, t, color=COLOR_MODELO[nombre], lw=2)
        ax.set_xscale('log')
        formato_es(ax, 'y')
        ax.set_xlim(1e-5, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(True, which='major')
        ax.set_title(f'{nombre} · AUC = {coma(r["auc"], 4)}', color=COLOR_MODELO[nombre], fontweight='bold', fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel('Tasa de falsos positivos (escala log.)')
    for ax in axes[:, 0]:
        ax.set_ylabel('Tasa de verdaderos positivos')
    fig.tight_layout()
    tipo = 'curvas ROC' if tarea == 'binaria' else 'curvas ROC macro (uno contra el resto)'
    guardar_figura(fig, nombre_fig, f'Escenario {esc}: {tipo} en el conjunto de prueba')

figura_matrices('E1', 'F03_E1_matrices_confusion'); figura_roc('E1', 'F04_E1_curvas_roc')
figura_matrices('E2', 'F05_E2_matrices_confusion'); figura_roc('E2', 'F06_E2_curvas_roc')
figura_matrices('E3', 'F07_E3_matrices_confusion'); figura_roc('E3', 'F08_E3_curvas_roc')

# Comparación global: F1 y FPR por modelo y escenario (gráfico de puntos)
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
ypos = np.arange(len(MODELOS))
desplaz = {'E1': -0.18, 'E2': 0.0, 'E3': 0.18}
for esc in ESCENARIOS:
    f1s = [RES[(esc, n)]['f1'] for n in MODELOS]
    fprs = [max(RES[(esc, n)]['fpr'], 1e-6) for n in MODELOS]
    axes[0].scatter(f1s, ypos + desplaz[esc], s=38, color=COLOR_ESCENARIO[esc], edgecolor='white', lw=1.2, zorder=3, label=esc)
    axes[1].scatter(fprs, ypos + desplaz[esc], s=38, color=COLOR_ESCENARIO[esc], edgecolor='white', lw=1.2, zorder=3)
axes[0].set_yticks(ypos, MODELOS)
axes[0].invert_yaxis()
axes[0].set_xlabel('F1 (macro en E2 y E3)')
axes[1].set_xlabel('Tasa de falsos positivos (escala log.)')
axes[1].set_xscale('log')
for ax in axes:
    ax.grid(axis='x', which='major')
    ax.set_axisbelow(True)
axes[0].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: coma(v, 3)))
fig.tight_layout(rect=(0, 0, 1, 0.9))
manejadores, etiquetas = axes[0].get_legend_handles_labels()
fig.legend(manejadores, etiquetas, loc='upper center', ncol=3, fontsize=8, frameon=False,
           bbox_to_anchor=(0.5, 1.0), title='Escenario', title_fontsize=8)
guardar_figura(fig, 'F09_comparacion_modelos', 'Comparación del F1 y de la tasa de falsos positivos de los doce modelos')

# %%
# --- Prueba de McNemar entre pares de modelos (con corrección de Bonferroni) ---
N_PARES = math.comb(len(MODELOS), 2)
ALFA_BONF = CFG['ALFA'] / N_PARES

def prueba_mcnemar(y, pa, pb):
    ca, cb = (pa == y), (pb == y)
    b, c = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
    if b + c == 0:
        return b, c, 0.0, 1.0, 'sin discrepancias'
    if b + c < 25:
        res = mcnemar([[0, b], [c, 0]], exact=True)
        return b, c, float(res.statistic), float(res.pvalue), 'exacta (binomial)'
    res = mcnemar([[0, b], [c, 0]], exact=False, correction=True)
    return b, c, float(res.statistic), float(res.pvalue), 'χ² con corrección'

filas = []
for esc in ESCENARIOS:
    for a, b_ in itertools.combinations(MODELOS, 2):
        ra, rb = RES[(esc, a)], RES[(esc, b_)]
        bb, cc, est, p, metodo = prueba_mcnemar(ra['y'], ra['pred'], rb['pred'])
        filas.append([esc, a, b_, bb, cc, est, p, 'Sí' if p < ALFA_BONF else 'No', metodo])
t15 = pd.DataFrame(filas, columns=['Escenario', 'Modelo A', 'Modelo B', 'b (A acierta, B falla)', 'c (A falla, B acierta)',
                                   'Estadístico', 'p-valor', f'¿Significativa? (α = {ALFA_BONF:.4f})', 'Versión de la prueba'])
guardar_tabla(t15, 'T15_mcnemar', 'Prueba de McNemar entre cada par de modelos (corrección de Bonferroni)')

# --- Selección del mejor modelo con el criterio del apartado 4.5 ---
def seleccionar(candidatos, alfa):
    """1) mayor F1; 2) entre los que no difieren significativamente del primero, menor FPR; 3) menor tiempo de inferencia."""
    orden = sorted(candidatos, key=lambda k: -RES[k]['f1'])
    primero = orden[0]
    traza, empatados = [], [primero]
    for k in orden[1:]:
        _, _, est, p, _ = prueba_mcnemar(RES[primero]['y'], RES[primero]['pred'], RES[k]['pred'])
        sig = p < alfa
        traza.append((k, p, sig))
        if not sig:
            empatados.append(k)
    elegido = min(empatados, key=lambda k: (RES[k]['fpr'], RES[k]['t_inf_ms_1000']))
    return elegido, primero, traza, empatados

filas, MEJOR = [], {}
for esc in ESCENARIOS:
    elegido, primero, traza, empatados = seleccionar([(esc, n) for n in MODELOS], ALFA_BONF)
    MEJOR[esc] = elegido
    detalle = '; '.join(f'{k[1]}: p = {p:.2e} ({"distinto" if s else "sin diferencia significativa"})' for k, p, s in traza)
    filas.append([esc, primero[1], detalle, ', '.join(k[1] for k in empatados), elegido[1]])
# Mejor modelo multiclase: comparación directa del mejor de E2 con el mejor de E3 (una sola prueba, α = 0,05)
el_m, pr_m, traza_m, emp_m = seleccionar([MEJOR['E2'], MEJOR['E3']], CFG['ALFA'])
MEJOR['multiclase'] = el_m
MEJOR['binario'] = MEJOR['E1']
filas.append(['Multiclase (E2 frente a E3)', f'{pr_m[0]} · {pr_m[1]}',
              '; '.join(f'{k[0]} · {k[1]}: p = {p:.2e} ({"distinto" if s else "sin diferencia significativa"})' for k, p, s in traza_m),
              ', '.join(f'{k[0]} · {k[1]}' for k in emp_m), f'{el_m[0]} · {el_m[1]}'])
t16 = pd.DataFrame(filas, columns=['Escenario', 'Mayor F1', 'Comparación de McNemar con el de mayor F1',
                                   'Sin diferencia significativa', 'Modelo seleccionado'])
guardar_tabla(t16, 'T16_seleccion_modelo', 'Aplicación del criterio de selección del mejor modelo (apartado 4.5)')
RESUMEN['mejor_modelo'] = {k: list(v) for k, v in MEJOR.items()}
print('Mejor modelo binario   :', MEJOR['binario'])
print('Mejor modelo multiclase:', MEJOR['multiclase'])

# %% [markdown]
# ## Fase 7 · Interpretabilidad con SHAP (apartado 4.4.6)
# Se explican el mejor modelo binario y el mejor modelo multiclase sobre una submuestra estratificada del conjunto de
# prueba. En las figuras se muestran los valores originales de cada característica (sin estandarizar) para que sean
# interpretables.

# %%
def explicador(nombre, modelo, X_fondo):
    if nombre in ('Random Forest', 'XGBoost'):
        return shap.TreeExplainer(modelo), 'TreeExplainer'
    fondo = X_fondo[np.random.default_rng(SEMILLA).choice(len(X_fondo), size=min(100, len(X_fondo)), replace=False)]
    if nombre == 'DNN':
        f = lambda X: modelo.predict(X, batch_size=8192, verbose=0)
    else:
        f = modelo.decision_function
    return shap.Explainer(f, shap.maskers.Independent(fondo), algorithm='permutation'), 'PermutationExplainer'

def calcular_shap(esc, nombre, X, X_original):
    modelo = MODELOS_ENTRENADOS[(esc, nombre)]
    exp, tipo = explicador(nombre, modelo, conj['ent']['X'] if ESCENARIOS[esc]['tarea'] == 'binaria' else conj['ent']['X_m'])
    n = len(X)
    prueba = min(100, n)
    t0 = time.perf_counter()
    sv0 = exp(X[:prueba], check_additivity=False) if tipo == 'TreeExplainer' else exp(X[:prueba])
    t_muestra = (time.perf_counter() - t0) / prueba
    max_n = int(CFG['SHAP_MAX_MINUTOS'] * 60 / max(t_muestra, 1e-6))
    reducido = False
    if max_n < n:
        print(f'   SHAP tardaría ~{t_muestra * n / 60:.0f} min; se reduce la muestra a {max_n} flujos')
        n = max(max_n, prueba)
        reducido = True
    resto = (exp(X[prueba:n], check_additivity=False) if tipo == 'TreeExplainer' else exp(X[prueba:n])) if n > prueba else None
    valores = sv0.values if resto is None else np.concatenate([sv0.values, resto.values])
    base = sv0.base_values if resto is None else np.concatenate([np.asarray(sv0.base_values), np.asarray(resto.base_values)])
    t_total = time.perf_counter() - t0
    return valores, base, n, tipo, t_total, reducido

def explicacion_clase(valores, base, X_orig, k=None):
    """Objeto Explanation para una clase (k) o para la salida única del modelo binario."""
    v = valores if valores.ndim == 2 else valores[:, :, k]
    b = np.asarray(base)
    b = b if b.ndim == 1 else b[:, k]
    return shap.Explanation(values=v, base_values=b, data=X_orig, feature_names=CARACT_SEL)

def muestra_shap(idx_pru_esc, y_estrato):
    n = min(CFG['SHAP_N'], len(idx_pru_esc))
    pos = submuestra(y_estrato, n=n)
    return pos

SHAP_RES = {}
for clave_mejor in ('binario', 'multiclase'):
    esc, nombre = MEJOR[clave_mejor]
    d = datos_escenario(esc)
    estr = cat[d['idx_pru']]
    pos = muestra_shap(d['idx_pru'], estr)
    X_s = d['X_pru'][pos]
    X_o = datos.loc[d['idx_pru'][pos], CARACT_SEL].to_numpy(np.float64)
    print(f'\nSHAP para el mejor modelo {clave_mejor}: {esc} · {nombre} ...')
    valores, base, n_usado, tipo, t_shap, reducido = calcular_shap(esc, nombre, X_s, X_o)
    pos, X_s, X_o = pos[:n_usado], X_s[:n_usado], X_o[:n_usado]
    SHAP_RES[clave_mejor] = dict(esc=esc, nombre=nombre, valores=valores, base=base, pos=pos, X=X_s, X_orig=X_o,
                                 tipo=tipo, n=n_usado, t=t_shap, reducido=reducido)
    print(f'   {tipo} · {n_usado} flujos · {t_shap:.1f} s')
RESUMEN['shap'] = {k: dict(modelo=f"{v['esc']} · {v['nombre']}", explicador=v['tipo'], n_flujos=int(v['n']),
                           tiempo_s=float(v['t'])) for k, v in SHAP_RES.items()}

# %%
# --- SHAP global del mejor modelo binario ---
sb = SHAP_RES['binario']
k_pos = sb['valores'].shape[2] - 1 if sb['valores'].ndim == 3 else None   # si hay varias salidas, la última es "ataque"
exp_bin = explicacion_clase(sb['valores'], sb['base'], sb['X_orig'], k_pos)
imp = np.abs(exp_bin.values).mean(axis=0)
orden = np.argsort(-imp)
signo = [np.corrcoef(exp_bin.data[:, j], exp_bin.values[:, j])[0, 1] if np.std(exp_bin.data[:, j]) > 0 else np.nan
         for j in range(len(CARACT_SEL))]
t17 = pd.DataFrame({'Posición': np.arange(1, len(orden) + 1), 'Característica': np.array(CARACT_SEL)[orden],
                    'Media |SHAP|': imp[orden], 'Correlación valor-SHAP': np.array(signo)[orden]})
guardar_tabla(t17.head(20), 'T17_shap_ranking_binario',
              f'Importancia SHAP global del mejor modelo binario ({sb["esc"]} · {sb["nombre"]})')
RESUMEN['shap_top10_binario'] = t17['Característica'].head(10).tolist()
pos_port = int(t17.loc[t17['Característica'] == 'Destination Port', 'Posición'].iloc[0]) if 'Destination Port' in CARACT_SEL else None
RESUMEN['posicion_destination_port_binario'] = pos_port
print('Posición de Destination Port en la importancia SHAP:', pos_port)

plt.figure()
shap.plots.bar(exp_bin, max_display=15, show=False)
fig = plt.gcf(); fig.set_size_inches(6.3, 4.6); traducir_shap(fig)
guardar_figura(fig, 'F10_shap_binario_barras', f'Importancia global SHAP (media del valor absoluto) del mejor modelo binario ({sb["nombre"]})')
plt.figure()
shap.plots.beeswarm(exp_bin, max_display=15, show=False)
fig = plt.gcf(); fig.set_size_inches(6.6, 5.0); traducir_shap(fig)
guardar_figura(fig, 'F11_shap_binario_beeswarm', f'Gráfico de resumen SHAP (beeswarm) del mejor modelo binario ({sb["nombre"]})')

# %%
# --- SHAP del mejor modelo multiclase: importancia por categoría ---
sm_ = SHAP_RES['multiclase']
filas = []
EXP_CLASE = {c: explicacion_clase(sm_['valores'], sm_['base'], sm_['X_orig'], i) for i, c in enumerate(CLASES_MULTI)}
fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.4))
for i, (ax, c) in enumerate(zip(axes.ravel(), CLASES_MULTI)):
    e = EXP_CLASE[c]
    imp_c = np.abs(e.values).mean(axis=0)
    o = np.argsort(-imp_c)[:10]
    for r, j in enumerate(o, 1):
        filas.append([c, r, CARACT_SEL[j], imp_c[j]])
    ax.barh(np.arange(10), imp_c[o][::-1], color=AZUL, height=0.7)
    ax.set_yticks(np.arange(10), [CARACT_SEL[j] for j in o][::-1], fontsize=6.3)
    ax.set_title(c, fontweight='bold')
    ax.tick_params(axis='x', labelsize=6.5)
    ax.grid(axis='x')
    ax.set_axisbelow(True)
    formato_es(ax, 'x')
axes.ravel()[-1].axis('off')
fig.supxlabel('Media del valor absoluto de SHAP', fontsize=8.5)
fig.tight_layout()
guardar_figura(fig, 'F12_shap_multiclase_por_categoria',
               f'Las 10 características más importantes según SHAP para cada categoría ({sm_["esc"]} · {sm_["nombre"]})')
for c in CLASES_MULTI[1:]:
    plt.figure()
    shap.plots.beeswarm(EXP_CLASE[c], max_display=12, show=False)
    f2 = plt.gcf(); f2.set_size_inches(6.4, 4.4); traducir_shap(f2)
    guardar_figura(f2, f'F13_shap_multiclase_beeswarm_{c}', f'Gráfico de resumen SHAP para la categoría {c} ({sm_["esc"]} · {sm_["nombre"]})')
t18 = pd.DataFrame(filas, columns=['Categoría', 'Posición', 'Característica', 'Media |SHAP|'])
guardar_tabla(t18, 'T18_shap_ranking_multiclase', f'Importancia SHAP por categoría del mejor modelo multiclase ({sm_["esc"]} · {sm_["nombre"]})')

# %%
# --- Explicaciones locales: un ejemplo representativo de cada ataque y falsos positivos (modelo binario) ---
esc_b, nom_b = MEJOR['binario']
r_b = RES[(esc_b, nom_b)]
pos_b = sb['pos']
cat_s = cat[conj['pru']['idx']][pos_b]
y_s, pred_s = r_b['y'][pos_b], r_b['pred'][pos_b]
punt_s = r_b['punt'][pos_b]
ejemplos = []
for c in ['DoS', 'DDoS', 'PortScan', 'Bot', 'Heartbleed']:
    cand = np.where((cat_s == c) & (pred_s == 1))[0]
    if len(cand):
        mediana = np.median(punt_s[cand])
        ejemplos.append((c, cand[np.argmin(np.abs(punt_s[cand] - mediana))]))
for etiqueta, j in ejemplos:
    plt.figure()
    shap.plots.waterfall(exp_bin[j], max_display=10, show=False)
    fig = plt.gcf(); fig.set_size_inches(6.4, 4.2); traducir_shap(fig)
    guardar_figura(fig, f'F14_shap_local_{etiqueta}', f'Explicación SHAP local de un flujo de {etiqueta} detectado por el mejor modelo binario')

# Falsos positivos: se buscan en todo el conjunto de prueba (los dos con mayor puntuación de ataque)
fp_todos = np.where((r_b['y'] == 0) & (r_b['pred'] == 1))[0]
RESUMEN['falsos_positivos_modelo_binario'] = int(len(fp_todos))
if len(fp_todos) == 0:
    print('El mejor modelo binario no tiene falsos positivos en el conjunto de prueba.')
else:
    elegidos = fp_todos[np.argsort(-np.asarray(r_b['punt'])[fp_todos])][:2]
    exp_fp_obj, tipo_fp = explicador(nom_b, MODELOS_ENTRENADOS[(esc_b, nom_b)], conj['ent']['X'])
    X_fp = conj['pru']['X'][elegidos]
    sv_fp = exp_fp_obj(X_fp, check_additivity=False) if tipo_fp == 'TreeExplainer' else exp_fp_obj(X_fp)
    k_fp = sv_fp.values.shape[2] - 1 if sv_fp.values.ndim == 3 else None
    exp_fp = explicacion_clase(sv_fp.values, sv_fp.base_values,
                               datos.loc[conj['pru']['idx'][elegidos], CARACT_SEL].to_numpy(np.float64), k_fp)
    for i in range(len(elegidos)):
        plt.figure()
        shap.plots.waterfall(exp_fp[i], max_display=10, show=False)
        fig = plt.gcf(); fig.set_size_inches(6.4, 4.2); traducir_shap(fig)
        guardar_figura(fig, f'F14_shap_local_falso_positivo_{i + 1}',
                       'Explicación SHAP local de un falso positivo (tráfico benigno clasificado como ataque) del mejor modelo binario')

# %% [markdown]
# ## Fase 8 · Arquitectura de referencia y simulación del prototipo (apartado 4.4.7)

# %%
# --- Esquema de la arquitectura de referencia (mismo estilo que la Ilustración 7) ---
C_RED, C_DET, C_SEG, BORDE = '#dce6f2', '#2f5496', '#8eaadb', '#1f3864'
modulos = [
    ('1. Captura', 'Réplica de puertos\n(port mirroring)\no dispositivo TAP', C_RED, TINTA),
    ('2. Flujos', 'CICFlowMeter agrupa\nlos paquetes en flujos\ny calcula sus\ncaracterísticas', C_RED, TINTA),
    ('3. Preprocesamiento', 'Selección de\ncaracterísticas y\nestandarización', C_DET, 'white'),
    ('4. Inferencia', 'El modelo clasifica\ncada flujo como\nbenigno o ataque', C_DET, 'white'),
    ('5. Explicación', 'Valores SHAP de\nlas características\nde cada alerta', C_DET, 'white'),
    ('6. SIEM / panel', 'El analista recibe\nla alerta junto con\nsu explicación', C_SEG, TINTA),
]
fig, ax = plt.subplots(figsize=(7.0, 4.3))
ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis('off')
W, H = 27, 17
xs = [2, 36.5, 71]
pos = [(xs[0], 36), (xs[1], 36), (xs[2], 36), (xs[2], 9), (xs[1], 9), (xs[0], 9)]   # recorrido en "serpiente"
for (tit, cuerpo, fc, tc), (x, y) in zip(modulos, pos):
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle='round,pad=0.4,rounding_size=1.6', fc=fc, ec=BORDE, lw=1.2))
    ax.text(x + W / 2, y + H - 2.6, tit, ha='center', va='center', fontsize=8.6, fontweight='bold', color=tc)
    ax.plot([x + 2, x + W - 2], [y + H - 5] * 2, color=tc, lw=0.6, alpha=0.6)
    ax.text(x + W / 2, y + (H - 5) / 2, cuerpo, ha='center', va='center', fontsize=7.2, color=tc, linespacing=1.3)
def flecha(p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=13, lw=1.3, color=BORDE))
flecha((xs[0] + W + 0.6, 36 + H / 2), (xs[1] - 0.6, 36 + H / 2))
flecha((xs[1] + W + 0.6, 36 + H / 2), (xs[2] - 0.6, 36 + H / 2))
flecha((xs[2] + W / 2, 36 - 0.6), (xs[2] + W / 2, 9 + H + 0.6))
flecha((xs[2] - 0.6, 9 + H / 2), (xs[1] + W + 0.6, 9 + H / 2))
flecha((xs[1] - 0.6, 9 + H / 2), (xs[0] + W + 0.6, 9 + H / 2))
ax.text(50, 2.5, 'Los módulos 3 a 5 cargan los artefactos guardados con joblib: lista de características, escalador y modelo',
        ha='center', va='center', fontsize=7, color=TINTA_2, style='italic')
for lx, fc, lab in ((4, C_RED, 'Red corporativa'), (31, C_DET, 'Módulo de detección (este trabajo)'),
                    (73, C_SEG, 'Equipo de seguridad')):
    ax.add_patch(FancyBboxPatch((lx, 58), 2.4, 2.2, boxstyle='round,pad=0.1', fc=fc, ec=BORDE, lw=0.8))
    ax.text(lx + 3.6, 59.1, lab, fontsize=7.8, va='center')
guardar_figura(fig, 'F15_arquitectura_referencia', 'Arquitectura de referencia propuesta para integrar el modelo en la monitorización de la red')

# %%
# --- Simulación: el conjunto de prueba llega al módulo de inferencia en lotes de 1.000 flujos ---
class ModuloInferencia:
    """Reproduce los módulos 3 a 5 de la arquitectura a partir de los artefactos guardados en disco."""
    def __init__(self, ruta_pre, ruta_modelo, nombre, tarea):
        pre = joblib.load(ruta_pre)
        self.caract, self.escalador = pre['caracteristicas'], pre['escalador']
        self.nombre, self.tarea = nombre, tarea
        self.modelo = keras.models.load_model(ruta_modelo) if nombre == 'DNN' else joblib.load(ruta_modelo)
        if hasattr(self.modelo, 'n_jobs'):
            self.modelo.set_params(n_jobs=CFG['N_JOBS'])
        self.explicador = shap.TreeExplainer(self.modelo) if nombre in ('Random Forest', 'XGBoost') else None

    def procesar(self, lote):
        X = self.escalador.transform(lote[self.caract].to_numpy(np.float32)).astype(np.float32)
        pred = predecir_dnn(self.modelo, X, self.tarea) if self.nombre == 'DNN' else self.modelo.predict(X)
        return X, pred

    def explicar(self, X_alertas):
        if self.explicador is None or len(X_alertas) == 0:
            return None
        return self.explicador(X_alertas, check_additivity=False)

def simular(clave_mejor, lotes_explicacion=10, alertas_por_lote=10):
    esc, nombre = MEJOR[clave_mejor]
    tarea = ESCENARIOS[esc]['tarea']
    mod = ModuloInferencia(DIR_MOD / 'preprocesamiento.joblib', ruta_modelo(esc, nombre), nombre, tarea)
    trafico = datos.loc[datos_escenario(esc)['idx_pru'], CARACT_SEL].reset_index(drop=True)
    L = CFG['LOTE_PROTOTIPO']
    if len(trafico) < 30 * L:                 # conjunto de prueba muy pequeño: se repite hasta tener 30 lotes
        trafico = pd.concat([trafico] * math.ceil(30 * L / max(len(trafico), 1)), ignore_index=True)
    n_lotes = len(trafico) // L
    latencias, alertas, t_expl, n_expl = [], [], 0.0, 0
    for i in range(n_lotes):
        lote = trafico.iloc[i * L:(i + 1) * L]
        t0 = time.perf_counter()
        X, pred = mod.procesar(lote)
        dt = time.perf_counter() - t0
        es_alerta = (pred != 0)
        if i >= 3:                                            # los tres primeros lotes son de calentamiento
            latencias.append(dt * 1000)
            alertas.append(int(es_alerta.sum()))
        if 3 <= i < 3 + lotes_explicacion and es_alerta.any():
            X_al = X[es_alerta][:alertas_por_lote]
            t0 = time.perf_counter()
            mod.explicar(X_al)
            t_expl += time.perf_counter() - t0
            n_expl += len(X_al)
    lat = np.array(latencias)
    return dict(clave=clave_mejor, esc=esc, nombre=nombre, lotes=len(lat), lat=lat, alertas=np.array(alertas),
                expl_ms=(t_expl / n_expl * 1000) if n_expl else np.nan)

SIM = [simular('binario'), simular('multiclase')]
filas = []
for s in SIM:
    filas.append([f'{s["esc"]} · {s["nombre"]} ({s["clave"]})', s['lotes'], float(np.mean(s['lat'])),
                  float(np.percentile(s['lat'], 50)), float(np.percentile(s['lat'], 95)), float(np.percentile(s['lat'], 99)),
                  float(CFG['LOTE_PROTOTIPO'] / (np.mean(s['lat']) / 1000)), float(np.mean(s['alertas'])), s['expl_ms']])
t19 = pd.DataFrame(filas, columns=['Modelo', 'Lotes medidos', 'Latencia media (ms/lote)', 'Mediana (ms)', 'Percentil 95 (ms)',
                                   'Percentil 99 (ms)', 'Flujos por segundo', 'Alertas por lote (media)',
                                   'Explicación SHAP por alerta (ms)'])
guardar_tabla(t19, 'T19_prototipo_latencia', f'Simulación del prototipo en lotes de {CFG["LOTE_PROTOTIPO"]} flujos')
RESUMEN['prototipo'] = t19.to_dict(orient='records')

fig, ax = plt.subplots(figsize=(6.3, 2.6))
s = SIM[0]
ax.hist(s['lat'], bins=40, color=AZUL, edgecolor='white', linewidth=0.6)
p95 = np.percentile(s['lat'], 95)
ax.axvline(p95, color=TINTA_2, lw=1.2, ls='--')
ax.text(p95, ax.get_ylim()[1] * 0.92, f'  percentil 95 = {coma(p95, 1)} ms', fontsize=7.5, color=TINTA_2, va='top')
ax.set_xlabel(f'Latencia por lote de {CFG["LOTE_PROTOTIPO"]:,} flujos (ms)'.replace(',', '.'))
ax.set_ylabel('Número de lotes')
ax.grid(axis='y')
ax.set_axisbelow(True)
formato_es(ax)
guardar_figura(fig, 'F16_latencia_prototipo', f'Distribución de la latencia por lote del modelo binario seleccionado ({s["nombre"]})')

# %% [markdown]
# ## Exportación final
# Todas las tablas se guardan también en un único Excel (`tablas_resultados.xlsx`), junto con un resumen en JSON, la
# información del entorno de ejecución y las versiones exactas de las librerías (`requirements_usadas.txt`).

# %%
import importlib.metadata as md
try:
    import psutil
    ram_gb = round(psutil.virtual_memory().total / 1024 ** 3, 1)
except Exception:
    ram_gb = None
LIBRERIAS = ['numpy', 'pandas', 'scikit-learn', 'imbalanced-learn', 'xgboost', 'tensorflow', 'keras', 'shap',
             'matplotlib', 'scipy', 'statsmodels', 'joblib', 'pyarrow', 'openpyxl']
versiones = {}
for lib in LIBRERIAS:
    try:
        versiones[lib] = md.version(lib)
    except md.PackageNotFoundError:
        try:
            versiones[lib] = md.version(lib + '-cpu')
        except md.PackageNotFoundError:
            versiones[lib] = None
entorno = dict(python=platform.python_version(), sistema=f'{platform.system()} {platform.release()}',
               procesador=platform.processor(), nucleos_logicos=os.cpu_count(), ram_gb=ram_gb, colab=EN_COLAB,
               gpu_tensorflow=[g.name for g in tf.config.list_physical_devices('GPU')], dispositivo_xgboost=DISPOSITIVO_XGB,
               librerias=versiones, tiempo_total_min=round((time.perf_counter() - T_INICIO_GLOBAL) / 60, 1),
               configuracion={k: (list(v) if isinstance(v, tuple) else v) for k, v in CFG.items()})
(DIR_RES / 'entorno.json').write_text(json.dumps(entorno, ensure_ascii=False, indent=1), encoding='utf-8')
(DIR_RES / 'requirements_usadas.txt').write_text('\n'.join(f'{k}=={v}' for k, v in versiones.items() if v) + '\n', encoding='utf-8')

with pd.ExcelWriter(DIR_RES / 'tablas_resultados.xlsx') as xls:
    indice = pd.DataFrame([(n, desc) for n, (desc, _) in TABLAS.items()], columns=['Hoja', 'Contenido'])
    indice.to_excel(xls, sheet_name='Indice', index=False)
    for n, (desc, df) in TABLAS.items():
        df.to_excel(xls, sheet_name=n[:31], index=False)

RESUMEN['metricas'] = {f'{e} · {n}': dict(precision=float(r['precision']), sensibilidad=float(r['sensibilidad']),
                                          f1=float(r['f1']), auc=float(r['auc']), fpr=float(r['fpr']),
                                          inferencia_ms_1000=float(r['t_inf_ms_1000'])) for (e, n), r in RES.items()}
RESUMEN['tablas'] = {n: d for n, (d, _) in TABLAS.items()}
RESUMEN['figuras'] = dict(FIGURAS)
RESUMEN['entorno'] = entorno
(DIR_RES / 'resumen.json').write_text(json.dumps(RESUMEN, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
print(f'\nListo. {len(TABLAS)} tablas y {len(FIGURAS)} figuras en {DIR_RES}')
print(f'Tiempo total: {entorno["tiempo_total_min"]} min')

# %% [markdown]
# ## Comprobaciones para mantener el Capítulo 4 coherente con lo ejecutado

# %%
avisos = []
if MODO_PRUEBA:
    avisos.append('MODO PRUEBA: estos resultados no valen para el TFM. Cambia a MODO_PRUEBA = False para la ejecución real.')
if (t01['¿Coincide?'] == 'NO').any():
    avisos.append('Tabla 3: hay recuentos distintos a los de los archivos descargados (ver T01) y hay que corregir la tabla.')
if len(CARACT) != 77:
    avisos.append(f'Tras eliminar la columna duplicada quedan {len(CARACT)} características; los apartados 4.1 y 4.3.2 hablan de 77.')
if not MODO_PRUEBA and INFO[('E1', 'SVM')]['n_combinaciones'] != 12:
    avisos.append(f'SVM: se han probado {INFO[("E1", "SVM")]["n_combinaciones"]} combinaciones; el apartado 4.4.5 habla de 12.')
if not MODO_PRUEBA and INFO[('E1', 'DNN')]['n_combinaciones'] != 24:
    avisos.append(f'DNN: se han probado {INFO[("E1", "DNN")]["n_combinaciones"]} configuraciones; el apartado 4.4.5 habla de 24.')
if SHAP_RES['binario']['reducido'] or SHAP_RES['multiclase']['reducido']:
    avisos.append(f'SHAP se calculó sobre {SHAP_RES["binario"]["n"]} / {SHAP_RES["multiclase"]["n"]} flujos en lugar de '
                  f'{CFG["SHAP_N"]} por tiempo de cálculo: indícalo en el Capítulo 5 (el apartado 4.4.6 habla de 5.000).')
if SHAP_RES['binario']['tipo'] != 'TreeExplainer' or SHAP_RES['multiclase']['tipo'] != 'TreeExplainer':
    avisos.append('Algún modelo seleccionado no es de árboles: SHAP se calculó con PermutationExplainer, como prevé el apartado 4.4.6. '
                  'Menciónalo al presentar la interpretabilidad en el Capítulo 5.')
gpu = ', '.join(entorno['gpu_tensorflow']) if entorno['gpu_tensorflow'] else 'sin GPU para TensorFlow'
avisos.append(f'Equipo usado, para completar el apartado 4.3.3: {entorno["sistema"]}, procesador {entorno["procesador"] or "(ver entorno.json)"}, '
              f'{entorno["nucleos_logicos"]} núcleos lógicos, {entorno["ram_gb"]} GB de RAM, Python {entorno["python"]}, {gpu}. '
              f'Tiempo total de la ejecución: {entorno["tiempo_total_min"]} min.')
texto_avisos = 'Puntos del Capítulo 4 que conviene revisar o completar:\n - ' + '\n - '.join(avisos)
(DIR_RES / 'avisos_capitulo4.txt').write_text(texto_avisos + '\n', encoding='utf-8')
print(texto_avisos)
