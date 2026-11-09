# Genera CSV SINTÉTICOS con el mismo esquema que MachineLearningCVE (CICIDS2017).
# Solo sirven para comprobar que el notebook se ejecuta sin errores: sus cifras NO significan nada y
# NUNCA deben usarse en el TFM.
#
# Uso (desde la carpeta Codigo_TFM):
#     python pruebas/generar_datos_sinteticos.py
# Crea los CSV en pruebas/base_sintetica/datos/MachineLearningCVE, separados de los datos reales.
# Para probar el notebook con ellos, pon en su primera celda:
#     MODO_PRUEBA = True
#     DIR_BASE = 'pruebas/base_sintetica'
# y los resultados quedarán en pruebas/base_sintetica/resultados_prueba (bórralos cuando acabes).
import numpy as np, pandas as pd, sys
from pathlib import Path

CABECERA = (" Destination Port, Flow Duration, Total Fwd Packets, Total Backward Packets,Total Length of Fwd Packets,"
            " Total Length of Bwd Packets, Fwd Packet Length Max, Fwd Packet Length Min, Fwd Packet Length Mean,"
            " Fwd Packet Length Std,Bwd Packet Length Max, Bwd Packet Length Min, Bwd Packet Length Mean,"
            " Bwd Packet Length Std,Flow Bytes/s, Flow Packets/s, Flow IAT Mean, Flow IAT Std, Flow IAT Max,"
            " Flow IAT Min,Fwd IAT Total, Fwd IAT Mean, Fwd IAT Std, Fwd IAT Max, Fwd IAT Min,Bwd IAT Total,"
            " Bwd IAT Mean, Bwd IAT Std, Bwd IAT Max, Bwd IAT Min,Fwd PSH Flags, Bwd PSH Flags, Fwd URG Flags,"
            " Bwd URG Flags, Fwd Header Length, Bwd Header Length,Fwd Packets/s, Bwd Packets/s, Min Packet Length,"
            " Max Packet Length, Packet Length Mean, Packet Length Std, Packet Length Variance,FIN Flag Count,"
            " SYN Flag Count, RST Flag Count, PSH Flag Count, ACK Flag Count, URG Flag Count, CWE Flag Count,"
            " ECE Flag Count, Down/Up Ratio, Average Packet Size, Avg Fwd Segment Size, Avg Bwd Segment Size,"
            " Fwd Header Length,Fwd Avg Bytes/Bulk, Fwd Avg Packets/Bulk, Fwd Avg Bulk Rate, Bwd Avg Bytes/Bulk,"
            " Bwd Avg Packets/Bulk,Bwd Avg Bulk Rate,Subflow Fwd Packets, Subflow Fwd Bytes, Subflow Bwd Packets,"
            " Subflow Bwd Bytes,Init_Win_bytes_forward, Init_Win_bytes_backward, act_data_pkt_fwd,"
            " min_seg_size_forward,Active Mean, Active Std, Active Max, Active Min,Idle Mean, Idle Std, Idle Max,"
            " Idle Min, Label").split(',')
assert len(CABECERA) == 79, len(CABECERA)

rng = np.random.default_rng(7)
PERFIL = {   # puerto, duración media (log), paquetes fwd, paquetes bwd, tamaño fwd, tamaño bwd, iat (log)
    'BENIGN': ([80, 443, 53, 22, 8080, 123], 11.0, 8, 7, 200, 600, 9.0),
    'DoS Hulk': ([80], 9.5, 6, 5, 350, 3000, 8.0),
    'DoS GoldenEye': ([80], 15.0, 8, 6, 400, 2000, 13.0),
    'DoS slowloris': ([80], 16.5, 5, 2, 230, 0, 15.5),
    'DoS Slowhttptest': ([80], 16.0, 4, 2, 200, 0, 15.0),
    'Heartbleed': ([444], 18.0, 2000, 3000, 30, 8000, 9.0),
    'Bot': ([8080, 80], 10.0, 4, 3, 150, 120, 9.5),
    'PortScan': (list(range(1, 1024)), 3.0, 1, 1, 0, 6, 2.0),
    'DDoS': ([80], 13.0, 4, 4, 20, 1800, 12.0),
}

def generar(etiqueta, n):
    puertos, dur, pf, pb, tf_, tb, iat = PERFIL[etiqueta]
    ruido = 0.9
    D = {}
    D[' Destination Port'] = rng.choice(puertos, n)
    duracion = np.exp(rng.normal(dur, ruido * 1.5, n)).round()
    duracion[rng.random(n) < 0.01] = 0
    D[' Flow Duration'] = duracion
    tfp = np.maximum(1, rng.poisson(pf, n)); tbp = rng.poisson(pb, n)
    D[' Total Fwd Packets'] = tfp; D[' Total Backward Packets'] = tbp
    lf = np.abs(rng.normal(tf_, tf_ * 0.4 + 5, n)); lb = np.abs(rng.normal(tb, tb * 0.4 + 5, n))
    D['Total Length of Fwd Packets'] = (lf * tfp).round(); D[' Total Length of Bwd Packets'] = (lb * tbp).round()
    D[' Fwd Packet Length Max'] = (lf * 1.5).round(); D[' Fwd Packet Length Min'] = (lf * 0.2).round()
    D[' Fwd Packet Length Mean'] = lf; D[' Fwd Packet Length Std'] = lf * 0.3
    D['Bwd Packet Length Max'] = (lb * 1.4).round(); D[' Bwd Packet Length Min'] = (lb * 0.1).round()
    D[' Bwd Packet Length Mean'] = lb; D[' Bwd Packet Length Std'] = lb * 0.35
    with np.errstate(divide='ignore', invalid='ignore'):
        bytes_s = (D['Total Length of Fwd Packets'] + D[' Total Length of Bwd Packets']) / (duracion / 1e6)
        paq_s = (tfp + tbp) / (duracion / 1e6)
    D['Flow Bytes/s'] = bytes_s; D[' Flow Packets/s'] = paq_s
    iatm = np.exp(rng.normal(iat, ruido, n))
    D[' Flow IAT Mean'] = iatm; D[' Flow IAT Std'] = iatm * 0.5; D[' Flow IAT Max'] = iatm * 2; D[' Flow IAT Min'] = iatm * 0.1
    D['Fwd IAT Total'] = iatm * tfp; D[' Fwd IAT Mean'] = iatm * 1.1; D[' Fwd IAT Std'] = iatm * 0.4
    D[' Fwd IAT Max'] = iatm * 2.2; D[' Fwd IAT Min'] = iatm * 0.05
    D['Bwd IAT Total'] = iatm * tbp; D[' Bwd IAT Mean'] = iatm * 0.9; D[' Bwd IAT Std'] = iatm * 0.3
    D[' Bwd IAT Max'] = iatm * 1.8; D[' Bwd IAT Min'] = iatm * 0.02
    D['Fwd PSH Flags'] = rng.integers(0, 2, n); D[' Bwd PSH Flags'] = 0; D[' Fwd URG Flags'] = 0; D[' Bwd URG Flags'] = 0
    hdr = tfp * rng.choice([20, 32, 40], n); D[' Fwd Header Length'] = hdr; D[' Bwd Header Length'] = tbp * 32
    with np.errstate(divide='ignore', invalid='ignore'):
        D['Fwd Packets/s'] = tfp / (duracion / 1e6 + 1e-9); D[' Bwd Packets/s'] = tbp / (duracion / 1e6 + 1e-9)
    D[' Min Packet Length'] = np.minimum(D[' Fwd Packet Length Min'], D[' Bwd Packet Length Min'])
    D[' Max Packet Length'] = np.maximum(D[' Fwd Packet Length Max'], D['Bwd Packet Length Max'])
    D[' Packet Length Mean'] = (lf + lb) / 2; D[' Packet Length Std'] = (lf + lb) * 0.2
    D[' Packet Length Variance'] = D[' Packet Length Std'] ** 2
    for f in ['FIN Flag Count', ' SYN Flag Count', ' RST Flag Count', ' PSH Flag Count', ' ACK Flag Count', ' URG Flag Count']:
        D[f] = rng.integers(0, 2, n)
    D[' CWE Flag Count'] = 0; D[' ECE Flag Count'] = rng.integers(0, 2, n) * (rng.random(n) < 0.01)
    D[' Down/Up Ratio'] = (tbp / tfp).round()
    D[' Average Packet Size'] = D[' Packet Length Mean'] * 1.1; D[' Avg Fwd Segment Size'] = lf; D[' Avg Bwd Segment Size'] = lb
    D[' Fwd Header Length.1'] = hdr
    for f in ['Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk', ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk',
              ' Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate']:
        D[f] = 0
    D['Subflow Fwd Packets'] = tfp; D[' Subflow Fwd Bytes'] = D['Total Length of Fwd Packets']
    D[' Subflow Bwd Packets'] = tbp; D[' Subflow Bwd Bytes'] = D[' Total Length of Bwd Packets']
    D['Init_Win_bytes_forward'] = np.where(rng.random(n) < 0.3, -1, rng.choice([8192, 29200, 65535, 251], n))
    D[' Init_Win_bytes_backward'] = np.where(rng.random(n) < 0.4, -1, rng.choice([0, 235, 28960], n))
    D[' act_data_pkt_fwd'] = np.maximum(0, tfp - 1); D[' min_seg_size_forward'] = rng.choice([20, 32], n)
    act = np.exp(rng.normal(8, 1, n)) * (rng.random(n) < 0.3)
    D['Active Mean'] = act; D[' Active Std'] = act * 0.2; D[' Active Max'] = act * 1.3; D[' Active Min'] = act * 0.7
    idle = np.exp(rng.normal(15, 1, n)) * (rng.random(n) < 0.3)
    D['Idle Mean'] = idle; D[' Idle Std'] = idle * 0.1; D[' Idle Max'] = idle * 1.2; D[' Idle Min'] = idle * 0.8
    D[' Label'] = etiqueta
    df = pd.DataFrame(D)
    return df

ARCHIVOS = {
    'Monday-WorkingHours.pcap_ISCX.csv': {'BENIGN': 6000},
    'Wednesday-workingHours.pcap_ISCX.csv': {'BENIGN': 4400, 'DoS Hulk': 2300, 'DoS GoldenEye': 300,
                                             'DoS slowloris': 200, 'DoS Slowhttptest': 200, 'Heartbleed': 11},
    'Friday-WorkingHours-Morning.pcap_ISCX.csv': {'BENIGN': 1900, 'Bot': 120},
    'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv': {'BENIGN': 1300, 'PortScan': 1600},
    'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv': {'BENIGN': 1000, 'DDoS': 1300},
}
base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / 'base_sintetica'
if (base / 'resultados').exists():
    sys.exit(f'{base} contiene una carpeta resultados/ (¿es la carpeta del experimento real?). Elige otra carpeta.')
destino = base / 'datos' / 'MachineLearningCVE'
destino.mkdir(parents=True, exist_ok=True)
for archivo, comp in ARCHIVOS.items():
    df = pd.concat([generar(e, n) for e, n in comp.items()], ignore_index=True)
    df = pd.concat([df, df.sample(frac=0.05, random_state=1)], ignore_index=True)        # duplicados
    neg = df.sample(frac=0.003, random_state=2).index
    df.loc[neg, ' Flow Duration'] = -1                                                    # negativos no válidos
    nul = df.sample(frac=0.001, random_state=3).index
    df.loc[nul, 'Flow Bytes/s'] = np.nan                                                  # nulos
    # 2 % de etiquetas cambiadas para que los modelos cometan algún error
    ruido = df.sample(frac=0.02, random_state=4).index
    df.loc[ruido, ' Label'] = rng.choice(list(comp), len(ruido))
    df = df.sample(frac=1, random_state=5).reset_index(drop=True)
    orden, visto = [], set()
    for c in CABECERA:                                  # mismo orden de columnas que el original
        orden.append(c if c not in visto else c + '.1'); visto.add(c)
    df = df[orden]
    for c in ['Flow Bytes/s', ' Flow Packets/s']:
        df[c] = df[c].astype(object).where(~np.isinf(df[c].astype(float)), 'Infinity')
    df.columns = CABECERA                               # misma cabecera que el original (columna duplicada incluida)
    df.to_csv(destino / archivo, index=False, na_rep='NaN')
    print(archivo, df.shape)
