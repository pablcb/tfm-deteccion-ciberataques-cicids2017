# Detección de ciberataques mediante técnicas de aprendizaje automático aplicadas al análisis de tráfico de red

Código y resultados del Trabajo Fin de Máster de **Pablo Claudio Baragaño** · Máster Universitario en Inteligencia
Artificial · Universidad Alfonso X el Sabio · Tutor: Edmundo Huertas Cejudo · Noviembre de 2026.

Este repositorio contiene el experimento completo del TFM: el notebook que implementa el diseño experimental del
Capítulo 4 de la memoria y todas las tablas y figuras del Capítulo 5, obtenidas en una única ejecución completa
realizada el 31 de octubre de 2026.

## Qué hace el experimento

- **Datos:** CICIDS2017 (Sharafaldin et al., 2018), versión MachineLearningCVE, jornadas del lunes, el miércoles y el
  viernes (1.925.866 flujos).
- **Limpieza:** eliminación de valores infinitos o nulos, de 222.346 registros duplicados y de valores negativos no
  válidos, hasta 1.699.472 flujos.
- **Preprocesamiento:** partición estratificada 70/15/15, eliminación de características constantes y muy
  correlacionadas (de 77 a 45) y estandarización ajustada solo con el conjunto de entrenamiento.
- **Modelos:** Random Forest, XGBoost, SVM y una red neuronal profunda (DNN) en tres escenarios: E1, clasificación
  binaria; E2, multiclase; y E3, multiclase con SMOTE.
- **Evaluación:** métricas por categoría y media macro, tasa de falsos positivos, prueba de McNemar con corrección de
  Bonferroni, interpretación con SHAP y simulación de una arquitectura de referencia en lotes de 1.000 flujos.

## Resultados principales

| Escenario | Mejor modelo | F1-score | Tasa de falsos positivos |
|---|---|---|---|
| E1 · binario | XGBoost | 0,9986 | 0,00027 |
| E2 · multiclase | XGBoost | 0,9626 (macro) | 0,00036 |
| E3 · multiclase con SMOTE | Random Forest | 0,9618 (macro) | 0,00074 |

El modelo binario seleccionado clasifica unos 728.500 flujos por segundo en un ordenador personal sin GPU. Todas las
tablas están en `resultados/tablas` (y juntas en `resultados/tablas_resultados.xlsx`) y las figuras, en
`resultados/figuras`.

## Estructura

```
├── README.md
├── TFM_experimento_CICIDS2017.ipynb      notebook del experimento
├── TFM_experimento_CICIDS2017.py         el mismo código en formato script
├── requirements.txt                      versiones exactas de las bibliotecas
├── ejecutar_experimento.bat              ejecución completa con doble clic (Windows)
├── registro_ejecucion_real.txt           registro de la ejecución del 31/10/2026
├── datos/LEEME.txt                       cómo descargar los datos (no se incluyen)
├── pruebas/generar_datos_sinteticos.py   datos falsos para probar el código sin el dataset
└── resultados/
    ├── tablas/                           tablas T01 a T19 en CSV
    ├── tablas_resultados.xlsx            todas las tablas en un único archivo
    ├── figuras/                          figuras F01 a F16 en PNG a 300 ppp
    ├── resumen.json                      cifras clave de la ejecución
    ├── entorno.json                      equipo, versiones y configuración
    └── requirements_usadas.txt           versiones instaladas durante la ejecución
```

## Cómo reproducir el experimento

1. Descargar los datos siguiendo `datos/LEEME.txt` (formulario del Canadian Institute for Cybersecurity) y copiar en
   `datos/` los cinco CSV del lunes, el miércoles y el viernes.
2. En Windows, hacer doble clic en `ejecutar_experimento.bat`. El archivo crea un entorno con Python 3.12 en
   `%USERPROFILE%\tfm_cicids2017_venv`, instala `requirements.txt`, hace una prueba rápida con el 2 % de los datos y
   lanza la ejecución completa.
3. En cualquier sistema con Python 3.12:

   ```
   python -m venv .venv
   .venv\Scripts\activate              (Linux o macOS: source .venv/bin/activate)
   pip install -r requirements.txt
   set TFM_MODO_PRUEBA=0               (Linux o macOS: export TFM_MODO_PRUEBA=0)
   python TFM_experimento_CICIDS2017.py
   ```

   También se puede abrir el notebook en Jupyter y poner `MODO_PRUEBA = False` en la primera celda.

La ejecución completa duró 73,6 minutos en un AMD Ryzen 7 7700X con 32 GB de RAM y sin GPU (Windows 11, Python
3.12.0). La semilla es fija (`random_state = 42`), por lo que las métricas deberían repetirse con las mismas
versiones; los tiempos dependen del equipo.

## Notas

- Los datos de CICIDS2017 no se incluyen: deben descargarse de la fuente oficial y citarse como Sharafaldin et al.
  (2018).
- Tampoco se incluyen los modelos entrenados ni la caché de datos, que se regeneran al ejecutar el notebook.
- En el registro de la ejecución se han sustituido las rutas locales del equipo por rutas relativas.

## Referencia del conjunto de datos

Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward Generating a New Intrusion Detection Dataset and
Intrusion Traffic Characterization. *Proceedings of the 4th International Conference on Information Systems Security
and Privacy (ICISSP)*, 108-116.
