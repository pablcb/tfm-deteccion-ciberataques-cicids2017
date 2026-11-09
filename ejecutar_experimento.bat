@echo off
setlocal
cd /d "%~dp0"
title Experimento del TFM - CICIDS2017

echo ============================================================
echo   Experimento del TFM - CICIDS2017
echo ============================================================
echo.

rem ---------------------------------------------------------------
rem  1. Comprobar que estan los cinco CSV de CICIDS2017 en datos\
rem ---------------------------------------------------------------
set FALTAN=
for %%F in (Monday-WorkingHours.pcap_ISCX.csv Wednesday-workingHours.pcap_ISCX.csv Friday-WorkingHours-Morning.pcap_ISCX.csv Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv) do (
    dir /s /b "datos\%%F" >nul 2>&1 || set FALTAN=1
)
if defined FALTAN (
    echo [!] Faltan datos. Sigue las instrucciones de datos\LEEME.txt y vuelve a abrir este archivo.
    pause
    exit /b 1
)
echo Datos encontrados.

rem ---------------------------------------------------------------
rem  2. Entorno de Python 3.12 en una carpeta sin tildes, porque
rem     TensorFlow puede fallar si la ruta tiene caracteres especiales
rem ---------------------------------------------------------------
set "VENV=%USERPROFILE%\tfm_cicids2017_venv"
set "PY=%VENV%\Scripts\python.exe"
if not exist "%VENV%\instalado.ok" (
    if not exist "%PY%" (
        echo Creando el entorno de Python 3.12 en %VENV% ...
        py -3.12 -m venv "%VENV%"
        if errorlevel 1 (
            echo [!] No se ha encontrado Python 3.12. Instalalo desde https://www.python.org/downloads/
            echo     marcando la opcion del lanzador py, y vuelve a abrir este archivo.
            pause
            exit /b 1
        )
    )
    echo Instalando las librerias. La primera vez tarda unos minutos ...
    "%PY%" -m pip install --upgrade pip
    "%PY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [!] Ha fallado la instalacion de las librerias. Revisa los mensajes de arriba.
        pause
        exit /b 1
    )
    echo ok > "%VENV%\instalado.ok"
)
echo Entorno listo.

set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
set MPLBACKEND=Agg
set TF_CPP_MIN_LOG_LEVEL=2

rem ---------------------------------------------------------------
rem  3. Prueba rapida con el 2 por ciento de los datos
rem ---------------------------------------------------------------
echo.
echo [1/2] Prueba rapida con el 2 por ciento de los datos. Tarda unos minutos ...
set TFM_MODO_PRUEBA=1
"%PY%" TFM_experimento_CICIDS2017.py > registro_prueba.txt 2>&1
if errorlevel 1 (
    echo [!] La prueba ha fallado. Mira el final del archivo registro_prueba.txt
    pause
    exit /b 1
)
echo       Prueba correcta. Sus resultados estan en resultados_prueba y NO valen para el TFM.

rem ---------------------------------------------------------------
rem  4. Ejecucion real
rem ---------------------------------------------------------------
echo.
echo [2/2] Ejecucion real. Puede tardar varias horas: no cierres esta ventana
echo       ni dejes que el PC se suspenda. El progreso se va guardando en
echo       registro_ejecucion_real.txt
echo       Inicio: %DATE% %TIME%
set TFM_MODO_PRUEBA=0
"%PY%" TFM_experimento_CICIDS2017.py > registro_ejecucion_real.txt 2>&1
if errorlevel 1 (
    echo [!] La ejecucion se ha detenido. Mira el final de registro_ejecucion_real.txt
    echo     Si vuelves a abrir este archivo, se reutiliza todo lo que ya estaba calculado.
    pause
    exit /b 1
)
echo       Fin: %DATE% %TIME%
echo.
echo ============================================================
echo   Listo. Los resultados estan en la carpeta resultados
echo   y los puntos del capitulo 4 que conviene revisar, en
echo   resultados\avisos_capitulo4.txt
echo ============================================================
pause
