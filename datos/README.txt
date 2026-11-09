DATOS DEL EXPERIMENTO (CICIDS2017, versión MachineLearningCVE)
================================================================

1. Entra en la página oficial del Canadian Institute for Cybersecurity:
   https://www.unb.ca/cic/datasets/ids-2017.html
   Rellena el formulario de descarga y baja el archivo MachineLearningCSV.zip.

2. Descomprímelo. Dentro hay una carpeta MachineLearningCVE con ocho CSV.

3. Copia en esta carpeta (datos/) los cinco que usa el trabajo. Pueden quedar dentro de la subcarpeta
   MachineLearningCVE; el notebook los busca también en subcarpetas:

     Monday-WorkingHours.pcap_ISCX.csv
     Wednesday-workingHours.pcap_ISCX.csv
     Friday-WorkingHours-Morning.pcap_ISCX.csv
     Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
     Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv

   Los otros tres (martes y jueves) no se usan.

4. No modifiques los CSV ni los abras y guardes con Excel: el notebook hace toda la limpieza y comprueba que los
   recuentos coinciden con la Tabla 3 del Capítulo 4.

Estos archivos pesan varios cientos de MB: no los adjuntes al TFM. Basta con citar la fuente
(Sharafaldin et al., 2018) e indicar de dónde se descargan.
