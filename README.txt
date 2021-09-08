Guía de instalación y uso del proyecto

1- Descarga de Python 

El primer paso para la instalación es poseer una versión relativamente reciente de Python, Python es necesario para la ejecución del programa y también para los pasos posteriores de la instalación.
Durante la creación del proyecto se ha usado la versión 3.8.5.

Enlace a la página oficial de Python:
https://www.python.org/downloads/

2- Descarga de requisitos

Una vez Python esté descargado y operativo, abrimos la CMD y nos dirigimos en ella a la carpeta donde se encuentra el script "detector_trayectorias.py".
Sin embargo, aún no es momento para la ejecución del script. Primero hemos de descargar los requisitos, para ello usaremos el comando:

pip install -r requirements.txt

Si ocurren errores con cualquier comando pip, se recomienda usar este comando y luego, volver a ejecutar el comando en cuestión:
python -m pip install --upgrade pip

Esto nos descargará todas las librerías necesarias para la ejecución del programa. En el archivo requirements.txt se pueden ver individualmente los requisitos.

3- Ejecución

Para la ejecución del programa, posicionamos la CMD en la carpeta del script y usamos el comando:

python detector_trayectorias.py [-v VIDEO] [-vs VIDEO_SALIDA] [-y YOLO] [-t TRACKER] [-tp TOLERANCIA] [-nv NUMERO_VECTORES] [-u UMBRAL] [-s SHOW] [-d DEBUG]

Breve explicación de los parámetros: 

-v/--video: Vídeo. Obligatorio. Nombre, extensión y dirección relativa del vídeo de entrada, el programa buscará el archivo y cargará el vídeo en el programa. Recomendable poner el vídeo en la carpeta media, así el comando de entrada quedaría como media/nombrevideo.extensión.

-vs/--video_salida: Vídeo de salida. Opcional. El programa generará un vídeo de salida con los resultados, este parámetro es el nombre que tendrá el vídeo de salida. El formato debe ser: 'nombrevideo.extensión'. Por defecto, video_salida.mp4

-y/--yolo: Opcional. Versión de Yolov5 a usar. Disponibles, de menos potente a más potente: s, m, l, x. Por defecto, s. Cuanto más potente sea la versión, más tardará el procesamiento de un frame.

-t/--tracker: Opcional. Versiones de Tracker a usar: BOOSTING, MIL, KCF,TLD, MEDIANFLOW, GOTURN, MOSSE, CSRT. Por defecto, CSRT.

-tp/--tolerancia_posicion: Opcional. Tolerancia a la diferencia de posición a la hora de creación de tracker. Cuanto menor sea la tolerancia, es más posible que se detecten varias veces un mismo objeto pero también es menos posible que no se detecte un objeto (Mayor cantidad de falsos positivos y positivos múltiples a cambio de menor cantidad de falsos negativos). Por defecto 0.05

-nv/--numero_vectores: Opcional. Número de vectores cercanos que se comprueban en el análisis de trayectorias. Un número alto ralentiza ligeramente el procesamiento y tiene a mejorar la calidad de las detecciones correctas pero también tiene tendencia a generar falsos positivos y falsos negativos, sobretodo al inicio de la ejecución. Por defecto, 5

-u/--umbral: Opcional. El umbral es el número que se usará para comprobar si un vehículo tiene trayectoria correcta o no. Cuanto mayor sea, se dará más flexibilidad a la hora de detectar como correcta o incorrecta una trayectoria. Por defecto, 30.

-s/--show: Opcional. Permite mostrar cada frame a la vez que es procesado. Puede ralentizar en un 15 por ciento la ejecución del programa la ejecución del programa. No es recomendable para ordenadores con gran poder de procesamiento pues la librería usada no es capaz de actualizar rápidamente las imágenes. True/False. Por defecto, False

-d/--debug: Opcional. Muestra por consola información de debug. True/False. Por defecto, False.


Comando de ejemplo:
python .\detector_trayectorias.py -v media/video1.mp4 -d False -s True -tp 0.07 -y m

pyyaml
tqdm
torch
torchvision
seaborn

Al ejecutar el programa, si no se tiene la versión de Yolov5 descargada (se trata de un archivo llamado yolov5*.pt), el programa se encargará automáticamente de buscar y descargar la versión requerida. Ésto puede llevar unos minutos.