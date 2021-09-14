import argparse

import os

from numpy.lib.function_base import angle 
os.environ["KMP_DUPLICATE_LIB_OK"]="True"

#Math
from math import sqrt, exp, pi
import statistics

#Numpy
import numpy as np

#OpenCV
import cv2

#Yolov5
import torch

#Kalman Filter
from filterpy.kalman import KalmanFilter, kalman_filter
from filterpy.common import Q_discrete_white_noise

#Time
import time

def createTrackerByName(trackerType):
  # Create a tracker based on tracker name
  if trackerType == "BOOSTING":
    tracker = cv2.TrackerBoosting_create()
  elif trackerType == "MIL":
    tracker = cv2.TrackerMIL_create()
  elif trackerType == "KCF":
    tracker = cv2.TrackerKCF_create()
  elif trackerType == "TLD":
    tracker = cv2.TrackerTLD_create()
  elif trackerType == "MEDIANFLOW":
    tracker = cv2.TrackerMedianFlow_create()
  elif trackerType == "GOTURN":
    tracker = cv2.TrackerGOTURN_create()
  elif trackerType == "MOSSE":
    tracker = cv2.TrackerMOSSE_create()
  elif trackerType == "CSRT":
    tracker = cv2.TrackerCSRT_create()
  else:
    tracker = None
    print('Tracker erróneo')
    print('Trackers disponibles:')
    print('BOOSTING, MIL, KCF,TLD, MEDIANFLOW, GOTURN, MOSSE, CSRT')
    exit()
  return tracker



def main():

  #Parámetros
  ap = argparse.ArgumentParser()
  #Obligatorio
  ap.add_argument("-v", "--video", required=True,
                  help="Parámetro obligatorio. Dirección relativa del vídeo de entrada. Recomendable poner el vídeo en la carpeta media, así el comando de entrada quedaría como media/nombrevideo.extensión")
  #Opcionales
  ap.add_argument("-vs", "--video_salida", default="video_salida.mp4", help="Nombre del vídeo procesado. El formato debe ser: 'nombrevideo.extensión'. Por defecto, video_salida.mp4")
  ap.add_argument("-y", "--yolo", default="s", help="Versión de Yolov5 a usar. Disponibles, de menos potente a más potente: s, m, l, x. Por defecto, s. Cuanto más potente sea la versión, más tardará el procesamiento de un frame.")
  ap.add_argument("-t", "--tracker", default="CSRT", help = "Versiones de Tracker: BOOSTING, MIL, KCF,TLD, MEDIANFLOW, GOTURN, MOSSE, CSRT. Por defecto, CSRT.")
  ap.add_argument("-tp", "--tolerancia_posicion", default="0.05", help= "Tolerancia a la diferencia de posición a la hora de creación de tracker. Cuanto menor sea la tolerancia, es más posible que se detecten varias veces un mismo objeto pero también es menos posible que no se detecte un objeto (Mayor cantidad de falsos positivos y positivos múltiples a cambio de menor cantidad de falsos negativos). Por defecto 0.05")
  ap.add_argument("-nv", "--numero_vectores", default="5", help="Número de vectores cercanos que se comprueban en el análisis de trayectorias. Un número alto ralentiza ligeramente el procesamiento y tiene a mejorar la calidad de las detecciones correctas pero también tiene tendencia a generar falsos positivos y falsos negativos, sobretodo al inicio de la ejecución. Por defecto, 5")
  ap.add_argument("-u", "--umbral", default="30", help="El umbral es el número que se usará para comprobar si un vehículo tiene trayectoria correcta o no. Cuanto mayor sea, se dará más flexibilidad a la hora de detectar como correcta o incorrecta una trayectoria. Por defecto, 30.")
  ap.add_argument("-s", "--show", default= "False", help= "Permite mostrar cada frame a la vez que es procesado. Puede ralentizar en un 15 por ciento la ejecución del programa la ejecución del programa. No es recomendable para ordenadores con gran poder de procesamiento pues la librería usada no es capaz de actualizar rápidamente las imágenes. True/False. Por defecto, False")
  ap.add_argument("-d", "--debug", default= "False", help= "Muestra por consola información de debug. True/False. Por defecto, False.")    

  args = vars(ap.parse_args())

  #Inicialización de variables
  #Variable debug
  if args['debug'] == "True":
    debug = True
  elif args['debug'] == "False":
    debug = False
  else:
    #Si se ha dado información incorrecta, muestra error
    print("Error, Debug Incorrecto")
    print("Debug admitido: True, False")
    exit()
  
  #Variable show
  if args['show'] == "True":
    show = True
  elif args['show'] == "False":
    show = False
  else:
    #Si se ha dado información incorrecta, muestra error
    print("Error, Show Incorrecto")
    print("Debug admitido: True, False")
    exit()
  
  #Parámetros mostrados en modo debug
  if debug:
      print("Parámetros")
      print(args)
      print("-----------------------------------")

  #Descarga de Yolo
  #Comprobación de la versión de Yolov5
  if(args["yolo"] != 's' and args["yolo"] != 'm' and args["yolo"] != 'l' and args["yolo"] != 'x' ):
      print("Error, versión de Yolo incorrecta")
      print("Versiones disponibles: s, m, l y x.")
      exit()
  
  #Descarga de la versión de Yolov5, si se encuentra el archivo yolov_.pt correspondiente a la versión, se usará ese en vez de descargarlo.
  model = torch.hub.load('ultralytics/yolov5', 'yolov5' + args['yolo'])
  model = model.autoshape()  # for PIL/cv2/np inputs and NMS

  #Procesamiento del vídeo
  cap = cv2.VideoCapture(args['video'])
  if (cap.isOpened()== False):
      #Si no se puede encontrar el vídeo, muestra error.
      print("Error al abrir el vídeo. Compruebe la dirección. Recomendable poner el vídeo en la carpeta media, así el comando de entrada quedaría como media/nombrevideo.extensión")
      print("Parámetro recibido: " + args['video'])
      exit()

  #Obtención de los fps y los frames totales del vídeo
  fps = cap.get(cv2.CAP_PROP_FPS)
  frames_totales = cap.get(cv2.CAP_PROP_FRAME_COUNT)

  #FPS y frames totales mostrados en modo debug
  if debug:
      print("FPS del vídeo")
      print(fps)
      print("Frames totales estimados del vídeo")
      print(frames_totales)
      print("-----------------------------------")

  #Inicialización de variables
  resultframes = []
  tiempo_frames = 0
  frame_number = 0
  boxes = []

  #Filtros Kalman
  #Lista de filtros en eje X e Y
  f_x = []
  f_y = []
  #Lista con la información sobre si un filtro Kalman está activo o no
  kalman_activos = []
  #Lista con la información proporcionada por los filtros Kalman a lo largo de la ejecución
  kalman_tracks = []
  
  #Inicialización de variables procedentes de los parámetros del programa
  tolerancia_pixel = float(args['tolerancia_posicion'])
  vectores = int(args['numero_vectores'])
  trackerType = args['tracker']
  umbral = float(args['umbral'])

  

  # Obtención del primer frame
  ret, frame = cap.read()

  # Creación de la variable del Multitracker
  multiTracker = cv2.MultiTracker_create()

  #Ejecución principal del programa
  while (cap.isOpened()):

    #Obtención del siguiente frame
    ret, frame = cap.read()

    #Si el frame se ha podido obtener
    if ret == True:
      #Dimensiones del frame
      height, width, layers = frame.shape


      #Tiempo en el que se inicia la ejecución del frame
      start = time.time()

      #Frame ejecutándose
      print("Frame nº " + str(frame_number) + " de unos " + str(frames_totales))
      frame_number+=1
      

      #Actualización a priori
      #Información mostrada antes de la actualización
      if debug:
        print("Trackers antes de la primera actualización")
        print(boxes)
        print("-----------------------------------")

      #Actualización a priori
      success, boxes = multiTracker.update(frame)

      #Información mostrada después de la actualización
      if debug:
        print("Trackers después de la primera actualización")
        print(boxes)
        print("-----------------------------------")
      

      #Detecciones de Yolo sobre el frame actual
      result = model(frame, size=640)
      img = []
      img.append(frame)


      #Comprobación y creación de trackers 
      for detection in result.tolist()[0].pred:
        crear = True
        j = 0

        #Posición de inicio y final de la detección de Yolo
        if debug:
          print("Posición de inicio y final de la Detección. Formato (posicion_inicio_x, posicion_inicio_y, posicion_final_x, posicion_final_y, confianza en la detección, clase)")
          print(detection)
          print("-----------------------------------")
        
        #Comprobación de los tracker para crear (o no) uno a la detección actual 
        for tracker in multiTracker.getObjects():

          #Sólo 
          if crear and kalman_activos[j]:
            #Distancia entre el tracker y la detección
            relacion_x = float(abs(detection[0]/height - tracker[0]/height))
            relacion_y = float(abs(detection[1]/width - tracker[1]/width))

            #Posición y anchura del tracker
            if debug:
              print("Posición y anchura del tracker "+ str(j)+ ". Formato (posicion_inicio_x, posicion_inicio_y, anchura_x, anchura_y)")
              print(tracker)
              print("Relaciones de la detección con el tracker "+ str(j)+ ". Formato (relacion_x, relacion_y)")
              print(str(relacion_x) + ", " + str(relacion_y))
            
            #Si hay un tracker demasiado cerca, evita que se cree otro
            if relacion_x < tolerancia_pixel and relacion_y < tolerancia_pixel:
              crear = False

              
              if debug:
                print("Esta detección no va a crear nada, está demasiado cerca de un tracker existente")
                print("+++++++++++++++++++++++++++++++++++")
          
          j += 1
        
        #Si no se ha encontrado ningún tracker lo suficientemente cerca de la detección, se procede a crear un tracker.
        if crear:
          if debug:
            print("Tracker creado en la detección")
            print("+++++++++++++++++++++++++++++++++++")
          
          #Creación del tracker a partir de los datos de la detección
          multiTracker.add(createTrackerByName(trackerType), frame, (detection[0], detection[1], detection[2]- detection[0], detection[3]- detection[1]))
          #El tracker creado siempre deberá empezar activo
          kalman_activos.append(True)
          #Información de la posición inicial
          kalman_tracks.append([[float(detection[0]), float(detection[1]), float(detection[2]- detection[0]), float(detection[3]- detection[1]) ]])
          
          # Kalman Filter Creation
          #Filtro de X
          aux_filter_x = KalmanFilter (dim_x=2, dim_z=1)
          #Predicción del estado estimado
          aux_filter_x.x = np.array([(detection[0] + detection[2]) /2, 0.])
          #Matriz de transicción de estado
          aux_filter_x.F = np.array([[1.,1.],[0.,1.]])
          #Función de medición
          aux_filter_x.H = np.array([[1.,0.],])
          #Matriz de covarianza
          aux_filter_x.P *= 1000.
          #Cantidad de ruido
          aux_filter_x.R = 5
          #Tipo de ruido
          aux_filter_x.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)
          f_x.append(aux_filter_x)

          #Filtro de Y
          aux_filter_y = KalmanFilter (dim_x=2, dim_z=1)
          #Predicción del estado estimado
          aux_filter_y.x = np.array([(detection[1] + detection[3]) /2, 0.])
          #Matriz de transicción de estado
          aux_filter_y.F = np.array([[1.,1.],[0.,1.]])
          #Función de medición
          aux_filter_y.H = np.array([[1.,0.],])
          #Matriz de covarianza
          aux_filter_y.P *= 1000.
          #Cantidad de ruido
          aux_filter_y.R = 5
          #Tipo de ruido
          aux_filter_y.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)
          f_y.append(aux_filter_y)
      
      #Información mostrada antes de la segunda actualización
      if debug:
        print("----------------------------------")
        print("Trackers antes de la segunda actualización")
        print(boxes)
        print("----------------------------------")

      #Segunda actualización
      old_boxes = boxes
      success, boxes = multiTracker.update(frame)
      
      if frame_number ==1:
        old_boxes = boxes

      #Información mostrada después de la segunda actualización
      if debug:
        print("Trackers después de la segunda actualización")
        print(boxes)
        print("----------------------------------")


      #Actualización y predicción de filtros Kalman, cálculo de medias
      if len(old_boxes) != 0:
        #Por cada tracker existente en el frame anterior...
        for i in range(old_boxes.shape[0]):   
          #... si el tracker está activo... 
          if kalman_activos[i]:
            #Parámetros del tracker
            x_center_coord = boxes[i][0] + boxes[i][2]/2
            y_center_coord = boxes[i][1] + boxes[i][3]/2

            #Parámetros anteriores del tracker
            x_old_center_coord = old_boxes[i][0] + old_boxes[i][2]/2
            y_old_center_coord = old_boxes[i][1] + old_boxes[i][3]/2
            
            #Si se ha producido movimiento...
            if (x_center_coord-x_old_center_coord) != 0 and (y_center_coord - y_old_center_coord) != 0:
              #Se guarda el vector de movimiento que se ha producido
              kalman_tracks[i].append([float(x_center_coord),float(y_center_coord),float((x_center_coord-x_old_center_coord)/(1/fps)),float((y_center_coord-y_old_center_coord)/(1/fps))])

            #Calcular media
            #Vector
            actual_vector= [float(x_center_coord),float(y_center_coord),float((x_center_coord-x_old_center_coord)/(1/fps)),float((y_center_coord-y_old_center_coord)/(1/fps)),i]

            vector_list = []
            #Por cada tracker...
            for k in range(len(kalman_tracks)):
              #Si el  tracker no es el tracker que estamos comprobando Y está activo...
              if k != i and kalman_activos[k]:
                #Por cada vector de movimiento que ha producido el tracker...
                for vector_comprobar in kalman_tracks[k]:
                  #Se guarda una tupla con la diferencia de posición y la diferencia entre vectores
                  vector_list.append((
                    sqrt( 
                      (vector_comprobar[0] + actual_vector[0]) **2 +
                      (vector_comprobar[1] + actual_vector[1]) **2 )
                    ,
                    abs( 
                      sqrt( (vector_comprobar[2] - actual_vector[2])**2 + (vector_comprobar[3] - actual_vector[3]) **2 )
                    )
                    ))

            #La lista se ordena en función de la distancia de posición
            vector_list_sorted = sorted(vector_list, key=lambda tup:tup[0])

            #Se realiza la media de los X vectores más cercanos
            mean= 0
            mean_list = []
            if vector_list_sorted:
              for tuple in vector_list_sorted[0:min(vectores-1, len(vector_list_sorted))]:
                 mean_list.append(tuple[1])

              mean = statistics.mean(mean_list)

              #Se muestra la media del tracker
              if debug:
                print("Media del tracker " + str(i))
                print(mean)
                print("----------------------------------")
            
            
            #Predicciones Kalman
            #Predicciones del filtro del eje X
            f_x[i].predict()
            f_x[i].update(float(x_center_coord),float((x_center_coord-x_old_center_coord)/(1/fps)))
            #Predicciones del filtro del eje Y
            f_y[i].predict()
            f_y[i].update(float(y_center_coord),float((y_center_coord-y_old_center_coord)/(1/fps)))

            #Si se prevee que el tracker vaya a abandonar la pantalla, se elimina
            if f_x[i].x[0] > width:
              kalman_activos[i] = False

              if debug:
                print("Tracker " + str(i) + " desactivado")
                print("---------------------------------")

            if f_x[i].x[0] < 0:
              kalman_activos[i] = False

              if debug:
                print("Tracker " + str(i) + " desactivado")
                print("---------------------------------")

            if f_y[i].x[0] > height:
              kalman_activos[i] = False
              
              if debug:
                print("Tracker " + str(i) + " desactivado")
                print("---------------------------------")

            if f_y[i].x[0] < 0:
              kalman_activos[i] = False
              
              if debug:
                print("Tracker " + str(i) + " desactivado")
                print("---------------------------------")

            #Se escribe en la imagen un cuadrado verde alrededor del tracker
            cv2.rectangle(img[0], 
                        (int(x_center_coord)-2,int(y_center_coord)+2),
                        (int(x_center_coord)+2,int(y_center_coord)-2),
                        (255,255,0), 3)

            #Se escribe en la imagen un cuadrado rojo con la predicción del tracker
            cv2.rectangle(img[0],
                        (int(f_x[i].x[0] + f_x[i].x[1]) - 5, int(f_y[i].x[0]+ f_y[i].x[1]) + 5),
                        (int(f_x[i].x[0] + f_x[i].x[1]) + 5, int(f_y[i].x[0]+ f_y[i].x[1]) - 5),
                        (0,255,0),
                        1
                        )

            #Se escribe el número del tracker
            cv2.putText(img[0],
                          str(i), 
                          (int(boxes[i][0]),int(boxes[i][1])),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          1,
                          (0,0,0),
                          2)

            #Se escribe en la imagen la media obtenida, el color varía con el valor de la media
            cv2.putText(img[0],
                        str(round(mean,3)), 
                        (int(x_center_coord) +2,int(y_center_coord)-2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0,255*(exp(-mean/umbral)),255*(1- exp(-mean/umbral))),
                        2)

      #Mostrar frame
      if show:
        cv2.imshow('Frame', img[0])
        cv2.waitKey(1)
      
      #Guardar frame
      resultframes += img

      #Tiempo en el que se finaliza el procesamiento del frame
      end = time.time()
      #Se guarda el tiempo tardado para calcular el tiempo medio de ejecución del frame
      tiempo_frames += end - start
      print("Tiempo de procesamiento del frame: " + str(end - start))

    else:
      break
    

  #Se destruyen todas las ventanas creadas y se deja de usar el vídeo
  cap.release()
  cv2.destroyAllWindows()

  #Se muestran datos sobre la ejecución del programa
  print("Tiempo de ejecución de los ", str(frames_totales) + ": " + str(tiempo_frames) + " segundos. Tiempo de ejecución medio: " + str(tiempo_frames/frames_totales))

  #Se genera el vídeo de salida
  height, width, layers = resultframes[0].shape
  size = (width,height)

  out = cv2.VideoWriter(args['video_salida'],cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
  for i in resultframes:
    # writing to a image array
    out.write(i)

  #Se libera el vídeo ya creado
  out.release()


        


if __name__ == '__main__':
    main()
