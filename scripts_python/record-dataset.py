import gdown
import zipfile
import numpy as np
import pandas as pd
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
import time
import shutil
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

#----------------------------------
# PARTE 1 - record_dataset
#----------------------------------

# Función para crear una carpeta limpia
def crear_carpeta_limpia(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

# Función para configurar la cámara
def configurar_camara():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands()
    cap = cv2.VideoCapture(0)
    return cap, hands, mp_hands, mp_drawing

# Función para capturar y preprocesar gestos
def capturar_gestos(cap, hands, mp_hands, mp_drawing, gestos, output_dir):
    dataset = []       # Para almacenar las coordenadas de los gestos
    labels = []        # Para almacenar las etiquetas (0, 1, 2)
    nombres_img = []   # Para almacenar los nombres de los archivos de las imágenes

    print("Presioná 0, 1 o 2 para capturar un gesto (e para salir)")

    # Bucle principal de captura
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Volteamos el frame horizontalmente para simular un espejo
        frame = cv2.flip(frame, 1)

        # Convertimos a RGB para la detección de manos
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesamos el frame para detectar las manos
        resultado = hands.process(rgb_frame)

        # Dibujamos las manos en la pantalla
        if resultado.multi_hand_landmarks:
            for landmarks in resultado.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Gestos', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('e'): # Para salir del bucle
            break
        
        # Capturamos los gestos correspondientes con 0, 1 y 2 
        elif key in [ord('0'), ord('1'), ord('2')]:
            gestos_label = int(chr(key))
            gestos_name = list(gestos.keys())[gestos_label]

            # Si se detectó la mano, extraemos los landmarks del primer conjunto de manos detectadas
            if resultado.multi_hand_landmarks:
                hand_landmarks = resultado.multi_hand_landmarks[0]
                landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark] # Obtenemos las coordenadas
                flat_landmarks = np.array(landmarks).flatten() # Las convertimos a un array plano

                # Verificamos la longitud del array
                if flat_landmarks.shape[0] == 42:
                    dataset.append(flat_landmarks)
                    labels.append(gestos_label)

                # Guardamos la imagen capturada
                img_name = f"{gestos_name}_{len(dataset)}.jpg"
                img_path = os.path.join(output_dir, img_name)
                cv2.imwrite(img_path, frame)

                # Guardamos el nombre del archivo de la imagen
                nombres_img.append(img_name)
                print(f"Gesto '{gestos_name}' capturado. Imagen: {img_name}")
            else:
                print("No se detectó una mano.")

        time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()

    # Retornamos los datasets de coordenadas, etiquetas y nombres de archivos
    #return np.array(dataset), np.array(labels), np.array(nombres_img)
    return np.array(dataset, dtype=np.float32), np.array(labels, dtype=np.int32), np.array(nombres_img)

# Función para cargar y mostrar el dataset guardado
def cargar_y_mostrar_datos():
    X = np.load('rps_dataset.npy')  # Dataset de coordenadas
    y = np.load('rps_labels.npy')   # Etiquetas

    print("Forma de X (dataset):", X.shape)
    print("Forma de y (etiquetas):", y.shape)

    print("\nPrimeras 3 muestras:")
    for i in range(3):
        print(f"Etiqueta: {y[i]}")
        print(f"Coordenadas: {X[i]}")

# Función principal
def main():
    output_dir = 'gestos_dataset'  # Destino de las imágenes
    crear_carpeta_limpia(output_dir)

    # Configuramos la cámara y los módulos de detección de manos
    cap, hands, mp_hands, mp_drawing = configurar_camara()

    # Asignamos los gestos a números
    gestos = {"piedra": 0, "papel": 1, "tijeras": 2}

    # Capturamos los gestos y guardarmos los datos
    dataset, labels, nombres_img = capturar_gestos(cap, hands, mp_hands, mp_drawing, gestos, output_dir)

    # Guardamos los datasets como archivos .npy
    np.save('rps_dataset.npy', dataset)
    np.save('rps_labels.npy', labels)
    np.save('nombres_img.npy', nombres_img)

    print("Archivos guardados: 'rps_dataset.npy', 'rps_labels.npy', 'nombres_img.npy'")

main()