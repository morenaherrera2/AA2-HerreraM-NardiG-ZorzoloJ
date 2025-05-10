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

# Función para capturar gestos. 0 piedra - 1 papel - 2 tijera
def capturar_gestos(cap, hands, mp_hands, mp_drawing, gestures, output_dir):
    dataset = []
    labels = []
    nombres_img = []
    print("Presioná 0, 1 o 2 para capturar un gesto (e para salir)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Gestos', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('e'):
            break
        elif key in [ord('0'), ord('1'), ord('2')]:
            gesture_label = int(chr(key))
            gesture_name = list(gestures.keys())[gesture_label]

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                flat_landmarks = np.array(landmarks).flatten()

                dataset.append(flat_landmarks)
                labels.append(gesture_label)

                img_name = f"{gesture_name}_{len(dataset)}.jpg"
                img_path = os.path.join(output_dir, img_name)
                cv2.imwrite(img_path, frame)

                # Agregar el nombre del archivo
                nombres_img.append(img_name)

                print(f"Gesto '{gesture_name}' capturado. Imagen: {img_name}")
            else:
                print("No se detectó una mano.")

        time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()

    return np.array(dataset), np.array(labels), np.array(nombres_img)

# Función para cargar y mostrar dataset (para chequear nosotras/informativo)
def cargar_y_mostrar_datos():
    X = np.load('rps_dataset.npy')
    y = np.load('rps_labels.npy')

    print("Forma de X (dataset):", X.shape)
    print("Forma de y (etiquetas):", y.shape)

    print("\nPrimeras 3 muestras:")
    for i in range(3):
        print(f"Etiqueta: {y[i]}")
        print(f"Coordenadas: {X[i]}")

# Función principal para ejecutar el flujo completo
def main():
    output_dir = 'gestos_dataset'
    crear_carpeta_limpia(output_dir)

    cap, hands, mp_hands, mp_drawing = configurar_camara()
    gestures = {"piedra": 0, "papel": 1, "tijeras": 2}

    dataset, labels, nombres_img = capturar_gestos(cap, hands, mp_hands, mp_drawing, gestures, output_dir)

    # Guardar los archivos
    np.save('rps_dataset.npy', dataset)
    np.save('rps_labels.npy', labels)
    np.save('nombres_img.npy', nombres_img)

    print("Archivos guardados: 'rps_dataset.npy', 'rps_labels.npy', 'nombres_img.npy'")

main()