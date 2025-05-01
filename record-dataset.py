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

""" El objetivo de este ejercicio es implementar un sistema de clasificación de gestos de "piedra", "papel" 
o "tijeras" utilizando MediaPipe para la detección de las manos y una red neuronal densa para realizar 
la clasificación. El ejercicio se dividirá en tres partes, cada una implementada en un script de Python."""

def crear_carpeta_limpia(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

#-------------------------------------------------------------------------------------------------------
### PARTE 1 GRABACION DEL DATASET
#-------------------------------------------------------------------------------------------------------

# Crear carpeta para guardar imágenes
output_dir = 'gestos_dataset'
crear_carpeta_limpia(output_dir)

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

# Configurar la cámara
cap = cv2.VideoCapture(0)

# Etiquetas y mapeo a números
gestures = {"piedra": 0, "papel": 1, "tijeras": 2}

# Inicializar dataset
dataset = []
labels = []

print("Presioná 0, 1 o 2 para capturar un gesto (q para salir)")

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
    if key == ord('q'):
        break
    elif key in [ord('0'), ord('1'), ord('2')]:
        gesture_label = int(chr(key))
        gesture_name = list(gestures.keys())[gesture_label]

        if results.multi_hand_landmarks:
            # Extraer landmarks de la primera mano detectada
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            flat_landmarks = np.array(landmarks).flatten()

            # Guardar landmarks y etiqueta
            dataset.append(flat_landmarks)
            labels.append(gesture_label)

            # Guardar imagen
            img_name = f"{gesture_name}_{len(dataset)}.jpg"
            img_path = os.path.join(output_dir, img_name)
            cv2.imwrite(img_path, frame)

            print(f"Gesto '{gesture_name}' capturado. Imagen: {img_name}")

        else:
            print("❌ No se detectó una mano. Intenta de nuevo.")

    time.sleep(0.1)

cap.release()
cv2.destroyAllWindows()

# Guardar dataset y etiquetas
np.save('rps_dataset.npy', np.array(dataset)) # coordenadas de los landmarks
np.save('rps_labels.npy', np.array(labels)) # etiquetas de los gestos
print("✅ Dataset y etiquetas guardados en .npy")


#### VER LANDMARKS Y ETIQUETAS
# Cargar los datos
X = np.load('rps_dataset.npy')
y = np.load('rps_labels.npy')

# Mostrar las formas (dimensiones)
print("Forma de X (dataset):", X.shape)
print("Forma de y (etiquetas):", y.shape)

# Mostrar algunos ejemplos
print("\nPrimeras 3 muestras:")
for i in range(3):
    print(f"Etiqueta: {y[i]}")
    print(f"Coordenadas: {X[i]}")



