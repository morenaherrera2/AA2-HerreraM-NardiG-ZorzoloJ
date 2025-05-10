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
# PARTE 3 - rock_paper_scissors
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
def capturar_landmarks(cap, hands, mp_hands, mp_drawing, output_dir):
    print("Presioná 'c' para capturar una imagen (e para salir)")
    landmarks_list = []
    file_names = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Capturar Landmarks', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('e'):
            break
        elif key == ord('c'):
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                flat_landmarks = np.array(landmarks).flatten()
                landmarks_list.append(flat_landmarks)

                # Guardar imagen
                img_name = f"imagen_{len(landmarks_list)}.jpg"
                img_path = os.path.join(output_dir, img_name)
                cv2.imwrite(img_path, frame)
                file_names.append(img_name)
                print(f"Imagen guardada como {img_name}")

        time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()

    # Guardar todos los landmarks en un único archivo npy
    landmarks_array_prueba = np.array(landmarks_list)
    landmarks_path = os.path.join(output_dir, "landmarks_prueba.npy")
    np.save(landmarks_path, landmarks_array_prueba)
    return landmarks_array_prueba, np.array(file_names)

def mostrar_imagenes(y_pred, file_test, class_names):
    cols = 5
    rows = len(y_pred) // cols + (1 if len(y_pred) % cols != 0 else 0)
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 4))
    axes = axes.ravel()

    for i in range(len(y_pred)):
        img_name = file_test[i]
        img_path = os.path.join('imagenes_prueba', img_name)

        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i].imshow(img_rgb)
        else:
            # Imagen en blanco si no se encuentra la imagen
            img = np.ones((100, 100, 3), dtype=np.uint8) * 255
            axes[i].imshow(img)

        predicted_name = class_names[y_pred[i]]
        axes[i].set_title(f"Pred: {predicted_name}")
        axes[i].axis('off')

    for j in range(len(y_pred), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# Función principal actualizada
def main():
    MODEL_PATH = 'modelo_gestos_rps.h5'
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = ['piedra', 'papel', 'tijeras']

    output_dir = 'imagenes_prueba'
    crear_carpeta_limpia(output_dir)

    cap, hands, mp_hands, mp_drawing = configurar_camara()
    landmarks_array_prueba, file_names = capturar_landmarks(cap, hands, mp_hands, mp_drawing, output_dir)

    # Clasificar los landmarks capturados usando el modelo
    y_pred = []
    for flat_landmarks in landmarks_array_prueba:
        flat_landmarks = flat_landmarks.reshape(1, -1)
        prediction = model.predict(flat_landmarks)
        class_index = np.argmax(prediction)
        y_pred.append(class_index)

    y_pred = np.array(y_pred)

    # Mostrar imágenes capturadas con predicciones
    mostrar_imagenes(y_pred, file_names, class_names)

main()