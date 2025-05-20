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
import joblib

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
def capturar_gestos_nuevos(cap, hands, mp_hands, mp_drawing, output_dir):
    print("Presioná 'c' para capturar una imagen (e para salir)")
    landmarks_list = []  # Para los landmarks capturados
    nombres_img = []  # Para los nombres de los archivos de las imágenes

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
            for hand_landmarks in resultado.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Capturar Landmarks', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('e'): # Para salir del bucle
            break
        elif key == ord('c'): # Para capturar los landmarks 
            # Si se detectó la mano, extraemos los landmarks del primer conjunto de manos detectadas
            if resultado.multi_hand_landmarks:
                hand_landmarks = resultado.multi_hand_landmarks[0]
                landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark] # Obtenemos las coordenadas
                flat_landmarks = np.array(landmarks).flatten() # Las convertimos a un array plano
                landmarks_list.append(flat_landmarks)

                # Guardamos la imagen capturada
                img_name = f"imagen_{len(landmarks_list)}.jpg"
                img_path = os.path.join(output_dir, img_name)
                cv2.imwrite(img_path, frame)
                nombres_img.append(img_name)
                print(f"Imagen guardada como {img_name}")

        time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()

    # Guardar los landmarks capturados en un archivo .npy
    landmarks_array_prueba = np.array(landmarks_list)
    landmarks_path = os.path.join(output_dir, "landmarks_prueba.npy")
    np.save(landmarks_path, landmarks_array_prueba)

    # Retornamos los landmarks capturados y nombres de archivos
    return landmarks_array_prueba, np.array(nombres_img)

# Función para mostrar los gestos con las etiquetas predichas
def mostrar_imagenes_nuevas(y_pred, file_test, class_names, output_dir):
    columnas = 5
    rows = len(y_pred) // columnas + (1 if len(y_pred) % columnas != 0 else 0)
    fig, axes = plt.subplots(rows, columnas, figsize=(20, rows * 4))
    axes = axes.ravel()

    for i in range(len(y_pred)):
        img_name = file_test[i]
        img_path = os.path.join(output_dir, img_name)

        # Verificamos si la imagen existe
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i].imshow(img_rgb)
        else:
            # Mostramos una imagen en blanco si no se encuentra la imagen
            img = np.ones((100, 100, 3), dtype=np.uint8) * 255
            axes[i].imshow(img)

        # Título con la predicción
        predicted_name = class_names[y_pred[i]]
        axes[i].set_title(f"Pred: {predicted_name}")
        axes[i].axis('off')

    # Desactivamos los ejes restantes
    for j in range(len(y_pred), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

"""# Función principal
def main():
    model_path = 'modelo_gestos_rps.h5'  # Ruta del modelo
    model = tf.keras.models.load_model(model_path)
    class_names = ['piedra', 'papel', 'tijeras']  # Clases del modelo

    output_dir = 'imagenes_prueba' # Destino de las imágenes capturadas
    crear_carpeta_limpia(output_dir)

    # Configuramos la cámara y los módulos de detección de manos
    cap, hands, mp_hands, mp_drawing = configurar_camara()

    # Capturamos los gestos y guardarmos los datos
    landmarks_array_prueba, file_names = capturar_gestos_nuevos(cap, hands, mp_hands, mp_drawing, output_dir)

    # Escalamos los landmarks
    scaler = MinMaxScaler()
    landmarks_array_prueba = scaler.fit_transform(landmarks_array_prueba)

    # Realizamos las predicciones sobre los landmarks capturados
    y_pred = []
    for flat_landmarks in landmarks_array_prueba:
        flat_landmarks = flat_landmarks.reshape(1, -1)
        prediction = model.predict(flat_landmarks)
        class_index = np.argmax(prediction)
        y_pred.append(class_index)

    y_pred = np.array(y_pred)

    # Mostramos las imágenes capturadas con sus predicciones
    mostrar_imagenes_nuevas(y_pred, file_names, class_names, output_dir)

main()

"""

# NUEVA FUNCIÓN PARA PREDICCIÓN EN TIEMPO REAL

def predecir_en_tiempo_real(model_path, class_names):
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load('scaler.pkl') 

    cap, hands, mp_hands, mp_drawing = configurar_camara()

    print("Mostrá un gesto con la mano para ver la predicción en tiempo real.")
    print("Presioná 'e' para salir.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = hands.process(rgb_frame)

        if resultado.multi_hand_landmarks:
            hand_landmarks = resultado.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            flat_landmarks = np.array(landmarks).flatten().reshape(1, -1)

            # Escalamos los landmarks (usamos el scaler del entrenamiento)
            flat_landmarks_scaled = scaler.transform(flat_landmarks)

            prediction = model.predict(flat_landmarks_scaled)
            class_index = np.argmax(prediction)
            predicted_label = class_names[class_index]

            # Mostramos la predicción en la imagen
            cv2.putText(frame, f"Gesto: {predicted_label}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("Predicción en Tiempo Real", frame)

        if cv2.waitKey(1) & 0xFF == ord('e'):
            break

    cap.release()
    cv2.destroyAllWindows()


# FUNCIÓN PRINCIPAL

def main():
    model_path = 'modelo_gestos_rps.h5'
    class_names = ['piedra', 'papel', 'tijeras']

    modo_prediccion_tiempo_real = True  # Cambiamos esto a False para usar el modo anterior con tecla 'c'

    if modo_prediccion_tiempo_real:
        predecir_en_tiempo_real(model_path, class_names)
    else:
        output_dir = 'imagenes_prueba'
        crear_carpeta_limpia(output_dir)

        cap, hands, mp_hands, mp_drawing = configurar_camara()
        landmarks_array_prueba, file_names = capturar_gestos_nuevos(cap, hands, mp_hands, mp_drawing, output_dir)

        scaler = MinMaxScaler()
        landmarks_array_prueba = scaler.fit_transform(landmarks_array_prueba)

        model = tf.keras.models.load_model(model_path)
        y_pred = []
        for flat_landmarks in landmarks_array_prueba:
            flat_landmarks = flat_landmarks.reshape(1, -1)
            prediction = model.predict(flat_landmarks)
            class_index = np.argmax(prediction)
            y_pred.append(class_index)

        y_pred = np.array(y_pred)
        mostrar_imagenes_nuevas(y_pred, file_names, class_names, output_dir)


main()