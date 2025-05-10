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
from sklearn.metrics import classification_report, confusion_matrix

#----------------------------------
# PARTE 2 - train_gesture_classifier
#----------------------------------

# Función para cargar y escalar el dataset
def cargar_dataset(dataset_path, labels_path, filenames_path):
    X = np.load(dataset_path)
    y = np.load(labels_path)
    file_names = np.load(filenames_path)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, file_names

# Función para dividir los datos en entrenamiento y prueba
def dividir_datos(X, y, file_names, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test, file_train, file_test = train_test_split(
        X, y, file_names, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test, file_train, file_test

# Función para crear el modelo
def crear_modelo(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Función para entrenar el modelo
def entrenar_modelo(model, X_train, y_train, X_test, y_test, epochs=30, batch_size=16):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    return history

# Función para evaluar el modelo
def evaluar_modelo(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nAccuracy en test: {accuracy:.2f}")
    print(f"\nLoss en test: {loss:.2f}")

# Función para generar reporte y matriz de confusión
def generar_reporte_clasificacion(model, X_test, y_test, class_names):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión")
    plt.show()

# Función para guardar el modelo
def guardar_modelo(model, filename):
    model.save(filename)
    print(f"Modelo guardado como '{filename}'")

# Función para mostrar todas las imágenes del conjunto y_test con sus valores reales y predichos
def mostrar_imagenes(y_test, y_pred, file_test, class_names):
    cols = 5
    rows = len(y_test) // cols + (1 if len(y_test) % cols != 0 else 0)
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 4))
    axes = axes.ravel()

    for i in range(len(y_test)):
        img_name = file_test[i]
        img_path = os.path.join('gestos_dataset', img_name)

        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i].imshow(img_rgb)
        else:
            img = np.ones((100, 100, 3), dtype=np.uint8) * 255
            axes[i].imshow(img)

        gesture_name = class_names[y_test[i]]
        predicted_name = class_names[y_pred[i]]

        axes[i].set_title(f"Real: {gesture_name}\nPred: {predicted_name}")
        axes[i].axis('off')

    for j in range(len(y_test), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# Función para mostrar todas las imágenes del conjunto de prueba sin predicciones
def mostrar_imagenes_data_test(file_test):
    cols = 5
    rows = len(file_test) // cols + (1 if len(file_test) % cols != 0 else 0)
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 4))
    axes = axes.ravel()

    for i in range(len(file_test)):
        img_name = file_test[i]
        img_path = os.path.join('gestos_dataset', img_name)

        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i].imshow(img_rgb)
        else:
            img = np.ones((100, 100, 3), dtype=np.uint8) * 255
            axes[i].imshow(img)

        axes[i].axis('off')

    for j in range(len(file_test), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def main():
    dataset_path = 'rps_dataset.npy'
    labels_path = 'rps_labels.npy'
    filenames_path = 'file_names.npy'
    class_names = ['piedra', 'papel', 'tijeras']

    X, y, file_names = cargar_dataset(dataset_path, labels_path, filenames_path)
    X_train, X_test, y_train, y_test, file_train, file_test = dividir_datos(X, y, file_names)

    model = crear_modelo(input_shape=(42,))
    entrenar_modelo(model, X_train, y_train, X_test, y_test)

    evaluar_modelo(model, X_test, y_test)

    y_pred = np.argmax(model.predict(X_test), axis=1)
    generar_reporte_clasificacion(model, X_test, y_test, class_names)

    guardar_modelo(model, 'modelo_gestos_rps.h5')

    # Mostrar imágenes con etiquetas reales y predichas
    mostrar_imagenes(y_test, y_pred, file_test, class_names)
    
    # Mostrar solo las imágenes
    mostrar_imagenes_data_test(file_test)

main()