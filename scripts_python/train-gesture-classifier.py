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

#-------------------------------------------------------------------------------------------------------
### PARTE 2 ENTRENAMIENTO DEL CLASIFICADOR
#-------------------------------------------------------------------------------------------------------

# Etiquetas de clases
CLASS_NAMES = ['piedra', 'papel', 'tijeras']

# Cargar dataset y etiquetas
X = np.load('rps_dataset.npy')  # (n_samples, 42)
y = np.load('rps_labels.npy')   # (n_samples,)

# Escalar coordenadas entre 0 y 1
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Separar en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Definir el modelo
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(42,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 clases
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=30, batch_size=16,
                    validation_data=(X_test, y_test))

# Evaluar
loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Accuracy en test: {acc:.2f}")

# Predicciones
y_pred = np.argmax(model.predict(X_test), axis=1)

from sklearn.metrics import classification_report, confusion_matrix
# Reporte y matriz de confusión
print("\n📋 Reporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.show()

# Guardar modelo
model.save("modelo_gestos_rps.h5")
print("💾 Modelo guardado como 'modelo_gestos_rps.h5'")


### VER IMAGENES CON ETIQUETA PREDICHA
# Cargar las imágenes de prueba
test_images = []
test_labels = []

# Cargar las imágenes que se usaron para hacer las predicciones
for i, label in enumerate(y_test):
    img_name = f"{CLASS_NAMES[label]}_{i+1}.jpg"  # Asegúrate de que las imágenes tengan este formato
    img_path = os.path.join(output_dir, img_name)

    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        test_images.append(img)
        test_labels.append(label)

# Número de columnas para mostrar las imágenes
cols = 5
rows = len(test_images) // cols + (1 if len(test_images) % cols != 0 else 0)

# Crear una figura con subgráficos (subplots)
fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
axes = axes.ravel()  # Convertir a un arreglo unidimensional para facilidad

# Cargar y mostrar las imágenes
for i in range(len(test_images)):
    img_rgb = cv2.cvtColor(test_images[i], cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB para mostrar
    predicted_label = y_pred[i]
    real_label = test_labels[i]
    
    # Mostrar la imagen en el subplot correspondiente
    axes[i].imshow(img_rgb)
    axes[i].set_title(f"Pred: {CLASS_NAMES[predicted_label]}\nReal: {CLASS_NAMES[real_label]}")
    axes[i].axis('off')  # Desactivar los ejes

# Si hay más espacio en los subgráficos, desactivarlos
for i in range(len(test_images), len(axes)):
    axes[i].axis('off')

# Ajustar el espaciado
plt.tight_layout()
plt.show()

# Mostrar imágenes de y_test
for i in range(len(y_test)):
    # Crear el nombre de la imagen a partir de la etiqueta de y_test
    gesture_name = CLASS_NAMES[y_test[i]]
    img_name = f"{gesture_name}_{i+1}.jpg"  # Aquí asumimos que las imágenes tienen este formato
    img_path = os.path.join(output_dir, img_name)

    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB para mostrar con matplotlib
        plt.figure(figsize=(3, 3))
        plt.imshow(img_rgb)
        plt.title(f"Etiqueta real: {gesture_name}")
        plt.axis('off')
        plt.show()
    else:
        print(f"❌ No se encontró la imagen: {img_name}")