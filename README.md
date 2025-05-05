# Trabajo Práctico - Aprendizaje Automático II
Este repositorio contiene la solución a tres problemas propuestos
en el Trabajo Práctico N°1 de la materia Aprendizaje Automático II.
Abarca técnicas de regresión con redes neuronales, clasificación de gestos con MediaPipe y redes densas,
y clasificación de imágenes usando redes convolucionales y transfer learning.

## 🖋 Autores
Herrera Morena, Nardi Gianella, Zorzolo Rubio Juana

## 🧠 Problema 1: Predicción del Rendimiento Académico
Descripción:
Se utilizó un conjunto de datos que incluye información sobre hábitos de estudio y estilo de vida de estudiantes universitarios para predecir su rendimiento académico (índice de rendimiento) utilizando un modelo de regresión basado en redes neuronales.

Variables del dataset:

- Hours Studied
- Previous Scores
- Extracurricular Activities
- Sleep Hours
- Sample Question Papers Practiced

Variable objetivo:
- Performance Index (rango de 10 a 100)

Objetivo:
- Construir un modelo de regresión con redes neuronales capaz de predecir el Performance Index.

Contenido entregado:

- Análisis exploratorio y preprocesamiento de los datos.
- Entrenamiento del modelo con validación.
- Evaluación del desempeño con métricas de regresión.
- Visualizaciones de resultados.

Ubicación:

📄 AA2 - TP1.ipynb (Colab Notebook)

## ✋ Problema 2: Clasificación de Gestos con MediaPipe
Descripción:
Sistema para reconocer gestos de "piedra", "papel" y "tijeras" usando detección de manos con MediaPipe y una red neuronal densa.

Etapas:

- record-dataset.py: Captura de datos usando webcam y MediaPipe. Guarda coordenadas de landmarks en archivos .npy.
- train-gesture-classifier.py: Entrenamiento de un modelo denso con las coordenadas.
- rock-paper-scissors.py: Prueba en tiempo real del modelo con webcam.

Objetivo:
- Entrenar un clasificador capaz de identificar correctamente el gesto de la mano basado en los landmarks de MediaPipe.

Contenido entregado:

- 3 scripts funcionales en Python.
- Imágenes que demuestran el funcionamiento del sistema.
- Modelo entrenado (.h5).
- Código comentado.

Ubicación:

📁 scripts_python

## 🌍 Problema 3: Clasificación de Escenas Naturales con CNN
Descripción:
Se trabajó con un dataset de aproximadamente 14.000 imágenes de escenas naturales clasificadas en seis categorías:

- buildings
- forest
- glacier
- mountain
- sea
- street

Objetivo:

Construir varios modelos de clasificación de imágenes:

- Modelo con capas densas.
- Modelo con capas convolucionales + densas.
- Modelo con bloques residuales (identidad).
- Modelo con transfer learning usando un backbone de tf.keras.applications.

Contenido entregado:

- Preprocesamiento de imágenes.
- Implementación y entrenamiento de los 4 modelos.
- Evaluación de desempeño con métricas y visualizaciones.
- Comparación entre arquitecturas.

Ubicación:

📄 AA2 - TP1.ipynb (Colab Notebook)

## 📁 Estructura del Repositorio

AA2 - TP1.ipynb # Notebook con la solución de los problemas 1 y 3 (Rendimiento Académico y Clasificación de Escenas Naturales).

scripts_python/

record-dataset.py # Script para grabar el dataset de gestos usando MediaPipe.

train-gesture-classifier.py # Script para entrenar el clasificador de gestos.

rock-paper-scissors.py # Script para probar el sistema de clasificación de gestos en tiempo real.

imágenes/

... # Imágenes que muestran el funcionamiento del sistema de clasificación de gestos.

... # Ejemplos de imágenes procesadas en la clasificación de escenas naturales.

## 📝 Instrucciones de Ejecución

### Ejercicio 1 y 3 - Predicción del Rendimiento Académico y Clasificación de Escenas Naturales
1. Accede al archivo **AA2 - TP1.ipynb** en Google Colab.
2. Sube los datasets correspondientes al entorno de Google Colab.
3. Ejecuta las celdas en orden para realizar el análisis, entrenamiento del modelo y evaluación.
4. Las visualizaciones y resultados de las métricas de desempeño se generarán automáticamente.

### Ejercicio 2 - Clasificación de Gestos con MediaPipe
1. Para grabar el dataset de gestos:
   - Ejecuta el script **record-dataset.py** para capturar imágenes de la cámara y guardar las coordenadas de los landmarks.
   - Las coordenadas se guardarán en archivos `.npy` (por ejemplo, `rps_dataset.npy` y `rps_labels.npy`).
2. Para entrenar el modelo:
   - Ejecuta el script **train-gesture-classifier.py** para entrenar el modelo con los datos grabados.
   - El modelo entrenado se guardará en un archivo `.h5` (por ejemplo, `rps_model.h5`).
3. Para probar el sistema:
   - Ejecuta el script **rock-paper-scissors.py** para realizar la clasificación de gestos en tiempo real.
   - El modelo predecirá y mostrará el gesto de la mano (piedra, papel o tijeras) en pantalla.

## 📚 Requisitos

- Python 3.x
- TensorFlow 2.x
- MediaPipe
- NumPy
- Matplotlib (para visualización de resultados)
- OpenCV (para captura de video)

Instalación de dependencias:

```bash
pip install tensorflow mediapipe numpy matplotlib opencv-python
