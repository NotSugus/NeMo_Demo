# **Introducción a Nvidia NeMo**

Los archivos dentro de esta carpeta contienen los tutoriales de Nvidia NeMo TTS en conjunto con demos de acceso rápido a la aplicación; un tipo de resumen a los tutoriales.

## **Tutoriales Nvidia**

*Los tutoriales han sido corregidos en lo necesario y están hechos para correr en Google Colab.*

1. Introducción a los distintos modelos disponibles de TTS; te permite comparar los resultados que los diferentes modelos ofrecen.
2. Introducción a la modulación de pitch en la salida de audio del modelo, menciona las restricciones del feature.
3. Introducción al modelo TacoTron2 y al entrenamiento del mismo. **PRESENTA FALLAS**
4. Introducción al entrenamiento para el modelo FastPitch. **PRESENTA FALLAS**

## **Demos**

Los demos permiten correr un modelo FastPitch + HiFigan para reproducir un simple string input (**en inglés**) por parte del usuario. La versión notebook está enfocada al uso dentro de Google Colab, mientras que la versión Python interactive está enfocada a correr en un ambiente local con todas las dependencias ya instaladas.

## **Issues en desarrollo:**

### Modelo TTS en español

* **Encontrar base de datos adecuada** para entrenamiento checar la [sección TTS de NeMo](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/tts/intro.html) para ver requerimientos. El principal modelo pretendiente es FastPitch2, por lo que los requerimientos tienen detalles adicionales.
* Indagar sobre el papel de **Riva+TAO** para realizar el **entrenamiento** del modelo Fastpitch.
 * Importar modelo de Nemo -> Riva mediante una función del toolkit.
* Conseguir correr localmente los programas realizados de TTS.
 * Indagar sobre el uso de una **máquina virtual** (requerimientos de linux).
