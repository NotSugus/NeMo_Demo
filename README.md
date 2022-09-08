# **Introducción a Nvidia NeMo**

Los archivos dentro de esta carpeta contienen los tutoriales de Nvidia NeMo TTS en conjunto con demos de acceso rápido a la aplicación; un tipo de resumen a los tutoriales.

## **Instalación Local**

Para instalar localmente NeMo, se recomienda seguir las instrucciones incluidas en el (repositorio oficial de NeMo) [https://github.com/NVIDIA/NeMo]. Para facilitar este proceso, añado la versión simplificada:

1. Crea un ambiente wsl2 local en la computadora a utilizar con NeMo
   * **Importante seguir las instrucciones para instalar los drivers apropiados de NVidia**
   * Posterior a la instalacion, intalar la extension *remote* en VSCode para facilitar los siguientes pasos.
2. Instala Miniconda y crea un ambiente local (puedes llamarlo NeMo).
3. Instala el paquete Cython
4. Instala la version mas actualizada de NeMo

Comparto los problemas mas frecuentes durante el proceso en la ultima seccion.

## **Tutoriales Nvidia**

*Los tutoriales han sido corregidos en lo necesario y están hechos para correr en Google Colab.*

1. Introducción a los distintos modelos disponibles de TTS; te permite comparar los resultados que los diferentes modelos ofrecen.
2. Introducción a la modulación de pitch en la salida de audio del modelo, menciona las restricciones del feature.
3. Introducción al modelo TacoTron2 y al entrenamiento del mismo. **PRESENTA FALLAS**
4. Introducción al entrenamiento para el modelo FastPitch. **PRESENTA FALLAS**

## **Demos**

Los demos permiten correr un modelo FastPitch + HiFigan para reproducir un simple string input (**en inglés**) por parte del usuario. La versión notebook está enfocada al uso dentro de Google Colab, mientras que la versión Python interactive está enfocada a correr en un ambiente local con todas las dependencias ya instaladas.

## **En desarrollo:**

### Modelo TTS en español

Issue solucionado, se encontró modelo apto para utilizar. Se debe de considerar la posibilidad de mejorarlo. Se ha cambiado desarrollo a otro repo.

### **Script requirements**

Generar un script de dependencias para facilitar la implementación local en cualquier máquina. Se están considerando *docker files* o *bash scripts* para ir a la raiz del problema con las instalaciones nuevas.

## **Issues conocidos:**

Se sabe que la instalación de todas las librerías puede ser un proceso con complicaciones. A continuación hay un listado de los problemas más comunes:

* **Cython**
* **Pynini**
* **Fasttext**
* **PESQ** --> SOLUCION: Esta libreria requiere un compilador de C para ser instalado.

Otro posible problema se encuentra al no tener los drivers o requierimientos adecuados en la GPU. Se recomiendan al menos 2 gb de VRAM para utilizar el toolkit. Una forma rapida para sabr si está disponible CUDA en tu computadora es correr:

```python
import torch
torch.cuda.is_available()
```

De regresar `false`, el sistema no cuenta con los requerimientos de hardware o drivers necesarios.
