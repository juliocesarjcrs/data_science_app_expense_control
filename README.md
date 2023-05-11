# Time series forecast in expense control app.

## Requisitos
 Python 3.11.3 Bibliotecas de Python especificadas en el archivo
requirements.txt
pip 23.1.2
Para verificar sus versiones actuales
```sh
python -V
pip -V
```

Para instalarlas las librerias, ejecuta el comando `pip install -r
requirements.txt`

El dataset se alojará en la ruta in/preprocess con el nombre **df_time_monthly**

Tabla
------------------------------------------------------------

date         |  cost       | days_in_month
------------ | ----------- | ---------------------
2019-09-30   | 1440500     |      30
2019-10-31   | 1440500     |      31
## Estructura del proyecto
```
├── .gitignore
├── Dockerfile.dev
├── README.md
├── api/
│   ├── app/
│   │   ├── models.py
│   │   ├── utils.py
│   │   └── views.py
│   ├── main.py
│   └── requirements.txt
├── docker-compose.yml
├── folder_structure.txt
├── requirements.txt
└── src/
    ├── in/
    │   └── preprocess/
    ├── main.py
    ├── models/
    ├── notebooks/
    │   ├── 1_EDA.ipynb
    │   └── 2_Models.ipynb
    ├── out/
    ├── time_series.py
    └── utils.py
```

## Descripción de la estructura
* **.gitignore:** Archivo de configuración de Git para ignorar ciertos archivos o carpetas que no deben ser agregados al repositorio.

* **Dockerfile.dev:** Imagen de Docker para usarla en desarrollo.
* **README.md:** Archivo markdown con la documentación del proyecto.
* **api/:** Carpeta que contiene el código de la API.
* * **app/:** Carpeta que contiene el código de la aplicación FastAPI.
* * * **models.py**: Archivo con las definiciones de los modelos de datos utilizados en la aplicación.
* * * **utils.py:** Archivo con funciones de utilidad para la aplicación.
* * * **views.py:** Archivo con las definiciones de las rutas de la API.

* * **main.py:** Archivo principal de la API que se encarga de ejecutar el servidor.
* * **requirements.txt:** Archivo con las dependencias necesarias para la ejecución de la API FastAPI.
* **docker-compose.yml:** Archivo Docker compose para correr las diferentes imagenes.
* **requirements.txt:** Archivo con las dependencias necesarias para la ejecución del proyecto.
* **src/:** Carpeta que contiene el código fuente.
* * **notebooks/:** Carpeta que contiene los Notebook
* * * **1_EDA.ipynb:** Notebook de Jupyter con el análisis exploratorio de datos.
* * * **2_Models.ipynb:** Notebook de Jupyter con la construcción de modelos de Machine Learning.
* * **in/:** Carpeta que contiene los archivos de entrada del proyecto.
* * * **preprocess/:** Carpeta que contiene el archivo "df_time_monthly" preprocesado.
* * **main.py:** Archivo principal del proyecto que ejecuta todo el flujo.
* * **models/:** Carpeta que contiene el archivo "best_model_47_5%.pkl" con el modelo de Machine Learning entrenado.
* * **out/:** Carpeta donde se guardan los archivos de salida generados por el proyecto.
* * **time_series.py:** Archivo con funciones relacionadas al análisis de series de tiempo.
* * **utils.py:** Archivo con funciones de utilidad para el proyecto
## Ejecución del proyecto
Para ejecutar el proyecto podemos seguir uno de los siguiente pasos:

## Desarrollo

* Para correr la api fastAPI en local usar:

```sh
uvicorn api.main:app --reload
```
* Ingresar a la url generada Ej: http://127.0.0.1:8000

* También digitando http://127.0.0.1:8000/docs#/  autocompletará una documentación donde puedes probar el Api
* Para correr el código de entrenamiento o generar un modelo en la terminal.
Correr lo siguiente ``` python main.py ```

## Desarrollo con  Docker
En una terminal en la raiz del proyecto correr la  api
```
docker compose up devtimeseries
docker compose up --build  devtimeseries # si se necesita reconstruir la imagen
```
El puerto de la imagen en dev 8000 de la izquierda es el que voy a exponer
http://localhost:8000/

For instance, would publish port 80 from the container to port 8080 on the host or external network.

docker run -p 8080:80
## Autor
Julio Cesar Rico.

## Referencias
Referencias a bibliotecas, artículos, datasets, etc.
utilizados en el proyecto.
