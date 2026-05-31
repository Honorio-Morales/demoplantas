# Clasificador de Orquideas (TFLite + Streamlit)

Aplicacion web simple para clasificar orquideas desde una imagen usando un modelo TensorFlow Lite.

## Archivos principales

- `app.py`: aplicacion Streamlit.
- `orchid_classifier_float32.tflite`: modelo usado por la app.
- `model_metadata.json`: clases y parametros de preprocesamiento.
- `requirements.txt`: dependencias de Python.

## Requisitos

- Python 3.10+ (recomendado: entorno virtual)

## Instalacion

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecutar

```bash
source .venv/bin/activate
streamlit run app.py
```

Abre la URL que muestra Streamlit (normalmente http://localhost:8501).

## Entrada

La app permite:

- Subir imagen (jpg, jpeg, png)
- Tomar foto con camara
- Pegar imagen desde portapapeles
