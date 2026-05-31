# Demo de Clasificacion de Orquideas (TFLite + Streamlit)

Este repositorio es un demo para probar un modelo de clasificacion de orquideas.

## Alcance

- Solo esta pensado para validar rapidamente el modelo con imagenes.
- No es una aplicacion de produccion.
- No incluye garantias de precision clinica/cientifica.

## Archivos principales

- `app.py`: interfaz Streamlit para prueba manual.
- `orchid_classifier_float32.tflite`: modelo principal usado por la demo.
- `model_metadata.json`: clases y parametros de preprocesamiento.
- `requirements.txt`: dependencias de Python.

## Requisitos

- Python 3.10 o superior (recomendado: entorno virtual).

## Instalacion

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecucion

```bash
source .venv/bin/activate
streamlit run app.py
```

Luego abre la URL local que muestra Streamlit (usualmente http://localhost:8501).

## Que permite probar

- Subir imagen (jpg, jpeg, png).
- Tomar foto con camara.
- Pegar imagen desde el portapapeles.
