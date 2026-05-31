import json
import textwrap
import numpy as np
import streamlit as st
from ai_edge_litert.interpreter import Interpreter

from PIL import Image
from streamlit_paste_button import paste_image_button


# =====================================================
# CONFIGURACIÓN
# =====================================================

st.set_page_config(
    page_title="Clasificador de Orquídeas",
    page_icon="🌺",
    layout="wide",
)


# =====================================================
# ESTILOS
# =====================================================

def inject_styles() -> None:
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Spectral:wght@400;600;700&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">

        <style>
            /* ---- Variables ---- */
            :root {
                color-scheme: light;
                --bg:           #f5f3ee;
                --bg-card:      #ffffff;
                --primary:      #2d5a3d;
                --primary-mid:  #40916c;
                --primary-lt:   #74c69d;
                --primary-pale: #d8f3dc;
                --accent:       #b5838d;
                --text:         #1c1c1c;
                --text-mid:     #4b5563;
                --text-muted:   #9ca3af;
                --border:       #e4e0d8;
                --radius-sm:    8px;
                --radius:       14px;
                --radius-lg:    20px;
                --shadow-sm:    0 1px 4px rgba(0,0,0,0.06);
                --shadow:       0 4px 18px rgba(0,0,0,0.09);
            }

            /* ---- Global ---- */
            html, body, [class*="css"] {
                font-family: 'DM Sans', sans-serif;
                color: var(--text);
            }

            .stApp {
                background-color: var(--bg);
            }

            .main .block-container {
                max-width: 1080px;
                padding-top: 2.5rem;
                padding-bottom: 3rem;
            }

            #MainMenu, footer { visibility: hidden; }

            /* ---- Header ---- */
            .app-header {
                display: flex;
                align-items: center;
                gap: 18px;
                margin-bottom: 4px;
            }

            .app-logo {
                width: 56px;
                height: 56px;
                background: linear-gradient(145deg, var(--primary-lt), var(--primary));
                border-radius: 16px;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 4px 16px rgba(45,90,61,0.28);
                flex-shrink: 0;
            }

            .app-logo i {
                font-size: 26px;
                color: #fff;
            }

            .app-title {
                font-family: 'Spectral', Georgia, serif;
                font-size: 2rem;
                font-weight: 700;
                color: #0f2318 !important;   /* verde muy oscuro, legible sobre fondo claro */
                margin: 0;
                line-height: 1.15;
                letter-spacing: -0.3px;
            }

            .app-caption {
                font-size: 0.82rem;
                color: var(--text-muted);
                margin: 4px 0 0;
                letter-spacing: 0.2px;
            }

            /* ---- Divider ---- */
            .hr { 
                border: none;
                border-top: 1px solid var(--border);
                margin: 22px 0;
            }

            /* ---- Section label ---- */
            .section-label {
                display: flex;
                align-items: center;
                gap: 8px;
                font-size: 0.75rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.9px;
                color: var(--primary);
                margin-bottom: 10px;
            }

            .section-label i {
                font-size: 0.95rem;
                color: var(--primary-mid);
            }

            /* ---- Tabs ---- */
            .stTabs [data-baseweb="tab-list"] {
                background: var(--bg-card);
                border: 1px solid var(--border);
                border-radius: var(--radius);
                padding: 5px;
                gap: 4px;
                box-shadow: var(--shadow-sm);
            }

            .stTabs [data-baseweb="tab"] {
                border-radius: var(--radius-sm);
                padding: 7px 20px;
                font-size: 0.875rem;
                font-weight: 500;
                color: var(--text-mid);
                background: transparent;
                border: none;
            }

            .stTabs [aria-selected="true"] {
                background: var(--primary) !important;
                color: #fff !important;
            }

            .stTabs [data-baseweb="tab-highlight"],
            .stTabs [data-baseweb="tab-border"] {
                display: none;
            }

            /* ---- Prediction card ---- */
            .pred-card {
                background: linear-gradient(135deg, var(--primary) 0%, var(--primary-mid) 100%);
                border-radius: var(--radius-lg);
                padding: 26px 28px;
                color: #fff;
                box-shadow: 0 8px 28px rgba(45,90,61,0.30);
                position: relative;
                overflow: hidden;
            }

            .pred-card::before {
                content: '';
                position: absolute;
                top: -40px; right: -40px;
                width: 160px; height: 160px;
                border-radius: 50%;
                background: rgba(255,255,255,0.06);
            }

            .pred-card::after {
                content: '';
                position: absolute;
                bottom: -25px; left: 90px;
                width: 90px; height: 90px;
                border-radius: 50%;
                background: rgba(255,255,255,0.04);
            }

            .pred-eyebrow {
                font-size: 0.68rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 1.6px;
                color: rgba(255,255,255,0.6);
                margin-bottom: 6px;
            }

            .pred-species {
                font-family: 'Spectral', Georgia, serif;
                font-size: 1.55rem;
                font-weight: 700;
                color: #fff;
                line-height: 1.2;
                margin-bottom: 4px;
            }

            .pred-confidence {
                font-size: 2.5rem;
                font-weight: 700;
                color: var(--primary-pale);
                line-height: 1;
                margin-bottom: 14px;
            }

            .pred-desc {
                font-size: 0.84rem;
                color: rgba(255,255,255,0.72);
                line-height: 1.55;
                border-top: 1px solid rgba(255,255,255,0.16);
                padding-top: 12px;
            }

            /* ---- Top-5 bars ---- */
            .pred-list {
                display: flex;
                flex-direction: column;
                gap: 13px;
            }

            .pred-row-meta {
                display: flex;
                justify-content: space-between;
                align-items: baseline;
                margin-bottom: 5px;
            }

            .pred-row-name {
                font-size: 0.875rem;
                font-weight: 400;
                color: var(--text-mid);
            }

            .pred-row-name.is-first {
                font-weight: 600;
                color: var(--primary);
            }

            .pred-row-pct {
                font-size: 0.8rem;
                font-weight: 600;
                color: var(--text-muted);
            }

            .pred-row-pct.is-first {
                color: var(--primary);
            }

            .bar-track {
                height: 5px;
                background: #ede9df;
                border-radius: 99px;
                overflow: hidden;
            }

            .bar-fill {
                height: 100%;
                border-radius: 99px;
                background: var(--primary-lt);
            }

            .bar-fill.is-first {
                background: linear-gradient(90deg, var(--primary-lt), var(--primary));
            }

            /* ---- Image caption row ---- */
            .img-caption-row {
                display: flex;
                align-items: center;
                gap: 7px;
                font-size: 0.78rem;
                color: var(--text-muted);
                margin-top: 8px;
            }

            /* ---- Placeholder ---- */
            .placeholder-box {
                background: var(--bg-card);
                border: 1.5px dashed var(--border);
                border-radius: var(--radius-lg);
                padding: 80px 24px;
                text-align: center;
            }

            .placeholder-icon-wrap {
                width: 68px;
                height: 68px;
                background: var(--primary-pale);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 18px;
            }

            .placeholder-icon-wrap i {
                font-size: 30px;
                color: var(--primary-mid);
            }

            .placeholder-title {
                font-family: 'Spectral', Georgia, serif;
                font-size: 1.2rem;
                font-weight: 600;
                color: var(--primary);
                margin-bottom: 8px;
            }

            .placeholder-sub {
                font-size: 0.875rem;
                color: var(--text-muted);
                max-width: 380px;
                margin: 0 auto;
                line-height: 1.65;
            }

            /* ---- Streamlit tweaks ---- */
            [data-testid="stImage"] img {
                border-radius: var(--radius);
                border: 1px solid var(--border);
            }

            [data-testid="stFileUploader"] > div {
                border-radius: var(--radius) !important;
            }

            [data-testid="stFileUploaderDropzone"] {
                background: var(--bg-card) !important;
                border: 1px solid var(--border) !important;
                border-radius: var(--radius) !important;
                box-shadow: var(--shadow-sm);
                padding: 10px 12px !important;
            }

            [data-testid="stFileUploaderDropzoneInstructions"] div {
                color: var(--text-mid) !important;
            }

            [data-testid="stFileUploaderDropzoneInstructions"] small,
            [data-testid="stFileUploaderDropzoneInstructions"] span,
            [data-testid="stFileUploaderDropzoneInstructions"] p,
            [data-testid="stFileUploaderDropzone"] small {
                color: #233028 !important;
                opacity: 1 !important;
            }

            [data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"] {
                background: #1f6b50 !important;
                color: #ffffff !important;
                border: 1px solid #174f3c !important;
                border-radius: var(--radius-sm) !important;
            }

            [data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"] *,
            [data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"] span {
                color: #ffffff !important;
                fill: #ffffff !important;
            }

            [data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"]:hover {
                background: #174f3c !important;
                border-color: #103a2c !important;
                color: #ffffff !important;
            }

            [data-testid="stFileUploaderDropzone"] svg {
                color: #ffffff !important;
            }

            /* Padding interior de cada panel de tab */
            .stTabs [data-baseweb="tab-panel"] {
                padding-top: 18px !important;
            }

            /* Paste button: usar el wrapper real (key-paste_button) y recortar la franja sobrante */
            [data-testid="stElementContainer"].st-key-paste_button {
                min-height: 0 !important;
                height: auto !important;
                padding: 0 !important;
                margin-top: 6px !important;
                overflow: visible !important;
            }

            [data-testid="stElementContainer"].st-key-paste_button > div {
                min-height: 0 !important;
                height: auto !important;
                padding: 0 !important;
                overflow: hidden !important;
                border-radius: var(--radius) !important;
            }

            [data-testid="stElementContainer"].st-key-paste_button iframe[data-testid="stCustomComponentV1"] {
                width: 100% !important;
                max-width: 100% !important;
                min-height: 0 !important;
                height: 56px !important;
                max-height: 56px !important;
                padding: 0 !important;
                margin: 0 !important;
                border: none !important;
                background: transparent !important;
                overflow: hidden !important;
                border-radius: var(--radius) !important;
                display: block !important;
            }

            [data-testid="stElementContainer"].st-key-paste_button > div:has(> iframe[data-testid="stCustomComponentV1"]) {
                height: 56px !important;
                min-height: 56px !important;
                max-height: 56px !important;
                overflow: hidden !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =====================================================
# METADATA
# =====================================================

@st.cache_data
def load_metadata() -> dict:
    with open("model_metadata.json", "r", encoding="utf-8") as f:
        return json.load(f)


# =====================================================
# INFORMACIÓN DE ESPECIES
# =====================================================

SPECIES_INFO: dict[str, str] = {
    "Brassia euodes":           "Orquídea epífita del género Brassia, reconocida por sus sépalos alargados.",
    "Elleanthus capitatus":     "Especie andina del género Elleanthus, habitual en bosques nublados.",
    "Gongora rufescens":        "Orquídea tropical del género Gongora con flores péndulas muy llamativas.",
    "Maxillaria platypetala":   "Especie del género Maxillaria de pétalos anchos y fragancia sutil.",
    "Oncidium praestanoides":   "Integrante del grupo Oncidium, abundante en zonas interandinas.",
    "Phragmipedium caudatum":   "Conocida por sus pétalos extremadamente alargados; género Phragmipedium.",
    "Pleurothallis chordata":   "Orquídea de montaña del género Pleurothallis con hojas cordiformes.",
    "Pleurothallis coriacardia":"Especie del género Pleurothallis de hojas coriáceas y flores diminutas.",
    "Pleurothallis loranthophylla": "Orquídea epífita andina del género Pleurothallis.",
    "Ponthieva garayana":       "Especie terrestre del género Ponthieva, poco frecuente en colecciones.",
    "Prosthechea farfanii":     "Miembro del género Prosthechea endémico del Perú.",
    "Sudamerlycaste heyndercxii": "Orquídea del grupo Sudamerlycaste de flores grandes y vistosas.",
    "Sudamerlycaste jamesiorum":"Especie andina del género Sudamerlycaste, habitual sobre 2 000 m s. n. m.",
    "Trichophillia fragans":    "Orquídea conocida por su intensa fragancia; género Trichopilia.",
    "Epidendrum secundum":      "Una de las especies de Epidendrum más comunes en Sudamérica.",
}


# =====================================================
# MODELO
# =====================================================

@st.cache_resource
def load_model() -> Interpreter:
    interpreter = Interpreter(
        model_path="orchid_classifier_float32.tflite"
    )
    interpreter.allocate_tensors()
    return interpreter


# =====================================================
# PROCESAMIENTO
# =====================================================

def preprocess_image(
    image: Image.Image,
    img_size: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, Image.Image]:
    resized = image.resize((img_size, img_size))
    img = np.array(resized, dtype=np.float32) / 255.0
    img = (img - mean) / std
    return np.expand_dims(img, axis=0), resized


def predict(
    interpreter: Interpreter,
    image: Image.Image,
    img_size: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, Image.Image]:
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    tensor, preview = preprocess_image(image, img_size, mean, std)
    interpreter.set_tensor(input_details[0]["index"], tensor)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])[0]
    exp_scores = np.exp(output - np.max(output))
    probs = exp_scores / exp_scores.sum()

    return probs, preview


# =====================================================
# COMPONENTES UI
# =====================================================

def ui_header() -> None:
    st.markdown(
        """
        <div class="app-header">
            <div class="app-logo">
                <i class="bi bi-flower1"></i>
            </div>
            <div>
                <h1 class="app-title">Clasificador de Orquídeas</h1>
                <p class="app-caption">
                    TensorFlow Lite &nbsp;·&nbsp; MobileNetV3 &nbsp;·&nbsp; 15 especies nativas
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def ui_section_label(icon: str, text: str) -> None:
    st.markdown(
        f'<div class="section-label"><i class="bi bi-{icon}"></i>{text}</div>',
        unsafe_allow_html=True,
    )


def ui_divider() -> None:
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)


def ui_top5_bars(probs: np.ndarray, class_names: list[str]) -> None:
    top5 = np.argsort(probs)[::-1][:5]
    rows = ""
    for i, idx in enumerate(top5):
        pct = float(probs[idx]) * 100
        name = class_names[idx]
        first = "is-first" if i == 0 else ""
        rows += textwrap.dedent(
            f"""
            <div>
                <div class="pred-row-meta">
                    <span class="pred-row-name {first}">{name}</span>
                    <span class="pred-row-pct {first}">{pct:.1f}%</span>
                </div>
                <div class="bar-track">
                    <div class="bar-fill {first}" style="width:{pct:.2f}%"></div>
                </div>
            </div>
            """
        ).strip()
    st.markdown(
        f'<div class="pred-list">{rows}</div>',
        unsafe_allow_html=True,
    )


def ui_results(
    image: Image.Image,
    probs: np.ndarray,
    preview: Image.Image,
    class_names: list[str],
    img_size: int,
) -> None:
    top5      = np.argsort(probs)[::-1][:5]
    best_idx  = int(top5[0])
    best_name = class_names[best_idx]
    best_pct  = float(probs[best_idx]) * 100
    desc      = SPECIES_INFO.get(best_name, "Sin descripción disponible.")

    # Imágenes
    col_img, col_proc = st.columns(2, gap="large")

    with col_img:
        ui_section_label("image", "Imagen original")
        st.image(image, use_container_width=True)
        st.markdown(
            '<div class="img-caption-row">'
            '<i class="bi bi-aspect-ratio"></i>'
            f'{image.width} × {image.height} px'
            '</div>',
            unsafe_allow_html=True,
        )

    with col_proc:
        ui_section_label("crop", f"Imagen preprocesada")
        st.image(preview, use_container_width=True)
        st.markdown(
            '<div class="img-caption-row">'
            '<i class="bi bi-aspect-ratio"></i>'
            f'Redimensionada a {img_size} × {img_size} px · normalizada'
            '</div>',
            unsafe_allow_html=True,
        )

    ui_divider()

    # Predicciones
    col_pred, col_top5 = st.columns(2, gap="large")

    with col_pred:
        ui_section_label("award", "Predicción principal")
        st.markdown(
            f"""
            <div class="pred-card">
                <div class="pred-eyebrow">Especie identificada</div>
                <div class="pred-species">{best_name}</div>
                <div class="pred-confidence">{best_pct:.1f}%</div>
                <div class="pred-desc">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_top5:
        ui_section_label("bar-chart-line", "Top 5 candidatas")
        ui_top5_bars(probs, class_names)


def ui_placeholder() -> None:
    st.markdown(
        """
        <div class="placeholder-box">
            <div class="placeholder-icon-wrap">
                <i class="bi bi-flower1"></i>
            </div>
            <div class="placeholder-title">Sin imagen seleccionada</div>
            <div class="placeholder-sub">
                Sube un archivo, usa la cámara o pega una captura
                para iniciar la clasificación.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =====================================================
# ENTRADAS DE IMAGEN
# =====================================================

def input_tabs() -> Image.Image | None:
    tab1, tab2, tab3 = st.tabs(
        ["Subir archivo", "Cámara", "Portapapeles"]
    )
    image: Image.Image | None = None

    with tab1:
        ui_section_label("cloud-upload", "Cargar desde dispositivo")
        uploaded = st.file_uploader(
            "Formatos admitidos: JPG, JPEG, PNG",
            type=["jpg", "jpeg", "png"],
            label_visibility="visible",
        )
        if uploaded:
            image = Image.open(uploaded).convert("RGB")

    with tab2:
        ui_section_label("camera", "Captura en tiempo real")
        camera = st.camera_input(label="Toma la fotografía")
        if camera:
            image = Image.open(camera).convert("RGB")

    with tab3:
        ui_section_label("clipboard", "Pegar desde portapapeles")
        # background-color coincide con --bg-card para que el iframe se integre
        pasted = paste_image_button(
            label="Pegar imagen del portapapeles",
            background_color="#1f6b50",
            hover_background_color="#174f3c",
            text_color="#ffffff",
            key="paste_button_v4",
        )
        if pasted.image_data is not None:
            image = pasted.image_data.convert("RGB")

    return image


# =====================================================
# APP PRINCIPAL
# =====================================================

def main() -> None:
    inject_styles()

    metadata   = load_metadata()
    class_names = metadata["class_names"]
    img_size   = metadata["img_size"]
    mean       = np.array(metadata["normalize_mean"], dtype=np.float32)
    std        = np.array(metadata["normalize_std"],  dtype=np.float32)

    interpreter = load_model()

    ui_header()
    ui_divider()

    image = input_tabs()

    ui_divider()

    if image is not None:
        with st.spinner("Clasificando imagen..."):
            probs, preview = predict(interpreter, image, img_size, mean, std)
        ui_results(image, probs, preview, class_names, img_size)
    else:
        ui_placeholder()


if __name__ == "__main__":
    main()