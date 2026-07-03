import math
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

# IMPORT CORRETTO PER STREAMLIT CLOUD:
# con alcune versioni di MediaPipe, mp.solutions non è disponibile come attributo.
# Importiamo direttamente il modulo legacy FaceMesh.
try:
    from mediapipe.python.solutions import face_mesh as mp_face_mesh
except Exception as exc:
    st.error("Errore nel caricamento di MediaPipe. Controlla requirements.txt e packages.txt su GitHub.")
    st.code(str(exc))
    st.stop()

st.set_page_config(page_title="Face Shape Detector", page_icon="✨", layout="centered")

# -----------------------------
# Landmark principali Face Mesh
# -----------------------------
LANDMARKS = {
    "top": 10,
    "chin": 152,
    "left_cheek": 234,
    "right_cheek": 454,
    "left_temple": 127,
    "right_temple": 356,
    "left_jaw": 172,
    "right_jaw": 397,
    "left_chin_side": 58,
    "right_chin_side": 288,
}

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]

@dataclass
class Metrics:
    face_ratio: float
    forehead_to_cheek: float
    jaw_to_cheek: float
    chin_to_cheek: float
    cheek_width: float
    forehead_width: float
    jaw_width: float
    chin_width: float
    face_height: float


def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def get_points(face_landmarks, width: int, height: int) -> Dict[str, Tuple[float, float]]:
    points = {}
    for name, idx in LANDMARKS.items():
        lm = face_landmarks.landmark[idx]
        points[name] = (lm.x * width, lm.y * height)
    return points


def calculate_metrics(points: Dict[str, Tuple[float, float]]) -> Metrics:
    face_height = dist(points["top"], points["chin"])
    cheek_width = dist(points["left_cheek"], points["right_cheek"])
    forehead_width = dist(points["left_temple"], points["right_temple"])
    jaw_width = dist(points["left_jaw"], points["right_jaw"])
    chin_width = dist(points["left_chin_side"], points["right_chin_side"])

    cheek_width = max(cheek_width, 1.0)
    return Metrics(
        face_ratio=face_height / cheek_width,
        forehead_to_cheek=forehead_width / cheek_width,
        jaw_to_cheek=jaw_width / cheek_width,
        chin_to_cheek=chin_width / cheek_width,
        cheek_width=cheek_width,
        forehead_width=forehead_width,
        jaw_width=jaw_width,
        chin_width=chin_width,
        face_height=face_height,
    )


def classify_face(m: Metrics) -> Tuple[str, str, int]:
    r = m.face_ratio
    f = m.forehead_to_cheek
    j = m.jaw_to_cheek
    c = m.chin_to_cheek

    scores = {
        "Ovale": 0,
        "Tondo": 0,
        "Squadrato": 0,
        "Rettangolare": 0,
        "Oblungo": 0,
        "Cuore": 0,
        "Triangolo inverso": 0,
        "Triangolo": 0,
        "Diamante": 0,
    }

    # Lettura geometrica di base: proporzioni verticali/orizzontali del volto.
    if 1.28 <= r <= 1.55:
        scores["Ovale"] += 3
    if 0.80 <= j <= 0.96 and 0.86 <= f <= 1.04:
        scores["Ovale"] += 2

    if r < 1.28:
        scores["Tondo"] += 3
    if 0.86 <= j <= 1.06 and 0.86 <= f <= 1.06 and r < 1.35:
        scores["Tondo"] += 2

    if r < 1.38 and j >= 0.92 and f >= 0.88:
        scores["Squadrato"] += 4
    if c >= 0.32 and j >= 0.90:
        scores["Squadrato"] += 1

    if 1.38 <= r <= 1.62 and j >= 0.88 and f >= 0.88:
        scores["Rettangolare"] += 4
    if c >= 0.30 and j >= 0.88:
        scores["Rettangolare"] += 1

    if r > 1.58:
        scores["Oblungo"] += 5
    if j < 0.90 and f < 1.02 and r > 1.50:
        scores["Oblungo"] += 1

    if f >= 0.98 and j < 0.86 and c < 0.34:
        scores["Cuore"] += 4
    if f > j + 0.12:
        scores["Cuore"] += 2

    if f >= 1.02 and j < 0.82:
        scores["Triangolo inverso"] += 5
    if f > j + 0.18:
        scores["Triangolo inverso"] += 1

    if j >= 0.96 and f < 0.92:
        scores["Triangolo"] += 5
    if j > f + 0.12:
        scores["Triangolo"] += 2

    if f < 0.96 and j < 0.90 and 1.28 <= r <= 1.62:
        scores["Diamante"] += 5
    if c < 0.32 and m.cheek_width > m.forehead_width and m.cheek_width > m.jaw_width:
        scores["Diamante"] += 2

    shape = max(scores, key=scores.get)
    confidence = min(95, max(55, 50 + scores[shape] * 7))

    groups = {
        "Tondo": "Forma allargata",
        "Squadrato": "Forma allargata / spigolosa",
        "Cuore": "Forma allargata",
        "Triangolo": "Forma allargata",
        "Diamante": "Forma spigolosa",
        "Triangolo inverso": "Forma spigolosa",
        "Rettangolare": "Forma allungata",
        "Oblungo": "Forma allungata",
        "Ovale": "Forma equilibrata",
    }
    return shape, groups.get(shape, "Forma da verificare"), confidence


def draw_overlay(image: Image.Image, face_landmarks) -> Image.Image:
    img = image.convert("RGB").copy()
    w, h = img.size
    draw = ImageDraw.Draw(img)

    oval_points: List[Tuple[float, float]] = []
    for idx in FACE_OVAL:
        lm = face_landmarks.landmark[idx]
        oval_points.append((lm.x * w, lm.y * h))

    if len(oval_points) > 2:
        draw.line(oval_points + [oval_points[0]], fill="white", width=4)
        draw.line(oval_points + [oval_points[0]], fill="black", width=2)

    for _, idx in LANDMARKS.items():
        lm = face_landmarks.landmark[idx]
        x, y = lm.x * w, lm.y * h
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill="white", outline="black", width=2)

    return img


def analyze(image: Image.Image):
    image = image.convert("RGB")
    np_image = np.array(image)
    h, w = np_image.shape[:2]

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        result = face_mesh.process(np_image)

    if not result.multi_face_landmarks:
        return None, None, None, None, None

    face_landmarks = result.multi_face_landmarks[0]
    points = get_points(face_landmarks, w, h)
    metrics = calculate_metrics(points)
    shape, group, confidence = classify_face(metrics)
    overlay = draw_overlay(image, face_landmarks)
    return shape, group, confidence, metrics, overlay


st.title("Face Shape Detector")
st.caption("App demo per riconoscere la forma geometrica del volto da foto frontale.")

with st.sidebar:
    st.subheader("Come scattare")
    st.write("• Guarda dritto in camera")
    st.write("• Capelli lontani dal contorno del viso")
    st.write("• Luce uniforme")
    st.write("• Niente filtri beauty o grandangolo troppo vicino")
    st.divider()
    st.write("Forme riconosciute: ovale, tondo, squadrato, rettangolare, oblungo, cuore, triangolo inverso, triangolo, diamante.")

mode = st.radio("Scegli input", ["Scatta con webcam", "Carica immagine"], horizontal=True)

uploaded = None
if mode == "Carica immagine":
    uploaded = st.file_uploader("Carica una foto frontale", type=["jpg", "jpeg", "png", "webp"])
else:
    uploaded = st.camera_input("Scatta una foto frontale")

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Foto acquisita", use_container_width=True)

    with st.spinner("Analisi in corso..."):
        shape, group, confidence, metrics, overlay = analyze(image)

    if shape is None:
        st.error("Non riesco a rilevare un volto frontale. Prova con viso più vicino, luce uniforme e capelli lontani dal perimetro del viso.")
    else:
        st.success(f"Forma rilevata: {shape}")
        st.subheader(shape)
        st.write(f"**Gruppo Face Design:** {group}")
        st.write(f"**Affidabilità stimata:** {confidence}%")

        st.image(overlay, caption="Contorno e punti tecnici rilevati", use_container_width=True)

        with st.expander("Vedi misure tecniche"):
            st.write({
                "lunghezza / larghezza zigomi": round(metrics.face_ratio, 2),
                "fronte / zigomi": round(metrics.forehead_to_cheek, 2),
                "mascella / zigomi": round(metrics.jaw_to_cheek, 2),
                "mento / zigomi": round(metrics.chin_to_cheek, 2),
            })

        st.info("Questa è una stima geometrica automatica. Per Face Design va verificata con consulenza professionale, profilo, collo, altezza e obiettivo di stile.")
