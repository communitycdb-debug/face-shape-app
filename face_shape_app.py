import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# ------------------------------------------------------------
# FACE SHAPE APP - VERSIONE STREAMLIT CLOUD SENZA OPENCV / CV2
# Riconosce un volto frontale e classifica la forma del viso in:
# Ovale, Tondo, Squadrato, Rettangolare, Oblungo, Cuore,
# Triangolo inverso, Triangolo, Diamante.
# ------------------------------------------------------------

st.set_page_config(page_title="Face Shape Detector", page_icon="✨", layout="wide")

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]

LANDMARKS = {
    "top": 10,
    "chin": 152,
    "left_cheek": 234,
    "right_cheek": 454,
    "left_forehead": 54,
    "right_forehead": 284,
    "left_jaw": 172,
    "right_jaw": 397,
    "left_chin_side": 58,
    "right_chin_side": 288,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "nose_tip": 1,
}

FACE_GROUPS = {
    "TONDO": "Forma allargata",
    "SQUADRATO": "Forma allargata",
    "TRIANGOLO": "Forma allargata",
    "CUORE": "Forma allargata / superiore",
    "DIAMANTE": "Forma spigolosa o morbida, da valutare",
    "RETTANGOLARE": "Forma allungata",
    "OBLUNGO": "Forma allungata",
    "OVALE": "Forma bilanciata",
    "TRIANGOLO INVERSO": "Forma spigolosa / superiore",
}


@dataclass
class FaceMeasurements:
    ratio_length_width: float
    forehead_to_cheek: float
    jaw_to_cheek: float
    chin_to_cheek: float
    cheek_width: float
    face_length: float
    eye_tilt_degrees: float
    nose_offset: float


def dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return float(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def closeness(value: float, target: float, tolerance: float) -> float:
    return max(0.0, 1.0 - abs(value - target) / tolerance)


def lm_xy(landmarks, index: int, width: int, height: int) -> Tuple[float, float]:
    lm = landmarks[index]
    return float(lm.x * width), float(lm.y * height)


def get_measurements(landmarks, width: int, height: int) -> FaceMeasurements:
    pts = {name: lm_xy(landmarks, idx, width, height) for name, idx in LANDMARKS.items()}

    cheek_width = dist(pts["left_cheek"], pts["right_cheek"])
    face_length = dist(pts["top"], pts["chin"])
    forehead_width = dist(pts["left_forehead"], pts["right_forehead"])
    jaw_width = dist(pts["left_jaw"], pts["right_jaw"])
    chin_width = dist(pts["left_chin_side"], pts["right_chin_side"])

    eye_dx = pts["right_eye_outer"][0] - pts["left_eye_outer"][0]
    eye_dy = pts["right_eye_outer"][1] - pts["left_eye_outer"][1]
    eye_tilt = math.degrees(math.atan2(eye_dy, eye_dx))
    face_center_x = (pts["left_cheek"][0] + pts["right_cheek"][0]) / 2
    nose_offset = abs(pts["nose_tip"][0] - face_center_x) / max(cheek_width, 1)

    return FaceMeasurements(
        ratio_length_width=face_length / max(cheek_width, 1),
        forehead_to_cheek=forehead_width / max(cheek_width, 1),
        jaw_to_cheek=jaw_width / max(cheek_width, 1),
        chin_to_cheek=chin_width / max(cheek_width, 1),
        cheek_width=cheek_width,
        face_length=face_length,
        eye_tilt_degrees=eye_tilt,
        nose_offset=nose_offset,
    )


def classify_shape(m: FaceMeasurements) -> List[Tuple[str, float]]:
    r = m.ratio_length_width
    f = m.forehead_to_cheek
    j = m.jaw_to_cheek
    c = m.chin_to_cheek

    scores: Dict[str, float] = {
        "TONDO": (
            1.45 * closeness(r, 1.15, 0.28)
            + 1.00 * closeness(f, 0.92, 0.22)
            + 1.00 * closeness(j, 0.86, 0.22)
            + 0.60 * closeness(c, 0.76, 0.25)
        ),
        "SQUADRATO": (
            1.35 * closeness(r, 1.18, 0.25)
            + 1.15 * closeness(f, 0.96, 0.16)
            + 1.15 * closeness(j, 0.96, 0.16)
            + 0.70 * closeness(c, 0.82, 0.20)
        ),
        "OVALE": (
            1.45 * closeness(r, 1.42, 0.30)
            + 0.95 * closeness(f, 0.90, 0.18)
            + 0.95 * closeness(j, 0.82, 0.20)
            + 0.60 * closeness(c, 0.70, 0.22)
        ),
        "RETTANGOLARE": (
            1.45 * closeness(r, 1.60, 0.30)
            + 1.05 * closeness(f, 0.96, 0.16)
            + 1.05 * closeness(j, 0.94, 0.16)
            + 0.60 * closeness(c, 0.78, 0.22)
        ),
        "OBLUNGO": (
            1.55 * closeness(r, 1.72, 0.35)
            + 0.90 * closeness(f, 0.88, 0.20)
            + 0.90 * closeness(j, 0.80, 0.22)
            + 0.55 * closeness(c, 0.68, 0.24)
        ),
        "CUORE": (
            1.10 * closeness(r, 1.36, 0.32)
            + 1.25 * closeness(f, 1.00, 0.20)
            + 1.20 * closeness(j, 0.74, 0.18)
            + 0.85 * closeness(c, 0.60, 0.20)
        ),
        "TRIANGOLO INVERSO": (
            1.00 * closeness(r, 1.32, 0.32)
            + 1.35 * closeness(f, 1.05, 0.22)
            + 1.25 * closeness(j, 0.70, 0.20)
            + 0.85 * closeness(c, 0.56, 0.20)
        ),
        "TRIANGOLO": (
            1.00 * closeness(r, 1.35, 0.34)
            + 1.35 * closeness(f, 0.78, 0.20)
            + 1.30 * closeness(j, 0.96, 0.20)
            + 0.70 * closeness(c, 0.78, 0.22)
        ),
        "DIAMANTE": (
            1.20 * closeness(r, 1.45, 0.34)
            + 1.25 * closeness(f, 0.80, 0.18)
            + 1.25 * closeness(j, 0.74, 0.18)
            + 0.80 * closeness(c, 0.62, 0.20)
        ),
    }

    balanced_widths = max(f, j, 1.0) - min(f, j, 1.0) < 0.10
    if balanced_widths and r < 1.32:
        scores["SQUADRATO"] += 0.45
    if balanced_widths and r >= 1.45:
        scores["RETTANGOLARE"] += 0.45
    if f > j + 0.16:
        scores["CUORE"] += 0.35
        scores["TRIANGOLO INVERSO"] += 0.45
    if j > f + 0.14:
        scores["TRIANGOLO"] += 0.50
    if f < 0.88 and j < 0.84:
        scores["DIAMANTE"] += 0.40
    if r > 1.62 and not balanced_widths:
        scores["OBLUNGO"] += 0.35
    if r < 1.22 and j < 0.92:
        scores["TONDO"] += 0.35

    total = sum(max(v, 0.0) for v in scores.values()) or 1.0
    return sorted(((k, (max(v, 0.0) / total) * 100) for k, v in scores.items()), key=lambda x: x[1], reverse=True)


@st.cache_resource
def load_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.55,
    )


def analyze_image(image_rgb: np.ndarray):
    h, w = image_rgb.shape[:2]
    face_mesh = load_face_mesh()
    result = face_mesh.process(image_rgb)

    if not result.multi_face_landmarks:
        return None, None, "Non ho rilevato un volto. Usa una foto frontale, ben illuminata, senza occhiali scuri."

    landmarks = result.multi_face_landmarks[0].landmark
    measurements = get_measurements(landmarks, w, h)
    ranking = classify_shape(measurements)
    return landmarks, (measurements, ranking), None


def _line(draw: ImageDraw.ImageDraw, p1, p2, fill, width=4):
    draw.line([tuple(map(int, p1)), tuple(map(int, p2))], fill=fill, width=width)


def _circle(draw: ImageDraw.ImageDraw, p, radius, fill):
    x, y = int(p[0]), int(p[1])
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)


def draw_overlay(image_rgb: np.ndarray, landmarks, ranking: List[Tuple[str, float]]) -> Image.Image:
    img = Image.fromarray(image_rgb).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Contorno viso.
    oval_pts = [lm_xy(landmarks, idx, w, h) for idx in FACE_OVAL]
    oval_pts_int = [tuple(map(int, p)) for p in oval_pts]
    draw.line(oval_pts_int + [oval_pts_int[0]], fill=(0, 210, 120), width=max(3, w // 220))

    # Misure principali.
    pairs = [
        ("left_cheek", "right_cheek"),
        ("left_forehead", "right_forehead"),
        ("left_jaw", "right_jaw"),
        ("top", "chin"),
    ]
    for a, b in pairs:
        p1 = lm_xy(landmarks, LANDMARKS[a], w, h)
        p2 = lm_xy(landmarks, LANDMARKS[b], w, h)
        _line(draw, p1, p2, fill=(255, 180, 0), width=max(3, w // 300))
        _circle(draw, p1, radius=max(4, w // 180), fill=(255, 180, 0))
        _circle(draw, p2, radius=max(4, w // 180), fill=(255, 180, 0))

    # Etichetta risultato.
    label = f"{ranking[0][0]} · compatibilità {ranking[0][1]:.0f}%"
    box_h = max(58, h // 12)
    draw.rectangle((20, 20, min(w - 20, 720), 20 + box_h), fill=(0, 0, 0))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(22, w // 38))
    except Exception:
        font = None
    draw.text((36, 32), label, fill=(255, 255, 255), font=font)
    return img


def quality_message(m: FaceMeasurements) -> str:
    warnings = []
    if abs(m.eye_tilt_degrees) > 7:
        warnings.append("testa inclinata")
    if m.nose_offset > 0.08:
        warnings.append("volto non perfettamente frontale")
    if not warnings:
        return "Scatto valido: volto abbastanza frontale."
    return "Attenzione: " + ", ".join(warnings) + ". Il risultato può cambiare con uno scatto più frontale."


st.title("Face Shape Detector")
st.caption("Rilevamento indicativo della forma geometrica del volto da foto frontale.")

with st.sidebar:
    st.header("Come scattare")
    st.write("• Guarda dritto in camera")
    st.write("• Capelli lontani dal contorno del viso")
    st.write("• Luce uniforme")
    st.write("• Niente filtri beauty o grandangolo troppo vicino")
    st.divider()
    st.write("Forme: ovale, tondo, squadrato, rettangolare, oblungo, cuore, triangolo inverso, triangolo, diamante.")

col1, col2 = st.columns([1, 1])

with col1:
    source = st.radio("Scegli input", ["Scatta con webcam", "Carica immagine"], horizontal=True)
    file = None
    if source == "Scatta con webcam":
        file = st.camera_input("Scatta una foto frontale")
    else:
        file = st.file_uploader("Carica una foto", type=["jpg", "jpeg", "png", "webp"])

if file is not None:
    image = Image.open(file).convert("RGB")
    image_rgb = np.array(image)
    landmarks, analysis, error = analyze_image(image_rgb)

    if error:
        st.error(error)
        st.image(image_rgb, caption="Immagine caricata", use_container_width=True)
    else:
        measurements, ranking = analysis
        overlay = draw_overlay(image_rgb, landmarks, ranking)
        best_shape = ranking[0][0]

        with col1:
            st.image(overlay, caption="Rilevamento volto e misure principali", use_container_width=True)

        with col2:
            st.subheader(f"Risultato: {best_shape}")
            st.success(FACE_GROUPS.get(best_shape, "Forma da valutare"))
            st.write(quality_message(measurements))

            st.metric("Rapporto lunghezza / larghezza", f"{measurements.ratio_length_width:.2f}")
            st.metric("Fronte / zigomi", f"{measurements.forehead_to_cheek:.2f}")
            st.metric("Mascella / zigomi", f"{measurements.jaw_to_cheek:.2f}")
            st.metric("Mento / zigomi", f"{measurements.chin_to_cheek:.2f}")

            st.subheader("Compatibilità forme")
            for name, score in ranking[:5]:
                st.progress(min(score / 100, 1.0), text=f"{name}: {score:.0f}%")

            st.info(
                "Nota: la classificazione è geometrica e indicativa. "
                "Per uso professionale conviene tarare le soglie su foto reali del tuo metodo Face Design."
            )
else:
    st.info("Scatta o carica una foto per iniziare.")
