
import streamlit as st
import numpy as np
import soundfile as sf
import json
from pathlib import Path
from math import gcd

from scipy.signal import resample_poly, stft
from scipy.ndimage import zoom
from tensorflow.keras.models import load_model

# KONFIGURASI
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "cnn_combo_best.keras"
CLASS_PATH = BASE_DIR / "config" / "class_names.json"

TARGET_SR = 44100
TARGET_DURATION = 2.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)

st.set_page_config(
    page_title="CNN Vehicle Sound Classification",
    page_icon="🚗",
    layout="centered"
)

# LOAD MODEL DAN KELAS
@st.cache_resource
def load_assets():
    model = load_model(MODEL_PATH)

    with open(CLASS_PATH, "r") as f:
        classes = json.load(f)

    if isinstance(classes, dict):
        classes = classes.get("class_names", list(classes.values()))

    return model, classes

model, class_names = load_assets()

# PREPROCESSING AUDIO
def preprocess_audio(audio_file):
    audio, sr = sf.read(
        audio_file,
        dtype="float32",
        always_2d=True
    )

    # Stereo ke mono
    audio = np.mean(audio, axis=1)

    # Resampling
    if sr != TARGET_SR:
        factor = gcd(TARGET_SR, sr)
        audio = resample_poly(
            audio,
            TARGET_SR // factor,
            sr // factor
        )

    # Truncate atau padding
    if len(audio) > TARGET_SAMPLES:
        audio = audio[:TARGET_SAMPLES]
    elif len(audio) < TARGET_SAMPLES:
        audio = np.pad(
            audio,
            (0, TARGET_SAMPLES - len(audio))
        )

    # STFT
    _, _, Zxx = stft(
        audio,
        fs=TARGET_SR,
        window="hann",
        nperseg=2048,
        noverlap=2048 - 512,
        nfft=2048,
        boundary=None,
        padded=False
    )

    spec_db = 20 * np.log10(np.abs(Zxx) + 1e-10)

    # Normalisasi
    spec_min = np.min(spec_db)
    spec_max = np.max(spec_db)

    spec = (spec_db - spec_min) / (
        spec_max - spec_min + 1e-10
    )

    # Resize menjadi 32 x 32
    spec = zoom(
        spec,
        (32 / spec.shape[0], 32 / spec.shape[1]),
        order=1
    )

    spec = spec[:32, :32]

    # Pastikan ukuran tepat 32 x 32
    spec = np.pad(
        spec,
        (
            (0, max(0, 32 - spec.shape[0])),
            (0, max(0, 32 - spec.shape[1]))
        )
    )

    return spec.astype(np.float32)[..., np.newaxis]

# TAMPILAN APLIKASI
st.title("CNN Vehicle Sound Classification")

st.write(
    "Aplikasi klasifikasi jenis kendaraan menggunakan "
    "model CNN Combo."
)

st.info(
    "Upload file audio WAV untuk melakukan klasifikasi kendaraan."
)

uploaded_file = st.file_uploader(
    "Pilih file audio",
    type=["wav"]
)

if uploaded_file is not None:

    st.audio(uploaded_file, format="audio/wav")

    if st.button("Klasifikasikan Audio", type="primary"):

        with st.spinner("Memproses audio..."):

            try:
                uploaded_file.seek(0)
                spec = preprocess_audio(uploaded_file)

                prediction = model.predict(
                    spec[np.newaxis, ...],
                    verbose=0
                )[0]

                pred_idx = int(np.argmax(prediction))
                confidence = float(prediction[pred_idx])

                st.success("Klasifikasi berhasil!")

                st.subheader("Hasil Prediksi")
                st.write("**Jenis kendaraan:**", class_names[pred_idx])
                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )

                st.subheader("Probabilitas Seluruh Kelas")

                results = sorted(
                    zip(class_names, prediction),
                    key=lambda x: x[1],
                    reverse=True
                )

                for label, score in results:
                    st.write(f"{label}: {score * 100:.2f}%")
                    st.progress(float(score))

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
