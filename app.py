
import streamlit as st
import numpy as np
import soundfile as sf
import json
from pathlib import Path
from math import gcd

from scipy.signal import resample_poly, stft
from scipy.ndimage import zoom
from tensorflow.keras.models import load_model

# =====================================================
# KONFIGURASI
# =====================================================

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

# =====================================================
# LOAD MODEL DAN KELAS
# =====================================================

@st.cache_resource
def load_assets():
    model = load_model(MODEL_PATH)

    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        classes = json.load(f)

    # Mendukung beberapa format class_names.json
    if isinstance(classes, dict):

        if "class_names" in classes:
            classes = classes["class_names"]

        elif all(str(k).isdigit() for k in classes.keys()):
            # Format: {"0": "Bus", "1": "Car", ...}
            classes = [
                name for idx, name in sorted(
                    classes.items(),
                    key=lambda item: int(item[0])
                )
            ]

        elif all(
            isinstance(v, int) for v in classes.values()
        ):
            # Format: {"Bus": 0, "Pickup": 7, ...}
            classes = [
                name for name, idx in sorted(
                    classes.items(),
                    key=lambda item: item[1]
                )
            ]

        else:
            raise ValueError(
                "Format class_names.json tidak dikenali."
            )

    if not isinstance(classes, list):
        raise ValueError(
            "Daftar kelas harus berbentuk list."
        )

    # Validasi jumlah kelas dengan output model
    total_output = int(model.output_shape[-1])

    if len(classes) != total_output:
        raise ValueError(
            f"Jumlah kelas JSON ({len(classes)}) "
            f"tidak sesuai dengan output model "
            f"({total_output})."
        )

    return model, classes


model, class_names = load_assets()

# =====================================================
# PREPROCESSING AUDIO
# =====================================================

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

    spec_db = 20 * np.log10(
        np.abs(Zxx) + 1e-10
    )

    # Normalisasi
    spec_min = np.min(spec_db)
    spec_max = np.max(spec_db)

    spec = (spec_db - spec_min) / (
        spec_max - spec_min + 1e-10
    )

    # Resize menjadi 32 x 32
    spec = zoom(
        spec,
        (
            32 / spec.shape[0],
            32 / spec.shape[1]
        ),
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

    return spec.astype(
        np.float32
    )[..., np.newaxis]


# =====================================================
# TAMPILAN APLIKASI
# =====================================================

st.title("CNN Vehicle Sound Classification")

st.write(
    "Aplikasi klasifikasi jenis kendaraan menggunakan "
    "model CNN Combo."
)

st.markdown(
    "**Developed by Samuel Andrew 56975**"
)

st.caption(
    "Undergraduate Thesis Project | 2026"
)

st.info(
    "Upload file audio WAV untuk melakukan "
    "klasifikasi kendaraan."
)

# =====================================================
# DAFTAR KELAS KENDARAAN (00 - 09)
# =====================================================

st.subheader("Daftar Kelas Kendaraan")

st.write(
    "Model CNN Combo menggunakan 10 kelas kendaraan "
    "dari dataset VISC dan IDMT-Traffic."
)

with st.expander(
    "Lihat Keterangan Kode Kelas 00 - 09",
    expanded=True
):

    class_table = {
        "Kode Kelas": [
            f"{i:02d}"
            for i in range(len(class_names))
        ],
        "Jenis Kendaraan": class_names
    }

    st.table(class_table)

st.caption(
    "Kode kelas menunjukkan indeks output model CNN. "
    "Nama kendaraan mengikuti konfigurasi "
    "class_names.json."
)

st.divider()

# =====================================================
# UPLOAD AUDIO
# =====================================================

uploaded_file = st.file_uploader(
    "Pilih file audio",
    type=["wav"]
)

if uploaded_file is not None:

    st.audio(
        uploaded_file,
        format="audio/wav"
    )

    if st.button(
        "Klasifikasikan Audio",
        type="primary"
    ):

        with st.spinner("Memproses audio..."):

            try:
                uploaded_file.seek(0)

                spec = preprocess_audio(
                    uploaded_file
                )

                prediction = model.predict(
                    spec[np.newaxis, ...],
                    verbose=0
                )[0]

                pred_idx = int(
                    np.argmax(prediction)
                )

                confidence = float(
                    prediction[pred_idx]
                )

                predicted_class = class_names[
                    pred_idx
                ]

                # =================================================
                # HASIL PREDIKSI
                # =================================================

                st.success(
                    "Klasifikasi berhasil!"
                )

                st.subheader("Hasil Prediksi")

                st.write(
                    "**Jenis kendaraan:**",
                    predicted_class
                )

                st.write(
                    "**Kode kelas:**",
                    f"{pred_idx:02d}"
                )

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )

                # =================================================
                # PROBABILITAS SELURUH KELAS
                # =================================================

                st.subheader(
                    "Probabilitas Seluruh Kelas"
                )

                results = sorted(
                    [
                        (
                            idx,
                            class_names[idx],
                            float(prediction[idx])
                        )
                        for idx in range(
                            len(class_names)
                        )
                    ],
                    key=lambda x: x[2],
                    reverse=True
                )

                for idx, label, score in results:

                    st.write(
                        f"**{idx:02d} - {label}**: "
                        f"{score * 100:.2f}%"
                    )

                    st.progress(
                        float(score)
                    )

            except Exception as e:

                st.error(
                    f"Terjadi kesalahan: {e}"
                )
