from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import json
import os
from datetime import datetime

# ── Load model & komponen ─────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    model = joblib.load(os.path.join(BASE_DIR, "model_irigasi.pkl"))
    le    = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
    with open(os.path.join(BASE_DIR, "fitur_model.json")) as f:
        FITUR = json.load(f)
    print("✅ Model berhasil dimuat!")
    print(f"   Fitur: {FITUR}")
except FileNotFoundError as e:
    raise RuntimeError(
        f"❌ File model tidak ditemukan: {e}\n"
        "   Pastikan model_irigasi.pkl, label_encoder.pkl, dan fitur_model.json "
        "berada di folder yang sama dengan main.py"
    )

# ── Aplikasi FastAPI ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Server ML Irigasi",
    description="Prediksi status pompa irigasi menggunakan Decision Tree",
    version="1.0.0"
)

# ── Schema input (dari ESP8266 / sensor) ─────────────────────────────────────
class SensorData(BaseModel):
    # Data sensor real-time
    suhu_udara              : float
    kelembaban_udara        : float
    kelembaban_tanah        : float
    suhu_tanah              : float

    # Fitur histori (dihitung di sisi klien / ESP8266)
    durasi_pompa_sebelumnya : float   # Durasi sesi ON terakhir (dari Log Irigasi)
    menit_sejak_on_terakhir : float   # Menit sejak pompa terakhir menyala
    kelembaban_tanah_trend  : float   # Kelembaban tanah - rolling mean 30 data
    suhu_tanah_trend        : float   # Suhu tanah - rolling mean 30 data
    delta_kelembaban        : float   # Kelembaban udara - kelembaban tanah
    delta_suhu_tanah        : float   # Perubahan suhu tanah dari data sebelumnya
    selisih_suhu            : float   # Suhu udara - suhu tanah
    indeks_kering           : float   # Suhu tanah * (1 - kelembaban tanah / 100)
    kelembaban_udara_trend  : float   # Kelembaban udara - rolling mean 30 data

# ── Halaman utama ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message" : "Server ML Irigasi Aktif",
        "version" : "1.0.0",
        "endpoint": "POST /prediksi"
    }

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status" : "ok",
        "model"  : type(model).__name__,
        "fitur"  : len(FITUR),
        "waktu"  : datetime.now().isoformat()
    }

# ── Endpoint prediksi ─────────────────────────────────────────────────────────
@app.post("/prediksi")
def prediksi(data: SensorData):
    try:
        # Susun dict sesuai nama kolom yang dipakai saat training
        row = {
            "Suhu Tanah"              : data.suhu_tanah,
            "Kelembaban Tanah"        : data.kelembaban_tanah,
            "Suhu Udara"              : data.suhu_udara,
            "Kelembaban Udara"        : data.kelembaban_udara,
            "Durasi Pompa Sebelumnya" : data.durasi_pompa_sebelumnya,
            "Menit Sejak ON Terakhir" : data.menit_sejak_on_terakhir,
            "Kelembaban Tanah Trend"  : data.kelembaban_tanah_trend,
            "Suhu Tanah Trend"        : data.suhu_tanah_trend,
            "Delta Kelembaban"        : data.delta_kelembaban,
            "Delta Suhu Tanah"        : data.delta_suhu_tanah,
            "Selisih Suhu"            : data.selisih_suhu,
            "Indeks Kering"           : data.indeks_kering,
            "Kelembaban Udara Trend"  : data.kelembaban_udara_trend,
        }

        # Buat DataFrame dengan urutan fitur sesuai training
        df_input = pd.DataFrame([row])[FITUR]

        # Prediksi
        pred_code  = model.predict(df_input)[0]
        pred_label = le.inverse_transform([pred_code])[0]
        proba      = model.predict_proba(df_input)[0].tolist()

        return {
            "status_pompa" : pred_label,                  # "ON" atau "OFF"
            "proba_off"    : round(proba[0], 4),
            "proba_on"     : round(proba[1], 4),
            "waktu"        : datetime.now().isoformat(),

            # Echo balik data sensor untuk logging / debugging
            "sensor": {
                "suhu_tanah"       : data.suhu_tanah,
                "kelembaban_tanah" : data.kelembaban_tanah,
                "suhu_udara"       : data.suhu_udara,
                "kelembaban_udara" : data.kelembaban_udara,
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediksi: {str(e)}")