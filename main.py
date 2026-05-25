from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import json
import os
from datetime import datetime

# Load model & komponen
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    model = joblib.load(os.path.join(BASE_DIR, "model_irigasi.pkl"))
    le    = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
    with open(os.path.join(BASE_DIR, "fitur_model.json")) as f:
        FITUR_INFO = json.load(f)
    FITUR = FITUR_INFO["features"]
    print("Model berhasil dimuat!")
    print(f"   Tipe Model    : {type(model).__name__}")
    print(f"   Jumlah Fitur  : {len(FITUR)}")
    print(f"   Daftar Fitur  : {FITUR}")
except FileNotFoundError as e:
    raise RuntimeError(
        f"File model tidak ditemukan: {e}\n"
        "Pastikan model_irigasi.pkl, label_encoder.pkl, dan fitur_model.json "
        "berada di folder yang sama dengan main.py"
    )

# Aplikasi FastAPI
app = FastAPI(
    title="Server ML Irigasi",
    description="Prediksi status pompa irigasi berdasarkan 4 fitur sensor (Stuard Tomato dataset)",
    version="3.0.0"
)

# Schema input dari ESP8266 (4 data sensor)
class SensorData(BaseModel):
    suhu_udara       : float
    kelembaban_udara : float
    suhu_tanah       : float
    kelembaban_tanah : float

# Halaman utama
@app.get("/")
def root():
    return {
        "message" : "Server ML Irigasi Aktif",
        "version" : "3.0.0",
        "endpoint": "POST /prediksi",
        "fitur"   : FITUR,
        "info"    : "ESP cukup kirim 4 data sensor: suhu_udara, kelembaban_udara, suhu_tanah, kelembaban_tanah"
    }

# Health check
@app.get("/health")
def health():
    return {
        "status"      : "ok",
        "model"       : type(model).__name__,
        "jumlah_fitur": len(FITUR),
        "fitur"       : FITUR,
        "waktu"       : datetime.now().isoformat()
    }

# Endpoint prediksi utama
@app.post("/prediksi")
def prediksi(data: SensorData):
    try:
        # Susun input dengan mapping nama ESP -> nama fitur model
        row = {
            "air_temperature"  : data.suhu_udara,
            "air_humidity"     : data.kelembaban_udara,
            "soil_temperature" : data.suhu_tanah,
            "soil_humidity"    : data.kelembaban_tanah
        }

        # Buat DataFrame dengan urutan fitur sesuai training
        df_input = pd.DataFrame([row])[FITUR]

        # Prediksi
        pred_code  = model.predict(df_input)[0]
        pred_label = le.inverse_transform([pred_code])[0]
        proba      = model.predict_proba(df_input)[0].tolist()

        return {
            "status_pompa" : pred_label,
            "proba_off"    : round(proba[0], 4),
            "proba_on"     : round(proba[1], 4),
            "waktu"        : datetime.now().isoformat(),

            "sensor_input": {
                "suhu_udara"       : data.suhu_udara,
                "kelembaban_udara" : data.kelembaban_udara,
                "suhu_tanah"       : data.suhu_tanah,
                "kelembaban_tanah" : data.kelembaban_tanah,
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediksi: {str(e)}")