from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import json
import os
import requests
from datetime import datetime
from dateutil import parser as date_parser

# ── Load model & komponen ─────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    model = joblib.load(os.path.join(BASE_DIR, "model_irigasi.pkl"))
    le    = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
    with open(os.path.join(BASE_DIR, "fitur_model.json")) as f:
        FITUR = json.load(f)
    print("✅ Model berhasil dimuat!")
    print(f"   Tipe Model: {type(model).__name__}")
    print(f"   Jumlah Fitur: {len(FITUR)}")
except FileNotFoundError as e:
    raise RuntimeError(
        f"❌ File model tidak ditemukan: {e}\n"
        "   Pastikan model_irigasi.pkl, label_encoder.pkl, dan fitur_model.json "
        "berada di folder yang sama dengan main.py"
    )

# ── Konfigurasi Apps Script ───────────────────────────────────────────────────
# URL Apps Script bisa di-override lewat environment variable (untuk Render)
APPS_SCRIPT_URL = os.environ.get(
    "APPS_SCRIPT_URL",
    "https://script.google.com/macros/s/AKfycbxyruti3HVePIXDTYEyk4-5o8dfcFiF6F6JzObk7BIAYzAjDATcuTG6SQbWMvvoQDbJ/exec"
)

# ── Aplikasi FastAPI ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Server ML Irigasi",
    description="Prediksi status pompa irigasi - membaca histori dari Google Sheets",
    version="2.0.0"
)

# ── Schema input dari ESP8266 (HANYA 4 DATA SENSOR MENTAH) ───────────────────
class SensorData(BaseModel):
    suhu_udara       : float
    kelembaban_udara : float
    suhu_tanah       : float
    kelembaban_tanah : float

# ── Fungsi: Baca histori dari Google Sheets via Apps Script ──────────────────
def baca_histori_dari_sheets():
    """
    Panggil Apps Script untuk dapat 30 data sensor terakhir + log pompa terakhir.
    Return: dict berisi data_sensor (list) dan log_pompa_terakhir (dict atau None)
    """
    try:
        response = requests.get(
            f"{APPS_SCRIPT_URL}?type=baca_histori",
            timeout=15
        )
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            raise RuntimeError(f"Apps Script error: {data['error']}")

        return data

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Timeout saat baca Google Sheets")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Gagal akses Google Sheets: {str(e)}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="Response Apps Script bukan JSON valid")

# ── Fungsi: Hitung 9 fitur histori dari data sensor sekarang + histori ───────
def hitung_fitur_histori(data_sekarang: SensorData, histori: dict):
    """
    Hitung 9 fitur histori yang dibutuhkan model.
    Input:
      - data_sekarang: data sensor real-time dari ESP
      - histori: dict hasil baca_histori_dari_sheets()
    Return: dict berisi 9 fitur histori
    """
    data_sensor = histori.get("data_sensor", [])
    log_pompa   = histori.get("log_pompa_terakhir")

    # ─── Fitur sederhana (tidak butuh histori) ───────────────────────────────
    delta_kelembaban  = data_sekarang.kelembaban_udara - data_sekarang.kelembaban_tanah
    selisih_suhu      = data_sekarang.suhu_udara - data_sekarang.suhu_tanah
    indeks_kering     = data_sekarang.suhu_tanah * (1 - data_sekarang.kelembaban_tanah / 100)

    # ─── Fitur trend (butuh rolling mean 30 data) ────────────────────────────
    if len(data_sensor) > 0:
        df_hist = pd.DataFrame(data_sensor)

        mean_kelembaban_tanah = df_hist["kelembaban_tanah"].mean()
        mean_suhu_tanah       = df_hist["suhu_tanah"].mean()
        mean_kelembaban_udara = df_hist["kelembaban_udara"].mean()

        kelembaban_tanah_trend = data_sekarang.kelembaban_tanah - mean_kelembaban_tanah
        suhu_tanah_trend       = data_sekarang.suhu_tanah - mean_suhu_tanah
        kelembaban_udara_trend = data_sekarang.kelembaban_udara - mean_kelembaban_udara

        # Delta suhu tanah = sekarang - data sebelumnya (paling terakhir di histori)
        suhu_tanah_sebelumnya = df_hist["suhu_tanah"].iloc[-1]
        delta_suhu_tanah      = data_sekarang.suhu_tanah - suhu_tanah_sebelumnya
    else:
        # Belum ada histori (data pertama) → set ke 0
        kelembaban_tanah_trend = 0.0
        suhu_tanah_trend       = 0.0
        kelembaban_udara_trend = 0.0
        delta_suhu_tanah       = 0.0

    # ─── Fitur dari Log Irigasi ──────────────────────────────────────────────
    if log_pompa is not None:
        durasi_pompa_sebelumnya = float(log_pompa.get("durasi", 0))

        # Hitung menit sejak pompa terakhir ON
        try:
            waktu_pompa_terakhir = date_parser.parse(log_pompa["waktu"])
            selisih_detik = (datetime.now() - waktu_pompa_terakhir).total_seconds()
            menit_sejak_on_terakhir = max(0, selisih_detik / 60)
        except Exception:
            menit_sejak_on_terakhir = 0.0
    else:
        # Pompa belum pernah ON
        durasi_pompa_sebelumnya = 0.0
        menit_sejak_on_terakhir = 0.0

    return {
        "durasi_pompa_sebelumnya" : float(durasi_pompa_sebelumnya),
        "menit_sejak_on_terakhir" : float(menit_sejak_on_terakhir),
        "kelembaban_tanah_trend"  : float(kelembaban_tanah_trend),
        "suhu_tanah_trend"        : float(suhu_tanah_trend),
        "delta_kelembaban"        : float(delta_kelembaban),
        "delta_suhu_tanah"        : float(delta_suhu_tanah),
        "selisih_suhu"            : float(selisih_suhu),
        "indeks_kering"           : float(indeks_kering),
        "kelembaban_udara_trend"  : float(kelembaban_udara_trend),
    }

# ── Halaman utama ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message" : "Server ML Irigasi Aktif",
        "version" : "2.0.0",
        "endpoint": "POST /prediksi",
        "info"    : "ESP cukup kirim 4 data sensor, fitur histori dihitung otomatis dari Google Sheets"
    }

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status"      : "ok",
        "model"       : type(model).__name__,
        "jumlah_fitur": len(FITUR),
        "waktu"       : datetime.now().isoformat()
    }

# ── Endpoint cek koneksi ke Google Sheets (untuk debugging) ──────────────────
@app.get("/cek-sheets")
def cek_sheets():
    """Endpoint untuk test apakah koneksi ke Apps Script jalan."""
    try:
        data = baca_histori_dari_sheets()
        return {
            "status"             : "ok",
            "jumlah_data_sensor" : data.get("jumlah_data", 0),
            "ada_log_pompa"      : data.get("log_pompa_terakhir") is not None,
        }
    except HTTPException as e:
        return {"status": "error", "detail": e.detail}

# ── Endpoint prediksi utama ───────────────────────────────────────────────────
@app.post("/prediksi")
def prediksi(data: SensorData):
    try:
        # 1. Baca histori dari Google Sheets
        histori = baca_histori_dari_sheets()

        # 2. Hitung 9 fitur histori
        fitur_histori = hitung_fitur_histori(data, histori)

        # 3. Susun dict lengkap dengan urutan kolom sesuai training
        row = {
            "Suhu Tanah"              : data.suhu_tanah,
            "Kelembaban Tanah"        : data.kelembaban_tanah,
            "Suhu Udara"              : data.suhu_udara,
            "Kelembaban Udara"        : data.kelembaban_udara,
            "Durasi Pompa Sebelumnya" : fitur_histori["durasi_pompa_sebelumnya"],
            "Menit Sejak ON Terakhir" : fitur_histori["menit_sejak_on_terakhir"],
            "Kelembaban Tanah Trend"  : fitur_histori["kelembaban_tanah_trend"],
            "Suhu Tanah Trend"        : fitur_histori["suhu_tanah_trend"],
            "Delta Kelembaban"        : fitur_histori["delta_kelembaban"],
            "Delta Suhu Tanah"        : fitur_histori["delta_suhu_tanah"],
            "Selisih Suhu"            : fitur_histori["selisih_suhu"],
            "Indeks Kering"           : fitur_histori["indeks_kering"],
            "Kelembaban Udara Trend"  : fitur_histori["kelembaban_udara_trend"],
        }

        # 4. Buat DataFrame dengan urutan fitur sesuai training
        df_input = pd.DataFrame([row])[FITUR]

        # 5. Prediksi
        pred_code  = model.predict(df_input)[0]
        pred_label = le.inverse_transform([pred_code])[0]
        proba      = model.predict_proba(df_input)[0].tolist()

        return {
            "status_pompa" : pred_label,
            "proba_off"    : round(proba[0], 4),
            "proba_on"     : round(proba[1], 4),
            "waktu"        : datetime.now().isoformat(),

            # Echo data sensor + fitur histori untuk debugging
            "sensor_input": {
                "suhu_udara"       : data.suhu_udara,
                "kelembaban_udara" : data.kelembaban_udara,
                "suhu_tanah"       : data.suhu_tanah,
                "kelembaban_tanah" : data.kelembaban_tanah,
            },
            "fitur_histori_dihitung": fitur_histori,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediksi: {str(e)}")