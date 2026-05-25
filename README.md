---
title: Irigasi Server
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# Server Prediksi Irigasi Otomatis

Server FastAPI untuk prediksi status pompa irigasi menggunakan Decision Tree.

## Cara Pakai

ESP8266 kirim 4 data sensor (suhu/kelembaban udara dan tanah) ke endpoint /prediksi.
Server otomatis membaca histori dari Google Sheets, hitung 9 fitur tambahan,
lalu mengembalikan status pompa (ON/OFF).

## Endpoint

- GET / - info server
- GET /health - health check
- GET /cek-sheets - test koneksi Google Sheets
- POST /prediksi - endpoint utama untuk prediksi
