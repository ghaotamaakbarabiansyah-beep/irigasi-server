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

Server FastAPI untuk prediksi status pompa irigasi otomatis berdasarkan data sensor.
Bagian dari skripsi sistem irigasi IoT + Machine Learning.

**Author:** Ghaotama Akbar Abiansyah (NIM 2205101066)  
**Kampus:** Universitas PGRI Madiun  
**Repo GitHub:** https://github.com/ghaotamaakbarabiansyah-beep/irigasi-server

---

## Cara Kerja

ESP8266 baca 4 data sensor, kirim ke server via HTTP POST.
Server prediksi pakai model Decision Tree, balikin status pompa (ON/OFF).
ESP terima respons, kontrol relay pompa.