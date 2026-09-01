"""
Fetch seluruh data AC (Inverter + Non-Inverter) dari SIMEBTKE Kementerian ESDM.
Source: https://simebtke.esdm.go.id/sinergi/skem-label/konsumen/pengondisi-udara-ac
Endpoint AJAX: /konsumen/ajax/pengondisi-udara-ac/field-3/{type}/-/-/-/-
"""
import json
import time
import urllib.request
import urllib.parse
import os

BASE_URL = "https://simebtke.esdm.go.id/sinergi/skem-label/konsumen/ajax/pengondisi-udara-ac/field-3"
AC_TYPES = ["Inverter", "Non-Inverter"]
PAGE_SIZE = 500

# Mapping field JSON -> nama kolom sesuai header tabel
COL_MAP = {
    "id": "NO.",
    "field-0": "Merek",
    "field-1": "Famili",
    "field-2": "Model",
    "field-3": "Tipe",
    "field-4": "Daya (watt)",
    "field-5": "Kapasitas Pendinginan (BTU/h)",
    "field-6": "Nilai Efisiensi (EER/CSPF)",
    "field-7": "Rating Bintang (1-5)",
    "field-8": "Konsumsi Energi Tahunan (kWh)",
    "field-9": "Biaya Listrik Tahunan (Rp)",
    "field-10": "No. Registrasi/No. SHE",
    "field-14": "Tanggal Terbit SHE",
    "field-15": "SHE Berlaku Sampai Dengan Tanggal",
    "nama_lspro": "LSPro",
}

os.makedirs("data/raw", exist_ok=True)

all_rows = []
for ac_type in AC_TYPES:
    offset = 0
    page = 1
    while True:
        params = urllib.parse.urlencode({
            "limit": PAGE_SIZE,
            "offset": offset,
            "page": page,
            "search": "",
        })
        url = f"{BASE_URL}/{ac_type}/-/-/-/-?{params}"
        req = urllib.request.Request(url, headers={
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "X-Requested-With": "XMLHttpRequest",
        })
        with urllib.request.urlopen(req) as resp:
            raw = resp.read().decode("utf-8")
        # Server kadang mengembalikan PHP notice sebelum JSON, jadi ambil bagian JSON saja
        json_start = raw.find('{"rows"')
        if json_start == -1:
            print(f"  ERROR: JSON tidak ditemukan untuk {ac_type} page {page}")
            break
        data = json.loads(raw[json_start:])
        rows = data.get("rows", [])
        total = data.get("total", 0)
        all_rows.extend(rows)
        print(f"  {ac_type}: page {page} -> {len(rows)} rows (total so far: {total})")
        if offset + len(rows) >= total or len(rows) == 0:
            break
        offset += PAGE_SIZE
        page += 1
        time.sleep(0.5)  # sopan ke server

print(f"\nTotal record fetched: {len(all_rows)}")

# Rename kolom
renamed = []
for row in all_rows:
    renamed.append({COL_MAP.get(k, k): v for k, v in row.items()})

# Simpan JSON raw
with open("data/raw/ac_simebtke_raw.json", "w", encoding="utf-8") as f:
    json.dump(renamed, f, ensure_ascii=False, indent=2)

# Simpan CSV
import csv
if renamed:
    fieldnames = list(renamed[0].keys())
    with open("data/raw/ac_simebtke_raw.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(renamed)

print(f"Disimpan ke data/raw/ac_simebtke_raw.json dan .csv")
