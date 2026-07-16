import pandas as pd
import glob
import os

# Cari file di direktori saat ini dan di subfolder 'data_mentah'
file_patterns = ['a*.csv', 'data_mentah/a*.csv', 'data_mentah/*.csv']
found_files = []

for pattern in file_patterns:
    found = glob.glob(pattern)
    if found:
        found_files.extend(found)
        break  # Hentikan jika sudah ditemukan di salah satu pola

if not found_files:
    # Jika tetap tidak ditemukan, cari semua file CSV di direktori dan subdirektori
    found_files = glob.glob('**/*.csv', recursive=True)
    # Filter hanya yang namanya mengandung 'a' atau angka
    found_files = [f for f in found_files if 'a' in os.path.basename(f).lower()]

if not found_files:
    print("❌ Tidak ada file CSV dengan pola 'a*.csv' ditemukan.")
    print("   Pastikan file berada di folder yang sama atau di dalam 'data_mentah'.")
    exit()

# Urutkan berdasarkan nama agar urutan a1, a2, ... a11
found_files.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))) or 0))

print(f"🔍 Menemukan {len(found_files)} file:")
for f in found_files:
    print(f"   - {f}")

# Baca dan gabungkan
dataframes = []
for file in found_files:
    try:
        df = pd.read_csv(file)
        dataframes.append(df)
        print(f"✔ {file} → {len(df)} baris")
    except Exception as e:
        print(f"✖ Gagal membaca {file}: {e}")

if dataframes:
    combined = pd.concat(dataframes, ignore_index=True)
    combined.to_csv('gabungan_11_file.csv', index=False)
    print(f"\n✅ Berhasil digabung! Total baris: {len(combined)}")
    print(f"📁 Hasil: gabungan_11_file.csv")
else:
    print("❌ Tidak ada data yang berhasil dibaca.")