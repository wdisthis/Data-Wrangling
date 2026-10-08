# Soal Takeaways Pertemuan 4 — Latihan 1–5

*Data Wrangling – Pertemuan 4: Penanganan Missing Data*

---

## Latihan 1 — Forensic missing-data audit (1/2)

**E1** `L1_survei_kerja_raw.csv`

| id | usia | prodi | status_kerja | jam | pendapatan | berat |
|---|---|---|---|---|---|---|
| 1 | 19 | Sains Data | tidak bekerja | | 0 | 52 |
| 2 | 20 | Sains Data | bekerja | 12 | 1500000 | 61 |
| 3 | 99 | Matematika | bekerja | 20 | 2000000 | 58 |
| 4 | 21 | | tidak bekerja | -1 | 0 | 999 |
| 5 | | Fisika | Unknown | | | - |
| 6 | 22 | Unknown | bekerja | 0 | 1200000 | 70 |
| 7 | 18 | Fisika | bekerja | 15 | N/A | 64 |
| 8 | 23 | Matematika | tidak bekerja | | - | 55 |
| 9 | 20 | Sains Data | bekerja | 999 | 2500000 | 0 |
| 10 | 19 | Kimia | tidak bekerja | | | 49 |
| 11 | 21 | Kimia | bekerja | 10 | 0 | 57 |
| 12 | 20 | Fisika | - | | | 62 |

**E2** Kamus data parsial: `usia` dari dropdown 17–99; `status_kerja` {bekerja, tidak bekerja}; `jam_kerja_minggu` diisi jika bekerja; `pendapatan_bulanan` rupiah, 0 = tidak berpenghasilan; `berat_kg` timbang mandiri. Kode `-1` dan `999` tidak didokumentasikan.

**E3** Teknisi: ekspor dari sistem lama "kadang" mengganti sel numerik kosong dengan `-1`; kolom mana saja tidak tercatat.

**E4** Tujuan: proporsi mahasiswa bekerja dan rata-rata jam kerja per prodi.

*Sumber: data simulasi pembelajaran; berkas tersedia di `data/latihan/`.*

## Latihan 1 — Forensic missing-data audit (2/2)

### TUGAS

1. Klasifikasikan setiap representasi (sel kosong, `-`, `N/A`, `Unknown`, `-1`, `999`, `0`, `99`) menjadi: missing, nilai valid, kosong struktural, atau belum dapat diputuskan.
2. Susun data dictionary mini yang memperbaiki E2.
3. Hitung missing profile per kolom setelah klasifikasi Anda, lalu bandingkan dengan profil jika hanya NaN bawaan pandas yang dihitung.
4. Jelaskan dampak klasifikasi Anda terhadap tujuan pada E4.

### Deliverables

missing-value map; tabel keputusan per representasi beserta evidence; assumption log; minimal lima pertanyaan untuk pemilik data; dan justifikasi. Informasi yang tersedia sengaja tidak cukup untuk memutuskan semuanya secara otomatis.

*Sumber: Sitinjak dkk., Subbab 4.2 (representasi eksplisit, implisit, struktural) dan 4.2.1; McKinney (2022), Subbab 6.1 (`na_values`).*

---

## Latihan 2 — Mechanism investigation (1/2)

Survei kesejahteraan mahasiswa (N = 480, simulasi). Skor stres (0–40) kosong pada 22% responden.

| Angkatan | 2025 | 2024 | 2023 | 2022 |
|---|---|---|---|---|
| **E1** Missing | 31% | 24% | 17% | 12% |
| **E2** Rata-rata skor | 24.1 | 22.8 | 21.9 | 23.5 |

E2 hanya dari responden yang mengisi.

**E3** Missing menurut jenis kelamin: L 21%, P 23%.

**E4** Item stres ada di halaman 4 dari 4. Sebanyak 18% responden berhenti sebelum halaman 4. Angkatan 2025 mengisi pada pekan UTS.

**E5** Tindak lanjut telepon pada 20 non-responden angkatan 2025: 9 bersedia menjawab, 6 di antaranya berskor ≥ 28.

*Sumber: kasus ilustrasi; angka simulasi untuk pembelajaran.*

## Latihan 2 — Mechanism investigation (2/2)

### TUGAS

Jangan hanya menulis satu label. Bangun argumen.

| Evidence | Mendukung | Melemahkan | Belum diketahui |
|---|---|---|---|
| E1 … E5 | hipotesis mana? | hipotesis mana? | apa yang tidak dapat disimpulkan? |

1. Rumuskan hipotesis mekanisme (boleh campuran) dan tingkat keyakinan Anda (rendah/sedang/tinggi) beserta alasannya.
2. Evaluasi kekuatan E5 sebagai evidence (ukuran, siapa yang bersedia menjawab).
3. Tentukan evidence tambahan yang dibutuhkan.
4. Jelaskan risiko jika hipotesis Anda salah terhadap estimasi rata-rata stres per angkatan.

*Sumber: Sitinjak dkk., Subbab 4.2.1.1–4.2.1.3, 4.2.2 (uji hubungan missingness), Latihan 4.6.1; Rattenbury dkk. (2017), Bab 2.*

---

## Latihan 3 — Treatment decision under constraints (1/2)

Survei pengeluaran bulanan mahasiswa (N = 150, simulasi). Tujuan: rata-rata pengeluaran per fakultas untuk usulan subsidi.

**E1** Missing profile

| Atribut | Missing | % |
|---|---|---|
| pengeluaran (juta Rp) | 24 | 16.0 |
| jarak_kos_km | 20 | 13.3 |
| uang_saku (juta Rp) | 8 | 5.3 |
| beasiswa | 1 | 0.7 |
| fakultas | 0 | 0.0 |

**E2** Missing pengeluaran: FS 17/62, FTI 4/55, FTIK 3/33.

**E3** pengeluaran teramati: min 0.81, Q1 1.75, median 2.00, Q3 2.38, maks 4.57; mean 2.10.

**E4** Korelasi dengan pengeluaran: uang_saku 0.80, jarak_kos_km 0.08.

**E5** Rata-rata uang_saku: 1.69 bila pengeluaran terisi, 2.30 bila kosong. Missing pada penerima beasiswa 23.4%, bukan penerima 12.7%.

*Sumber: data simulasi `data/latihan/L3_pengeluaran_mahasiswa.csv`.*

## Latihan 3 — Treatment decision under constraints (2/2)

### CONSTRAINT

- Data tidak dapat dikumpulkan ulang.
- pengeluaran adalah variabel utama kebijakan.
- FTIK hanya 33 responden.
- Laporan harus menyebut ketidakpastian.

### TUGAS

Bandingkan minimal deletion, simple imputation, dan advanced imputation (KNN atau MICE). Jalankan ketiganya pada berkas data, lalu susun decision matrix:

| Strategy | Evidence | Benefit | Risk | Assumption | Consequence | Decision |
|---|---|---|---|---|---|---|
| | | | | | | |

Jelaskan mengapa dua strategi lain ditolak, dan hasil fakultas mana yang paling sensitif terhadap pilihan tersebut.

*Sumber: Sitinjak dkk., Subbab 4.3–4.5, Latihan 4.6.2–4.6.4.*

---

## Latihan 4 — Conflicting cleaned datasets

Tiga analis mengolah berkas Latihan 3 dengan cara berbeda.

| | Treatment pengeluaran | n | Mean | Median | SD | FS | FTI / FTIK |
|---|---|---|---|---|---|---|---|
| A | deletion baris kosong | 126 | 2.10 | 2.00 | 0.57 | 2.10 | 2.08 / 2.13 |
| B | median imputation | 150 | 2.08 | 2.00 | 0.52 | 2.07 | 2.07 / 2.12 |
| C | MICE (5 dataset) | 150 | 2.18 | 2.08 | 0.60 | 2.26 | 2.11 / 2.13 |

C: rata-rata dari 5 dataset (rentang mean keseluruhan 2.16–2.19; FS 2.24–2.28). Evidence E1–E5 Latihan 3 tetap berlaku.

### Pertanyaan

1. Informasi apa yang berubah antarhasil?
2. Di mana potensi distorsi dan apa penyebabnya?
3. Keputusan mana yang paling dapat dipertanggungjawabkan untuk usulan subsidi per fakultas, dan apa batas kesimpulannya?
4. Dalam kondisi apa keputusan itu harus berubah?

*Sumber: hasil eksekusi pandas 3.0.6 dan scikit-learn 1.9.1 (IterativeImputer, `sample_posterior=True`) pada data simulasi.*

---

## Latihan 5 — Missing-data decision memo (1/2)

Layanan konseling kampus ingin melaporkan proporsi klien yang membaik (skor PHQ-9 turun ≥ 5) pada 2025. Data: 620 klien (simulasi).

**E1** Missing: phq9_awal 2%, phq9_akhir 38%, alasan_rujukan 24%, jenis_layanan dan jumlah_sesi 0%.

**E2** phq9_akhir kosong: 15% pada klien "selesai" (n = 380), 74% pada klien "berhenti" (n = 240).

**E3** Rata-rata phq9_awal: 11.0 bila skor akhir ada, 14.2 bila kosong.

**E4** Konselor: sebagian klien berhenti karena merasa membaik, sebagian karena memburuk atau pindah layanan. Alasan berhenti tidak pernah dicatat.

**E5** alasan_rujukan Januari–Maret hilang total akibat migrasi sistem; tidak dapat dipulihkan.

**E6** Laporan dalam 2 minggu; klien tidak boleh dihubungi ulang; pimpinan meminta "satu angka".

*Sumber: kasus ilustrasi; angka simulasi untuk pembelajaran.*

## Latihan 5 — Missing-data decision memo (2/2)

### MEMO MAKSIMAL ±500 KATA

1. masalah;
2. evidence (rujuk E1–E6);
3. dugaan mekanisme missing;
4. treatment yang dipilih;
5. alternatif yang ditolak;
6. alasan;
7. risiko;
8. assumption log;
9. validasi yang harus dilakukan;
10. kondisi yang akan mengubah keputusan Anda.

Anda tidak dinilai dari apakah memilih metode yang sama dengan dosen. Anda dinilai dari apakah keputusan Anda dapat ditelusuri, dipertanggungjawabkan, dan konsisten dengan evidence.

Tanggapi juga permintaan "satu angka" pada E6: dapatkah itu dipenuhi secara jujur?

*Sumber: Sitinjak dkk., Subbab 4.2.1.3 (analisis sensitivitas), 4.4, 4.5, Latihan 4.6.7; Subbab 1.5.1 (dokumentasi).*
