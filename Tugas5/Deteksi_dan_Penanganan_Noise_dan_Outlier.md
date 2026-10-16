# Latihan dan Studi Kasus: Produksi Lini Extruder Plastik

Kode di bawah dibuat agar **satu blok kode = satu sel Jupyter**. Jalankan berurutan dari Sel 0, karena hasil latihan sebelumnya dipakai di latihan berikutnya. Semua angka pada bagian **Interpretasi** berasal dari hasil menjalankan kode ini pada berkas lengkap (150 batch).

---

## Sel 0: Persiapan dan Pemeriksaan Data

Ubah path CSV sesuai lokasi berkas Anda.

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 140)

df = pd.read_csv("L_produksi_extruder.csv")   # ganti path jika perlu
kolom = ["suhu_barrel", "tekanan_bar", "laju_kg_jam", "cacat_persen"]

print(df.shape)
print(df.isna().sum())
df[kolom].describe().round(2)
```

**Interpretasi.** Data berisi 150 batch dan 5 kolom. Nilai hilang hanya ada di `cacat_persen` (3 batch). Pada `tekanan_bar` nilai maksimum 450 dan pada `laju_kg_jam` 1210, padahal median keduanya sekitar 44,6 dan 121,0. Jadi ada nilai yang sangat jauh dari sebaran data lainnya.

---

## Latihan 1: Menandai Kandidat dengan Batas IQR

### 1.1 Hitung Q1, Q3, IQR, dan batas (pengali 1,5)

```python
rows = []
for kol in kolom:
    q1, q3 = df[kol].quantile([0.25, 0.75])
    iqr = q3 - q1
    rows.append({
        "kolom": kol, "Q1": q1, "Q3": q3, "IQR": iqr,
        "bawah": q1 - 1.5 * iqr, "atas": q3 + 1.5 * iqr,
    })

batas = pd.DataFrame(rows).set_index("kolom")
batas.round(2)
```

**Interpretasi.** Hasilnya cocok dengan E3:

| Kolom | Q1 | Q3 | IQR | Batas bawah | Batas atas |
|---|---|---|---|---|---|
| suhu_barrel | 179,00 | 184,375 | 5,375 | 170,9 | 192,4 |
| tekanan_bar | 42,80 | 47,00 | 4,20 | 36,5 | 53,3 |
| laju_kg_jam | 106,40 | 135,625 | 29,225 | 62,6 | 179,5 |
| cacat_persen | 1,39 | 2,565 | 1,175 | -0,4 | 4,3 |

Batas bawah `cacat_persen` bernilai negatif (-0,37). Persentase cacat tidak mungkin negatif, jadi batas bawah itu tidak bermakna secara fisik dan tidak akan pernah menandai apa pun.

### 1.2 Kolom di luar batas pada 11 batch potongan data

```python
ids_potongan = [3, 12, 23, 45, 61, 88, 89, 90, 91, 104, 132]
potongan = df[df["id"].isin(ids_potongan)]

hasil = []
for _, r in potongan.iterrows():
    for kol in kolom:
        v = r[kol]
        if pd.isna(v):
            continue
        if v > batas.loc[kol, "atas"]:
            hasil.append({"id": int(r["id"]), "kolom": kol, "nilai": v,
                          "batas": round(batas.loc[kol, "atas"], 1), "arah": ">"})
        elif v < batas.loc[kol, "bawah"]:
            hasil.append({"id": int(r["id"]), "kolom": kol, "nilai": v,
                          "batas": round(batas.loc[kol, "bawah"], 1), "arah": "<"})

pd.DataFrame(hasil)
```

**Interpretasi.** Dari 11 batch, lima batch memiliki nilai di luar batas:

| id | Kolom di luar batas | Nilai |
|---|---|---|
| 23 | tekanan_bar | 450,0 > 53,3 |
| 45 | suhu_barrel | 193,5 > 192,4 |
| 61 | laju_kg_jam | 1210,0 > 179,5 |
| 104 | suhu_barrel | 196,4 > 192,4 |
| 104 | cacat_persen | 9,80 > 4,3 |
| 132 | tekanan_bar | 33,0 < 36,5 |

Batch 104 menjadi satu-satunya batch dengan dua kolom bermasalah sekaligus. Batch 12 tidak tertandai karena `cacat_persen`-nya kosong (NaN), padahal itu juga masalah kualitas data. Batch 88-91 tidak tertandai sama sekali (lihat Latihan 6).

### 1.3 Jumlah batch tertandai pada berkas lengkap, dan 1.4 ulangi dengan pengali 3

```python
def tandai_iqr(data, k=1.5):
    flag = pd.DataFrame(index=data.index)
    for kol in kolom:
        q1, q3 = data[kol].quantile([0.25, 0.75])
        iqr = q3 - q1
        flag[kol] = (data[kol] < q1 - k * iqr) | (data[kol] > q3 + k * iqr)
    return flag

flag15 = tandai_iqr(df, 1.5)
flag30 = tandai_iqr(df, 3)

for nama, f in [("1,5", flag15), ("3", flag30)]:
    print(f"--- pengali {nama} ---")
    for kol in kolom:
        print(f"{kol:13s} n={f[kol].sum():2d}  id={df.loc[f[kol], 'id'].tolist()}")
    print("batch unik tertandai:", df.loc[f.any(axis=1), "id"].tolist())
```

**Interpretasi.**

| Kolom | Pengali 1,5 | Pengali 3 |
|---|---|---|
| suhu_barrel | 5 batch (45, 104, 105, 138, 147) | 0 batch |
| tekanan_bar | 3 batch (23, 43, 132) | 1 batch (23) |
| laju_kg_jam | 4 batch (51, 61, 85, 103) | 1 batch (61) |
| cacat_persen | 1 batch (104) | 1 batch (104) |

Pengali 1,5 menandai 12 batch unik, sedangkan pengali 3 hanya menandai 3 batch. Batch yang **tetap tertandai** pada pengali 3 adalah **23 (tekanan 450), 61 (laju 1210), dan 104 (cacat 9,80)**. Ketiganya sangat jauh di luar batas, bukan sekadar tipis melewati batas. Batch 45, 132, dan suhu 104 gugur karena hanya sedikit melewati batas 1,5×, sehingga statusnya ambigu dan perlu bukti lain (batas domain, metode lain).

---

## Latihan 2: Menghitung Modified Z-score

### 2.1 Median, MAD, dan nilai M

Rumus: `M = 0.6745 * (x - median) / MAD`

```python
med = df[kolom].median()
mad = (df[kolom] - med).abs().median()
print(pd.DataFrame({"median": med, "MAD": mad}).round(2))

M = 0.6745 * (df[kolom] - med) / mad

tabel = [(23, "tekanan_bar"), (132, "tekanan_bar"), (61, "laju_kg_jam"),
         (45, "suhu_barrel"), (104, "suhu_barrel"), (104, "cacat_persen")]

out = []
for i, kol in tabel:
    m_i = df["id"] == i
    nilai = df.loc[m_i, kol].iloc[0]
    m = M.loc[m_i, kol].iloc[0]
    out.append({"id": i, "kolom": kol, "nilai": nilai, "M": round(m, 2),
                "|M|>3.5": abs(m) > 3.5,
                "IQR 1,5": bool(flag15.loc[m_i, kol].iloc[0])})

pd.DataFrame(out)
```

### 2.2 Tandai |M| > 3,5

| id | Kolom | Nilai | M | \|M\| > 3,5 |
|---|---|---|---|---|
| 23 | tekanan_bar | 450,0 | 133,39 | Ya |
| 132 | tekanan_bar | 33,0 | -3,82 | Ya |
| 61 | laju_kg_jam | 1210,0 | 49,13 | Ya |
| 45 | suhu_barrel | 193,5 | 2,88 | **Tidak** |
| 104 | suhu_barrel | 196,4 | 3,59 | Ya |
| 104 | cacat_persen | 9,80 | 9,08 | Ya |

### 2.3 Bandingkan dengan Latihan 1 pada berkas lengkap

```python
flagM = M.abs() > 3.5

for kol in kolom:
    a = set(df.loc[flag15[kol], "id"])
    b = set(df.loc[flagM[kol], "id"])
    print(f"{kol:13s} IQR saja: {sorted(a - b)} | ModZ saja: {sorted(b - a)} | keduanya: {sorted(a & b)}")
```

**Interpretasi.** Nilai yang tertandai oleh **satu metode saja** adalah **batch 45 (`suhu_barrel` 193,5)**: IQR menandai (193,5 > 192,4), tetapi Modified Z tidak (M = 2,88). Pada berkas lengkap, IQR juga menandai sendirian suhu batch 105, 138, 147 dan laju batch 51, 85. Tidak ada nilai yang hanya ditandai Modified Z. Modified Z (ambang 3,5) lebih konservatif daripada IQR 1,5×, jadi daftarnya menjadi subset dari IQR. Nilai yang ditandai kedua metode (23, 61, 132, suhu dan cacat 104, juga 43 dan 103) adalah kandidat paling kuat, tetapi tetap perlu diperiksa dengan batas domain.

### 2.4 Mengapa Z-score biasa pada kolom tekanan hanya menandai sedikit batch

```python
t = df["tekanan_bar"]
z = (t - t.mean()) / t.std()

print("mean =", round(t.mean(), 2), "| std =", round(t.std(), 2))
print("sigma robust (MAD/0.6745) =", round(mad["tekanan_bar"] / 0.6745, 2))
print("skewness =", round(t.skew(), 2))
print("\nz batch 23, 43, 132:")
print(z[df["id"].isin([23, 43, 132])].round(2))
print("\nBatch dengan |z| > 3:")
print(df.loc[z.abs() > 3, ["id", "tekanan_bar"]])
```

**Interpretasi.** Z-score biasa memakai rata-rata dan simpangan baku, dan keduanya ikut tertarik oleh outlier. Satu nilai 450 menaikkan rata-rata menjadi 47,40 (median 44,60) dan menggelembungkan simpangan baku menjadi 33,23, padahal sebaran "normal" yang sebenarnya hanya sekitar 3,04 (MAD/0,6745), jadi kira-kira **11 kali lebih besar**. Akibatnya z batch 23 sekitar 12,1 (satu-satunya yang lolos ambang 3), sedangkan batch 132 (33,0 bar) hanya memiliki z = -0,43 dan batch 43 (56,0 bar) z = 0,26, sehingga keduanya tidak terdeteksi. Ini disebut **masking effect**: outlier ekstrem menyembunyikan outlier yang lebih kecil. Skewness 12,09 menunjukkan distribusinya sangat miring akibat satu nilai tersebut. Median dan MAD tidak terpengaruh outlier, sehingga Modified Z tetap mampu menandai 23, 43, dan 132.

---

## Latihan 3: Membedakan Outlier Statistik dan Dugaan Error

### 3.1 Bandingkan kandidat dengan batas domain (E4)

Batas domain E4: tekanan kerja 35-55 bar, kapasitas maksimum 200 kg/jam, suhu leleh material 175-195 °C.

```python
domain = {"tekanan_bar": (35, 55), "laju_kg_jam": (0, 200), "suhu_barrel": (175, 195)}
kandidat = [(61, "laju_kg_jam"), (23, "tekanan_bar"), (132, "tekanan_bar"),
            (104, "suhu_barrel"), (45, "suhu_barrel")]

rows = []
for i, kol in kandidat:
    v = df.loc[df["id"] == i, kol].iloc[0]
    lo, hi = domain[kol]
    rows.append({"id": i, "kolom": kol, "nilai": v, "batas_domain": f"{lo}-{hi}",
                 "di_dalam_domain": lo <= v <= hi,
                 "selisih_thd_batas": round(v - hi, 1) if v > hi else round(v - lo, 1) if v < lo else 0})
pd.DataFrame(rows)
```

### 3.2 Klasifikasi, 3.3 kemungkinan penyebab, 3.4 informasi yang ditanyakan

```python
klasifikasi = pd.DataFrame([
    {"id": 61,  "gejala": "laju 1210,0 kg/jam",
     "klasifikasi": "dugaan error",
     "alasan": "6 kali kapasitas maksimum 200 kg/jam; tidak mungkin secara fisik",
     "kemungkinan_penyebab": "koma bergeser (121,0 menjadi 1210,0, tepat 10x median) atau lonjakan sensor flowmeter",
     "tanya": "log mentah PLC/flowmeter saat batch 61; apakah ada alarm atau kalibrasi"},
    {"id": 23,  "gejala": "tekanan 450,0 bar",
     "klasifikasi": "dugaan error",
     "alasan": "sekitar 8 kali batas atas kerja 55 bar; tepat 10x median (45,0)",
     "kemungkinan_penyebab": "koma bergeser atau sensor/transduser tekanan menyimpang (spike)",
     "tanya": "log mentah sensor tekanan; apakah transduser baru diganti atau dikalibrasi"},
    {"id": 132, "gejala": "tekanan 33,0 bar",
     "klasifikasi": "perlu konfirmasi",
     "alasan": "hanya 2 bar di bawah batas kerja 35 bar; secara fisik masih mungkin",
     "kemungkinan_penyebab": "penurunan tekanan nyata (pasokan material, screen/filter) atau sensor drift",
     "tanya": "apakah ada gangguan umpan material atau penggantian screen; catatan operator saat batch 132"},
    {"id": 104, "gejala": "suhu 196,4 C dan cacat 9,80%",
     "klasifikasi": "perlu konfirmasi (condong kejadian nyata)",
     "alasan": "suhu 1,4 C di atas rentang 175-195; cacat 4,9x median; keduanya menyimpang bersamaan",
     "kemungkinan_penyebab": "suhu berlebih menyebabkan cacat (kejadian proses nyata); atau cacat salah ketik (9,80 vs 0,98)",
     "tanya": "log heater saat batch 104; catatan operator; apakah angka cacat 9,80 benar"},
    {"id": 45,  "gejala": "suhu 193,5 C",
     "klasifikasi": "outlier statistik",
     "alasan": "masih di dalam rentang leleh 175-195; nilai valid tetapi jarang (hanya IQR yang menandai)",
     "kemungkinan_penyebab": "variasi proses wajar",
     "tanya": "cukup dipantau; cek apakah ada perubahan setpoint"},
])
klasifikasi[["id", "gejala", "klasifikasi", "alasan"]]
```

```python
# Ingat: cacat_persen diisi manual, jadi kemungkinan kesalahannya berbeda (salah ketik, lupa mengisi)
print("Batch dengan cacat_persen kosong:", df.loc[df["cacat_persen"].isna(), "id"].tolist())
```

**Interpretasi.**

| id | Gejala | Klasifikasi | Dasar penilaian |
|---|---|---|---|
| 61 | laju 1210,0 kg/jam | **Dugaan error** | Jauh melampaui kapasitas 200 kg/jam |
| 23 | tekanan 450,0 bar | **Dugaan error** | Sekitar 8 kali batas kerja 55 bar |
| 132 | tekanan 33,0 bar | **Perlu konfirmasi** | Hanya sedikit di bawah 35 bar, masih mungkin nyata |
| 104 | suhu 196,4 °C, cacat 9,80% | **Perlu konfirmasi** | Sedikit di atas 195 °C, kedua kolom menyimpang bersamaan |
| 45 | suhu 193,5 °C | **Outlier statistik** | Di dalam rentang 175-195 °C, valid tetapi jarang |

Poin penting: 1210 dan 450 masing-masing tepat 10 kali median kolomnya (121,0 dan sekitar 45,0), pola khas **koma bergeser**. Untuk 132 dan 104 statistik saja tidak cukup untuk menyatakan salah; keduanya mungkin kejadian proses nyata. Pada batch 104, suhu tinggi dan cacat tinggi muncul bersamaan, selaras dengan penjelasan fisik bahwa suhu berlebih menaikkan cacat. Karena `cacat_persen` diisi manual, risikonya salah ketik atau terlewat (batch 12, 77, 140 kosong), sedangkan kolom otomatis berisiko sensor menyimpang, macet, atau lonjakan. Keputusan akhir untuk batch yang "perlu konfirmasi" ada pada operator atau teknisi, bukan pada algoritma.

---

## Latihan 4: Deteksi Multivariat dengan Isolation Forest dan LOF

### 4.1 Jalankan kedua metode

```python
def deteksi_multivariat(data):
    d = data.dropna(subset=kolom)
    Xs = StandardScaler().fit_transform(d[kolom])
    iso = IsolationForest(contamination=0.05, random_state=0).fit(Xs)
    lof = LocalOutlierFactor(n_neighbors=20)
    ids_if = set(d.loc[iso.predict(Xs) == -1, "id"])
    ids_lof = set(d.loc[lof.fit_predict(Xs) == -1, "id"])
    return ids_if, ids_lof, len(d)

if_semua, lof_semua, n1 = deteksi_multivariat(df)
print("Data penuh (n =", n1, ")")
print("IF :", sorted(if_semua))
print("LOF:", sorted(lof_semua))
```

### 4.2 Keluarkan id 23 dan 61, jalankan ulang

```python
df_tanpa = df[~df["id"].isin([23, 61])]
if_tanpa, lof_tanpa, n2 = deteksi_multivariat(df_tanpa)
print("Tanpa 23 dan 61 (n =", n2, ")")
print("IF :", sorted(if_tanpa))
print("LOF:", sorted(lof_tanpa))

print("\nIF  hilang:", sorted(if_semua - if_tanpa), "| baru:", sorted(if_tanpa - if_semua))
print("LOF hilang:", sorted(lof_semua - lof_tanpa), "| baru:", sorted(lof_tanpa - lof_semua))
```

### 4.3 Isi tabel

```python
iqr_set = set(df.loc[flag15.any(axis=1), "id"])
modz_set = set(df.loc[flagM.any(axis=1), "id"])

tampil = [23, 61, 104, 132, 45, 98]
tabel = pd.DataFrame({
    "id": tampil,
    "IQR":  ["v" if i in iqr_set else "-" for i in tampil],
    "ModZ": ["v" if i in modz_set else "-" for i in tampil],
    "IF":   ["v" if i in if_semua else "-" for i in tampil],
    "LOF":  ["v" if i in lof_semua else "-" for i in tampil],
    "IF (tanpa 23, 61)":  ["v" if i in if_tanpa else "-" for i in tampil],
    "LOF (tanpa 23, 61)": ["v" if i in lof_tanpa else "-" for i in tampil],
})
tabel
```

```python
# Pemeriksaan kombinasi kolom untuk batch yang hanya ditandai metode multivariat
df[df["id"].isin([98, 105, 63])][["id"] + kolom]
```

**Interpretasi.** Hasil pada data penuh (147 batch tanpa NaN):

| id | IQR | ModZ | IF | LOF |
|---|---|---|---|---|
| 23 | v | v | v | v |
| 61 | v | v | v | v |
| 104 | v | v | v | v |
| 132 | v | v | v | - |
| 45 | v | - | - | v |
| 98 | - | - | v | v |

Perubahan setelah id 23 dan 61 dikeluarkan (145 batch):

- IF: 23, 61, 141 hilang dan 43, 103, 106 masuk. Batch 63, 98, 104, 105, 132 tetap tertandai.
- LOF: 23, 61, 45, 70, 98, 99, 116 hilang; 43, 103, 106, 132 masuk. Hanya 104 dan 105 yang tetap, ditambah 103, 106, 132, dan 43.

Penjelasannya: dua nilai ekstrem (450 bar dan 1210 kg/jam) menggelembungkan simpangan baku di `StandardScaler`, sehingga semua batch lain tampak "dekat" satu sama lain di ruang terstandarisasi. Setelah keduanya dibuang, skala menjadi realistis dan batch yang sebelumnya tersamarkan (43, 103, 106) muncul. Ini kembali masking effect, kali ini pada metode multivariat. Batch 104 dan 105 konsisten tertandai di semua kondisi, jadi paling kuat sebagai kandidat nyata.

Batch **98 tidak tertandai IQR maupun ModZ**, tetapi ditandai kedua metode multivariat pada data penuh: suhu 175,7 (rendah), tekanan 49,8 (relatif tinggi), laju 107,8, dan cacat 0,13 (sangat rendah). Tiap nilai berada dalam batas univariat, tetapi **kombinasinya** jarang, dan itulah yang ditangkap IF dan LOF. Batch seperti ini perlu diperiksa kombinasi kolomnya, bukan dibuang otomatis. Perlu dicatat juga bahwa `contamination=0.05` memaksa IF menandai sekitar 5% data (7-8 batch) berapa pun isinya, jadi "tertandai" tidak sama dengan "salah".

---

## Latihan 5: Membandingkan Treatment pada Kolom Laju

### 5.1 dan 5.2 Terapkan empat perlakuan, hitung n, mean, SD, skewness

```python
s = df["laju_kg_jam"]

df["laju_mask"] = s.mask(df["id"] == 61)         # id 61 dijadikan missing
df["laju_cap"] = s.clip(upper=179.5)              # capping di 179.5
df["laju_log"] = np.log1p(s)                      # log1p

perlakuan = {
    "Dipertahankan": s,
    "Id 61 dijadikan missing": df["laju_mask"],
    "Capping di 179.5": df["laju_cap"],
    "log1p": df["laju_log"],
}

hasil = pd.DataFrame({
    nama: {"n": x.count(), "Mean": x.mean(), "SD": x.std(), "Skew": x.skew()}
    for nama, x in perlakuan.items()
}).T
hasil.round(3)
```

### 5.3 Perlakuan mana yang menghapus informasi batch lain selain id 61

```python
ubah = df.loc[s.ne(df["laju_cap"]), ["id", "laju_kg_jam", "laju_cap"]]
print("Batch yang nilainya diubah oleh capping:")
print(ubah)

print("\nNilai laju tertinggi:")
print(s.nlargest(5))
```

**Interpretasi.** Hasil tabel:

| Perlakuan | n | Mean | SD | Skew |
|---|---|---|---|---|
| Dipertahankan | 150 | 129,883 | 91,626 | 11,136 |
| Id 61 dijadikan missing | 149 | 122,634 | 22,725 | 0,522 |
| Capping di 179,5 | 150 | 122,841 | 22,625 | 0,408 |
| log1p | 150 | 4,816 | 0,262 | 4,367 |

Nilai skewness 11,14 dan 0,52 cocok dengan E5. Catatan: mean dan SD `log1p` berada pada skala logaritma sehingga tidak bisa dibandingkan langsung dengan tiga baris lain.

**Perlakuan yang menghapus informasi batch lain** adalah **capping di 179,5**. Selain id 61, capping juga menurunkan batch **51 (184,6)** dan **103 (200,2)** menjadi 179,5, sehingga informasi bahwa kedua batch itu memang lebih tinggi hilang. Perlakuan "id 61 dijadikan missing" hanya menyentuh satu batch. `log1p` mengubah seluruh nilai (bukan hanya yang ekstrem); urutan tetap terjaga tetapi skalanya berubah, dan skewness-nya masih tinggi (4,37) karena 1210 tetap jauh meski sudah dikompres.

**Pilihan untuk menetapkan ambang kendali mutu:** jadikan id 61 missing. Nilai 1210 kg/jam melampaui kapasitas 200 kg/jam sehingga hampir pasti kesalahan pencatatan, dan menghapus hanya nilai itu membuat 149 batch lain tetap utuh dengan skewness mendekati normal (0,52). Ambang kendali harus mencerminkan perilaku proses nyata, jadi tidak boleh terdistorsi oleh satu angka mustahil, dan tidak perlu mengubah nilai valid lain seperti pada capping. Catatan: batch 103 (200,2 kg/jam) sedikit di atas kapasitas 200 sehingga tetap perlu konfirmasi ke teknisi.

---

## Latihan 6: Masalah Data yang Bukan Nilai Ekstrem

### 6.1 Tampilkan empat baris batch 88-91 dan deteksi deretan dengan kode dari slide

```python
s = df["suhu_barrel"]
sama = s.eq(s.shift())
deret = sama | sama.shift(-1, fill_value=False)
df.loc[deret, ["id", "suhu_barrel"]]
```

```python
df[df["id"].between(86, 93)][["id"] + kolom]
```

### 6.2 Fungsi otomatis untuk semua kolom

```python
def deretan_identik(data, kol, min_len=3):
    """Kembalikan baris yang berada dalam deretan nilai identik berurutan sepanjang >= min_len."""
    x = data[kol]
    grup = x.ne(x.shift()).cumsum()                 # id untuk setiap deretan
    panjang = x.groupby(grup).transform("size")     # panjang deretan tiap baris
    hasil = data.loc[(panjang >= min_len) & x.notna(), ["id", kol]]
    return hasil.assign(panjang_deretan=panjang[hasil.index])

for kol in kolom:
    print(f"--- {kol} (min_len = 3) ---")
    h = deretan_identik(df, kol, min_len=3)
    print(h.to_string(index=False) if len(h) else "tidak ada")
```

```python
# Pembanding: deretan dengan panjang 2 (kemungkinan kebetulan)
for kol in kolom:
    h = deretan_identik(df, kol, min_len=2)
    print(f"--- {kol} (min_len = 2) ---")
    print(h.to_string(index=False) if len(h) else "tidak ada")
```

```python
# Mengapa detektor Latihan 1-4 melewatkan batch 88-91?
m = df["id"].between(88, 91)
print("M suhu batch 88-91:", M.loc[m, "suhu_barrel"].round(2).tolist())
print("Terkena IQR suhu :", flag15.loc[m, "suhu_barrel"].any())
print("Terkena ModZ suhu:", flagM.loc[m, "suhu_barrel"].any())
print("Terkena IF/LOF   :", bool(set(df.loc[m, "id"]) & (if_semua | lof_semua)))
```

**Interpretasi.** Hasil pencarian otomatis dengan `min_len = 3`: hanya **`suhu_barrel` batch 88-91 (182,0 empat kali berturut-turut)** yang muncul. Kolom `tekanan_bar`, `laju_kg_jam`, dan `cacat_persen` tidak memiliki deretan sepanjang itu. Dengan `min_len = 2` ada pasangan lain (suhu id 43-44 = 181,0 dan 135-136 = 182,5; tekanan id 26-27 = 44,4), tetapi pasangan identik masih wajar terjadi kebetulan karena resolusi data 0,1 dan simpangan baku suhu sekitar 4,4 °C (kira-kira satu pasangan diharapkan per 150 batch). Empat berturut-turut hampir mustahil secara kebetulan, sedangkan kolom lain pada batch 88-91 tetap bervariasi normal (tekanan 45,0 / 44,1 / 47,6 / 44,5).

**Mengapa tidak satu pun detektor Latihan 1-4 menandainya:**

- Nilai 182,0 berada sangat dekat median (181,75), dengan M sekitar 0,1, jauh di dalam batas IQR dan ModZ. Secara **nilai**, ia tampak paling normal di seluruh kolom.
- IQR dan Modified Z menilai tiap nilai terhadap sebaran, bukan **urutan** antarbaris.
- IF dan LOF memperlakukan setiap baris sebagai independen, sehingga pola "nilai yang sama berulang" tidak terlihat; kombinasi kolom per barisnya juga normal.

Masalah ini adalah **pola/kualitas data**, bukan nilai ekstrem.

**Tindakan yang disarankan:**

1. Beri bendera (flag) pada suhu batch 88-91 sebagai dicurigai, dan jangan hapus barisnya. Kolom tekanan, laju, dan cacat tampak normal dan tetap bisa dipakai. Jika perlu perhitungan, jadikan suhu batch 89-91 missing (nilai pertama, id 88, mungkin masih asli) sambil menunggu konfirmasi.
2. Jangan dijadikan dasar ambang kendali suhu sampai ada kepastian.

**Informasi yang perlu diminta kepada teknisi:**

- Apakah sensor/termokopel barrel macet (flatline) atau sedang dikalibrasi/diganti pada jam batch 88-91?
- Apakah ada gangguan komunikasi PLC/logger sehingga nilai terakhir disalin terus (forward fill)?
- Log mentah suhu per menit (data terbaca tiap 10 menit) untuk periode tersebut: apakah nilai mentahnya juga konstan?
- Apakah ada zona/sensor lain atau termometer cadangan yang bisa dibandingkan?

---

## Ringkasan Seluruh Latihan

| Batch | Temuan | Kesimpulan |
|---|---|---|
| 61 | laju 1210 kg/jam | Dugaan error (koma bergeser), ditandai semua metode |
| 23 | tekanan 450 bar | Dugaan error (koma bergeser), ditandai semua metode |
| 132 | tekanan 33 bar | Perlu konfirmasi |
| 104 | suhu 196,4 °C, cacat 9,80% | Perlu konfirmasi, condong kejadian nyata; stabil di IF dan LOF |
| 45 | suhu 193,5 °C | Outlier statistik valid (hanya IQR) |
| 98 | kombinasi kolom jarang | Hanya terdeteksi multivariat; periksa kombinasi |
| 88-91 | suhu 182,0 empat kali | Masalah kualitas data, tidak terdeteksi Latihan 1-4 |
| 12, 77, 140 | cacat_persen kosong | Konfirmasi ke operator |

Pelajaran utama: tidak ada satu metode yang cukup. Metode robust (IQR, Modified Z) mengatasi masking, metode multivariat menangkap kombinasi tidak wajar, batas domain membedakan nilai langka dari nilai mustahil, dan pemeriksaan pola menangkap masalah yang tidak berupa nilai ekstrem.
