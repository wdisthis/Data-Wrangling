# Jawaban Takeaways Pertemuan 4 — Penanganan Missing Data

*Data Wrangling – Latihan 1–5.* Kode ditulis untuk dijalankan di notebook pada berkas data Anda. Nama kolom adalah asumsi dari soal, jadi sesuaikan bila berbeda. Angka hasil eksekusi tidak saya klaim. Angka yang dipakai hanya yang ada di soal atau hasil hitung tangan dari angka soal.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option("display.width", 140)
```

---

# Latihan 1 — Forensic missing-data audit

## Soal
Data survei kerja (12 baris, kolom `id, usia, prodi, status_kerja, jam, pendapatan, berat`) berisi representasi kosong yang beragam (sel kosong, `-`, `N/A`, `Unknown`, `-1`, `999`, `0`, `99`). Tugas:
1. Klasifikasikan setiap representasi: missing / valid / kosong struktural / belum dapat diputuskan.
2. Susun data dictionary mini yang memperbaiki E2.
3. Hitung missing profile setelah klasifikasi, bandingkan dengan NaN bawaan pandas.
4. Jelaskan dampaknya pada tujuan E4 (proporsi bekerja dan rata-rata jam per prodi).

Deliverables: missing-value map, tabel keputusan + evidence, assumption log, ≥5 pertanyaan untuk pemilik data, justifikasi.

## Kode

```python
txt = pd.read_csv("L1_survei_kerja_raw.csv", keep_default_na=False, dtype=str)
nat = pd.read_csv("L1_survei_kerja_raw.csv")

def classify(r):
    s = r.status_kerja
    bekerja, tdk = (s == "bekerja"), (s == "tidak bekerja")
    o = {}

    o["usia"] = ("missing" if r.usia == "" else
                 "undecided" if r.usia == "99" else "valid")

    o["prodi"] = "missing" if r.prodi in ("", "Unknown") else "valid"

    o["status_kerja"] = "missing" if s in ("", "-", "Unknown") else "valid"

    j = r.jam
    if tdk:
        o["jam"] = "struktural"
    elif bekerja:
        if j in ("", "-1", "999"):
            o["jam"] = "missing"
        elif j == "0":
            o["jam"] = "undecided"
        else:
            o["jam"] = "valid"
    else:
        o["jam"] = "undecided"

    p = r.pendapatan
    if p in ("", "-", "N/A"):
        o["pendapatan"] = "undecided" if tdk else "missing"
    elif p == "0":
        o["pendapatan"] = "valid" if tdk else "undecided"
    else:
        o["pendapatan"] = "valid"

    o["berat"] = "missing" if r.berat in ("", "-", "999", "0") else "valid"
    return pd.Series(o)

label = txt.apply(classify, axis=1)
label.insert(0, "id", txt["id"])
print(label)

profil = (label.drop(columns="id")
               .apply(lambda c: c.value_counts())
               .fillna(0).astype(int).T)
print(profil)

print(nat.isna().sum())
print(nat.dtypes)

txt["jam_n"] = pd.to_numeric(txt["jam"], errors="coerce")

naif = nat.assign(jam_n=pd.to_numeric(nat["jam"], errors="coerce")).groupby("prodi").apply(
    lambda g: pd.Series({"n": len(g),
                         "prop_bekerja": (g.status_kerja == "bekerja").mean(),
                         "mean_jam": g.jam_n.mean()}))
print(naif)

ok_s = (label.prodi == "valid") & (label.status_kerja == "valid")
bersih_prop = (txt[ok_s].groupby("prodi")
               .apply(lambda g: pd.Series({"n_valid": len(g),
                                           "bekerja": (g.status_kerja == "bekerja").sum(),
                                           "prop": (g.status_kerja == "bekerja").mean()})))
ok_j = (label.prodi == "valid") & (label.jam == "valid")
bersih_jam = txt[ok_j].groupby("prodi").jam_n.agg(["count", "mean"])
print(bersih_prop); print(bersih_jam)
```

## Jawaban

### 1. Tabel keputusan per representasi

| Representasi | Konteks | Klasifikasi | Evidence |
|---|---|---|---|
| sel kosong `jam` | status *tidak bekerja* (baris 1, 8, 10) | **Kosong struktural** | Kamus: `jam` diisi hanya jika bekerja |
| sel kosong `jam` | status Unknown / `-` (baris 5, 12) | **Belum dapat diputuskan** | Status sendiri hilang, jadi tidak tahu apakah wajib terisi |
| `-1` pada `jam` | baris 4, *tidak bekerja* | **Kosong struktural** | E3: `-1` = pengganti sel numerik kosong. Sel itu memang harus kosong |
| `999` pada `jam` | baris 9 | **Missing** | 999 jam/minggu > 168 jam, mustahil; kode tak terdokumentasi |
| `0` pada `jam` | baris 6, *bekerja*, pendapatan 1,2 jt | **Belum dapat diputuskan** | Bekerja tetapi 0 jam kontradiktif; bisa salah input atau sedang libur |
| `99` pada `usia` | baris 3 | **Belum dapat diputuskan** | Masih di rentang dropdown 17–99, tetapi 99 adalah ujung dropdown (kemungkinan nilai default/asal klik) |
| kosong pada `usia` | baris 5 | **Missing** | Kolom wajib dropdown, tidak ada alasan struktural |
| kosong / `Unknown` pada `prodi` | baris 4, 6 | **Missing** | Placeholder eksplisit |
| `Unknown` / `-` pada `status_kerja` | baris 5, 12 | **Missing** | Nilai di luar domain {bekerja, tidak bekerja} |
| `0` pada `pendapatan` | *tidak bekerja* (baris 1, 4) | **Valid** | Kamus: 0 = tidak berpenghasilan |
| `0` pada `pendapatan` | *bekerja* (baris 11) | **Belum dapat diputuskan** | Bisa magang tanpa gaji, bisa salah isi |
| `N/A` pada `pendapatan` | baris 7 (bekerja) | **Missing** | Placeholder eksplisit |
| kosong / `-` pada `pendapatan` | status tidak bekerja (baris 8, 10) | **Belum dapat diputuskan** | Seharusnya 0 menurut kamus, jadi bisa missing atau bisa "tidak ada" |
| kosong pada `pendapatan` | status Unknown/`-` (baris 5, 12) | **Missing** | Tidak ada alasan struktural |
| `-`, `999`, `0` pada `berat` | baris 5, 4, 9 | **Missing** | Berat 0 atau 999 kg mustahil; `-` placeholder |

### 2. Data dictionary mini (perbaikan E2)

| Kolom | Tipe | Domain valid | Kode khusus | Aturan struktural |
|---|---|---|---|---|
| `usia` | int | 17–99 (verifikasi 99) | kosong = missing | – |
| `prodi` | kategori | {Sains Data, Matematika, Fisika, Kimia} | `Unknown`/kosong = missing | – |
| `status_kerja` | kategori | {bekerja, tidak bekerja} | `Unknown`, `-`, kosong = missing | – |
| `jam_kerja_minggu` | int | 1–168 | `-1` = pengganti kosong, `999` = sentinel (**perlu konfirmasi**) | kosong jika `status_kerja = tidak bekerja` |
| `pendapatan_bulanan` | int (Rp) | ≥ 0; 0 = tidak berpenghasilan | `N/A`, `-` = missing | `0` diharapkan jika tidak bekerja |
| `berat_kg` | float | 30–200 (usulan) | `-`, `0`, `999` = tidak valid, jadi missing | – |

### 3. Perbandingan missing profile

Setelah klasifikasi (dari 12 baris):

| Kolom | missing | struktural | belum diputuskan | valid |
|---|---|---|---|---|
| usia | 1 | 0 | 1 | 10 |
| prodi | 2 | 0 | 0 | 10 |
| status_kerja | 2 | 0 | 0 | 10 |
| jam | 1 | 4 | 3 | 4 |
| pendapatan | 3 | 0 | 3 | 6 |
| berat | 3 | 0 | 0 | 9 |

NaN bawaan pandas: `usia` 1, `prodi` 1, `status_kerja` **0**, `jam` 5, `pendapatan` 4, `berat` **0**.

Perbedaannya:
- `status_kerja` dan `berat` terlihat **bersih** padahal tidak (`Unknown`, `-`, `999`, `0` lolos sebagai nilai valid).
- `jam` terlihat missing 5, padahal 4 di antaranya struktural (bukan hilang) atau belum bisa diputuskan, dan sentinel `-1`/`999` malah dianggap angka asli.
- `pendapatan` dan `berat` menjadi kolom bertipe teks, sehingga `mean()` gagal atau salah.

### 4. Dampak pada tujuan E4

- **Proporsi bekerja:** hanya 10 dari 12 baris punya status valid (6 bekerja, jadi 60%). Pada pendekatan naif, `Unknown` dan `-` ikut menjadi penyebut, dan baris `prodi = Unknown` menjadi "prodi" sendiri.
- **Rata-rata jam kerja:** pada pendekatan naif, Sains Data = (12 + 999)/2 = **505,5 jam/minggu**. Setelah `999` dibuang, nilainya 12 jam. Hanya 1 pekerja valid per prodi, jadi estimasi per prodi **tidak stabil secara statistik** dan hanya layak ditampilkan sebagai ilustrasi dengan catatan ukuran sampel.

### Assumption log
- A1: `-1` hanya muncul pada kolom `jam` (pada data ini ya), dan menggantikan sel yang seharusnya kosong.
- A2: `999` dan `0` kg adalah kode missing/invalid.
- A3: `99` pada `usia` diperlakukan sebagai belum diputuskan, dikeluarkan dari analisis usia sampai dikonfirmasi.
- A4: baris dengan status missing tidak diimputasi secara otomatis karena informasi tidak cukup.

### Pertanyaan untuk pemilik data
1. Apa arti kode `-1` dan `999`, dan di kolom mana saja keduanya muncul?
2. Apakah `99` pada usia adalah nilai default dropdown?
3. Apakah `-` berarti "tidak diisi", "tidak berlaku", atau "nol"?
4. Pada mahasiswa tidak bekerja, apakah `pendapatan` kosong sama dengan 0?
5. Apakah `jam = 0` pada responden bekerja valid (mis. cuti/libur) atau error?
6. Apakah ada aturan validasi form (wajib isi) pada `status_kerja` dan `prodi`?
7. Apakah `berat` ditimbang mandiri, sehingga 0 dan 999 dari input asal?

### Justifikasi
Representasi kosong tidak boleh diseragamkan. Satu kode (`0`, `-1`) bisa berarti nilai asli, missing, atau struktural, tergantung konteks kolom lain. Maka klasifikasi memakai aturan **kondisional** (status kerja), bukan sekadar mengganti string menjadi `NaN`. Kasus yang bukti-nya tidak cukup sengaja diberi label *belum dapat diputuskan* dan tidak dipaksa.

---

# Latihan 2 — Mechanism investigation

## Soal
Survei kesejahteraan (N = 480). Skor stres kosong 22%. Evidence E1–E5 (missing per angkatan 31/24/17/12%, rata-rata skor teramati, missing per gender, item di halaman 4 dan 18% responden berhenti sebelum halaman 4, angkatan 2025 mengisi saat UTS, tindak lanjut telepon 20 orang: 9 bersedia, 6 berskor ≥ 28). Tugas: tabel evidence, hipotesis + tingkat keyakinan, evaluasi E5, evidence tambahan, risiko terhadap rata-rata per angkatan.

## Kode

```python
from scipy import stats
import statsmodels.formula.api as smf

df["miss"] = df["skor_stres"].isna().astype(int)

print(df.groupby("angkatan").miss.mean())
print(df.groupby("gender").miss.mean())

for var in ["angkatan", "gender"]:
    tab = pd.crosstab(df[var], df["miss"])
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    print(var, f"chi2={chi2:.2f}, dof={dof}, p={p:.4f}")

m = smf.logit("miss ~ C(angkatan) + C(gender) + halaman_terakhir", data=df).fit()
print(m.summary())

def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return c - h, c + h

print("Wilson 6/9:", wilson(6, 9))
print("Tingkat respons telepon:", 9 / 20)

obs_mean = pd.Series({2025: 24.1, 2024: 22.8, 2023: 21.9, 2022: 23.5})
pi_mis   = pd.Series({2025: 0.31, 2024: 0.24, 2023: 0.17, 2022: 0.12})

rows = {}
for delta in [0, 2, 4, 6]:
    rows[f"delta={delta}"] = (obs_mean + pi_mis * delta).round(2)
sens = pd.DataFrame(rows)
print(sens)
```

## Jawaban

### Tabel evidence

| Evidence | Mendukung | Melemahkan | Belum diketahui |
|---|---|---|---|
| **E1** missing turun 31 → 12% menurut angkatan | MAR (bergantung angkatan/waktu pengisian); **bukan MCAR** | MCAR | Apakah selisih signifikan (butuh ukuran per angkatan) |
| **E2** mean teramati 24,1 / 22,8 / 21,9 / 23,5 | – (pola tidak monoton, tidak jelas) | Tren sederhana "makin tua makin stres" | Mean angkatan 2025 yang sebenarnya, karena 31% hilang |
| **E3** missing L 21% vs P 23% | MCAR terhadap gender | Hipotesis bahwa gender menentukan missingness | Interaksi gender × angkatan |
| **E4** item di halaman 4; 18% berhenti sebelum halaman 4; 2025 saat UTS | MAR via **attrition**/kelelahan survei dan **waktu UTS**; sebagian besar missing (≈18 dari 22 poin) berasal dari berhenti, bukan melewatkan item | MNAR murni oleh nilai skor | Mengapa berhenti (bosan, panjang, UTS, atau stres tinggi) |
| **E5** 9/20 bersedia; 6/9 berskor ≥ 28 | **MNAR**: yang hilang cenderung lebih stres daripada yang mengisi (rata-rata teramati hanya 24,1) | MCAR dan MAR sederhana | Skor 11 non-responden lain; apakah bersedia menjawab = tidak stres? |

### 1. Hipotesis dan keyakinan
- **H1: bukan MCAR.** Keyakinan **tinggi** (E1 dan E4: missingness jelas berbeda per angkatan dan terkait posisi item/pekan UTS).
- **H2: MAR melalui desain survei** (item di halaman akhir + waktu pengisian saat UTS). Keyakinan **sedang-tinggi** (E1, E4).
- **H3: MNAR parsial**, yaitu orang yang stres tinggi lebih sering berhenti. Keyakinan **sedang** (E5 searah, tetapi ukuran kecil dan terseleksi). Kemungkinan terbaik adalah **campuran H2 + H3**: UTS memicu kelelahan dan stres, dan keduanya membuat orang berhenti.

### 2. Kekuatan E5
Lemah-sedang. Alasan: (1) ukuran **n = 9** (bukan 20; 11 menolak). Interval Wilson 95% untuk 6/9 kira-kira **35%–88%**, sangat lebar. (2) **Seleksi ganda**: hanya 45% bersedia dijawab, dan yang bersedia mungkin berbeda dari yang tidak. (3) Hanya angkatan 2025, jadi tidak digeneralisasi ke angkatan lain. (4) Waktu telepon berbeda dari waktu survei, dan skor ≥ 28 bisa dipengaruhi UTS. Tetapi arahnya (mayoritas tinggi) konsisten dengan MNAR sehingga tidak boleh diabaikan.

### 3. Evidence tambahan
- Data mentah: halaman terakhir per responden (apakah berhenti di halaman 3 atau 4), waktu pengisian, ukuran kelompok per angkatan.
- Uji chi-square dan regresi logistik missingness ~ angkatan + gender + halaman + pekan UTS.
- Tindak lanjut acak yang lebih besar dan jawaban singkat (bukan telepon panjang), idealnya dengan insentif.
- Item stres dipindah ke halaman awal pada gelombang berikutnya (eksperimen desain).
- Data paradata (durasi, perangkat), korelasi skor kecemasan/IPK sebagai prediktor MAR.

### 4. Risiko jika hipotesis salah

Mean populasi, dengan $\pi$ = proporsi missing:

$$\bar{y} = (1-\pi)\,\bar{y}_{obs} + \pi\,\bar{y}_{mis}, \qquad \bar{y}_{mis} = \bar{y}_{obs} + \delta$$

$$\text{Bias}\big(\bar{y}_{obs}\big) = \bar{y}_{obs} - \bar{y} = -\pi\,\delta$$

Jika MNAR benar dan $\delta > 0$, rata-rata teramati **meremehkan** stres, terutama pada angkatan dengan $\pi$ besar:

| | 2025 ($\pi$=.31) | 2024 (.24) | 2023 (.17) | 2022 (.12) |
|---|---|---|---|---|
| $\delta=0$ | 24,10 | 22,80 | 21,90 | 23,50 |
| $\delta=2$ | 24,72 | 23,28 | 22,24 | 23,74 |
| $\delta=4$ | 25,34 | 23,76 | 22,58 | 23,98 |
| $\delta=6$ | 25,96 | 24,24 | 22,92 | 24,22 |

Pada $\delta \geq 4$, angkatan 2025 menjadi yang tertinggi dengan selisih jelas, padahal pada data teramati 2025 dan 2022 hampir sama (24,1 vs 23,5). Jadi **urutan dan pola antarangkatan bisa berubah**. Bias terbesar pada angkatan 2025, sehingga perbandingan antarangkatan paling tidak aman. Jika hipotesis MAR yang benar tetapi dianalisis sebagai MNAR (δ besar), kita sebaliknya melebih-lebihkan. Karena itu hasilnya dilaporkan sebagai **rentang**.

---

# Latihan 3 — Treatment decision under constraints

## Soal
Survei pengeluaran bulanan (N = 150). Tujuan: rata-rata `pengeluaran` per fakultas. E1–E5: missing `pengeluaran` 16%, FS 17/62, FTI 4/55, FTIK 3/33, korelasi dengan `uang_saku` 0,80, mean uang_saku 1,69 (pengeluaran terisi) vs 2,30 (kosong), missing pada penerima beasiswa 23,4% vs 12,7%. Constraint: tidak bisa kumpul ulang, FTIK hanya 33, harus menyebut ketidakpastian. Tugas: bandingkan deletion, simple imputation, dan advanced imputation (KNN/MICE), buat decision matrix, jelaskan penolakan dan fakultas paling sensitif.

## Kode

```python
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge
import statsmodels.formula.api as smf

df = pd.read_csv("L3_pengeluaran_mahasiswa.csv")

df["miss_peng"] = df["pengeluaran"].isna().astype(int)
print(df.isna().agg(["sum", "mean"]).T)
print(df.groupby("fakultas").miss_peng.agg(["sum", "count", "mean"]))
print(df.groupby("miss_peng").uang_saku.mean())
print(df.groupby("beasiswa", dropna=False).miss_peng.mean())

print(smf.logit("miss_peng ~ uang_saku + jarak_kos_km + C(fakultas)",
                data=df.dropna(subset=["uang_saku"])).fit().summary())

b = df["beasiswa"].astype("string").str.lower().map(
        {"ya": 1, "tidak": 0, "1": 1, "0": 0, "true": 1, "false": 0})
X = pd.concat([
        df[["pengeluaran", "uang_saku", "jarak_kos_km"]],
        b.rename("beasiswa_num"),
        pd.get_dummies(df["fakultas"], prefix="fak", dtype=float)
    ], axis=1)

def ringkas(y, fak, nama):
    t = pd.DataFrame({"y": y, "fak": fak})
    out = t.groupby("fak").y.agg(["count", "mean", "median", "std"])
    out.loc["Semua"] = [t.y.count(), t.y.mean(), t.y.median(), t.y.std()]
    out.insert(0, "strategi", nama)
    return out

A = df.dropna(subset=["pengeluaran"])
res_A = ringkas(A["pengeluaran"], A["fakultas"], "A deletion")

y_B = df["pengeluaran"].fillna(df["pengeluaran"].median())
res_B = ringkas(y_B, df["fakultas"], "B median")

m = 5
imputed = []
for s in range(m):
    imp = IterativeImputer(estimator=BayesianRidge(), sample_posterior=True,
                           max_iter=20, random_state=s)
    imputed.append(pd.DataFrame(imp.fit_transform(X), columns=X.columns, index=X.index))

def rubin_mean(datasets, groups):
    out = {}
    for g in groups.unique():
        idx = groups == g
        Q = np.array([d.loc[idx, "pengeluaran"].mean() for d in datasets])
        U = np.array([d.loc[idx, "pengeluaran"].var(ddof=1) / idx.sum() for d in datasets])
        Qbar, W, B = Q.mean(), U.mean(), Q.var(ddof=1)
        T = W + (1 + 1 / len(datasets)) * B
        out[g] = {"n": idx.sum(), "mean": Qbar, "se": np.sqrt(T),
                  "ci_low": Qbar - 1.96 * np.sqrt(T), "ci_high": Qbar + 1.96 * np.sqrt(T),
                  "rentang_antar_imputasi": f"{Q.min():.2f}-{Q.max():.2f}"}
    return pd.DataFrame(out).T

res_C = rubin_mean(imputed, df["fakultas"])
print(res_C)

bandingA = res_A["mean"].rename("A_deletion")
bandingB = res_B["mean"].rename("B_median")
bandingC = res_C["mean"].rename("C_mice")
cmp = pd.concat([bandingA, bandingB, bandingC], axis=1)
cmp["range_antar_strategi"] = cmp.max(axis=1) - cmp.min(axis=1)
print(cmp)

knn = KNNImputer(n_neighbors=5)
Xk = pd.DataFrame(knn.fit_transform(X), columns=X.columns, index=X.index)
print(Xk.groupby(df["fakultas"]).pengeluaran.mean())

obs = df["pengeluaran"].dropna()
mis_idx = df["pengeluaran"].isna()
plt.hist(obs, bins=20, alpha=.5, density=True, label="teramati")
plt.hist(imputed[0].loc[mis_idx, "pengeluaran"], bins=10, alpha=.5, density=True, label="imputasi")
plt.legend(); plt.xlabel("pengeluaran (juta Rp)"); plt.show()
```

Rumus pooling Rubin untuk $m$ dataset imputasi, dengan $Q_i$ = estimasi, $U_i$ = varians estimasi:

$$\bar{Q} = \frac{1}{m}\sum_{i=1}^{m} Q_i,\qquad W = \frac{1}{m}\sum_{i=1}^{m} U_i,\qquad B = \frac{1}{m-1}\sum_{i=1}^{m}(Q_i-\bar{Q})^2$$

$$T = W + \left(1+\frac{1}{m}\right)B$$

## Jawaban

### Decision matrix

Angka mean berasal dari hasil eksekusi yang tercantum pada tabel Latihan 4. Jalankan kode di atas untuk memverifikasi.

| Strategy | Evidence | Benefit | Risk | Assumption | Consequence | Decision |
|---|---|---|---|---|---|---|
| **A. Deletion** (126 baris) | E5: yang hilang punya uang saku lebih tinggi (2,30 vs 1,69) dan beasiswa lebih sering hilang; E4 r = 0,80 | Sederhana, tidak ada nilai buatan | Buang 16% data; **missing tidak acak**, jadi pengeluaran tinggi hilang → mean cenderung terlalu rendah. FS kehilangan 27% (17/62) | MCAR | Mean FS 2,10 tanpa memperhitungkan 17 yang hilang; CI lebih lebar | **Tolak** sebagai estimasi utama; dipakai sebagai sensitivitas |
| **B. Simple (median)** | E3: median 2,00 | Cepat, n tetap 150 | Menimpa 24 nilai dengan 2,00; **varians menyusut** (SD 0,57 → 0,52); mengabaikan `uang_saku` (r = 0,80); SE tampak lebih kecil dari seharusnya | MCAR dan semua yang hilang "tipikal" | Mean tertarik ke pusat (2,08); FS 2,07; ketidakpastian dilaporkan terlalu kecil | **Tolak** (melanggar syarat "menyebut ketidakpastian") |
| **C. MICE (m = 5)** | E4: `uang_saku` prediktor kuat; E5: missing bergantung pada `uang_saku`, `beasiswa` | Memakai hubungan antarvariabel; mempertahankan varians; ketidakpastian tercermin lewat Rubin | Bergantung pada model imputasi; jika MNAR masih bias; hanya 5 dataset | **MAR** bersyarat pada `uang_saku`, `beasiswa`, `fakultas` | Mean 2,18; FS 2,26; CI lebih jujur | **Pilih** sebagai estimasi utama, dengan A dan B sebagai sensitivitas |

### Mengapa A dan B ditolak
- **A (deletion):** syarat validnya MCAR, sedangkan E5 menunjukkan missing berhubungan dengan `uang_saku` dan `beasiswa`. Dengan N kecil dan FS paling banyak hilang, bias dan kehilangan presisi terbesar justru di fakultas yang paling menentukan.
- **B (median):** nilai pengganti sama untuk semua orang sehingga mengabaikan korelasi 0,80 dengan `uang_saku`, mengerutkan SD, dan menghasilkan SE terlalu kecil. Laporan yang harus menyebut ketidakpastian jadi menyesatkan.

### Fakultas paling sensitif
**FS**: tingkat missing tertinggi (17/62 ≈ 27%), dan mean berubah dari 2,07 (median) → 2,10 (deletion) → 2,26 (MICE), selisih sekitar 0,19. FTI (4/55 ≈ 7%) dan FTIK (3/33 ≈ 9%) nyaris tidak berubah (2,07–2,11 dan 2,12–2,13). Tetapi FTIK **paling tidak pasti** karena hanya 33 responden, jadi CI-nya lebar walaupun nilai tengahnya stabil.

---

# Latihan 4 — Conflicting cleaned datasets

## Soal
Tiga analis mengolah berkas Latihan 3 dengan cara berbeda (A deletion, B median, C MICE). Pertanyaan: (1) informasi apa yang berubah; (2) di mana distorsi dan penyebabnya; (3) keputusan paling bisa dipertanggungjawabkan untuk subsidi per fakultas dan batasnya; (4) kapan keputusan harus berubah.

## Kode

```python
n_all, n_obs, n_mis = 150, 126, 24
mean_A, med = 2.10, 2.00

mean_B = (n_obs * mean_A + n_mis * med) / n_all
print("mean B (hitung):", round(mean_B, 3))

sd_A = 0.57
sd_B_approx = np.sqrt((n_obs - 1) * sd_A**2 / (n_all - 1))
print("SD B (hampir):", round(sd_B_approx, 3))

tabel = pd.DataFrame({
    "fak": ["FS", "FTI", "FTIK"],
    "n": [62, 55, 33],
    "miss": [17, 4, 3],
    "mean_A": [2.10, 2.08, 2.13],
    "mean_B": [2.07, 2.07, 2.12],
    "mean_C": [2.26, 2.11, 2.13],
})
tabel["n_obs"] = tabel.n - tabel.miss
tabel["pct_miss"] = (tabel.miss / tabel.n * 100).round(1)

tabel["mean_imputed_C"] = ((tabel.n * tabel.mean_C - tabel.n_obs * tabel.mean_A) / tabel.miss).round(2)
tabel["selisih_C_vs_A"] = (tabel.mean_C - tabel.mean_A).round(2)
print(tabel)

mean_imp_all = (n_all * 2.18 - n_obs * 2.10) / n_mis
print("Rata-rata nilai imputasi MICE (keseluruhan):", round(mean_imp_all, 2))

se = 0.57 / np.sqrt(30)
print("FTIK CI 95% kasar:", 2.13 - 1.96 * se, 2.13 + 1.96 * se)
```

## Jawaban

### 1. Informasi yang berubah

| Aspek | A (deletion) | B (median) | C (MICE) |
|---|---|---|---|
| n efektif | 126 | 150 | 150 |
| Mean | 2,10 | 2,08 | **2,18** |
| Median | 2,00 | 2,00 | 2,08 |
| SD | 0,57 | **0,52** (menyusut) | 0,60 (tidak menyusut) |
| Mean FS | 2,10 | 2,07 | **2,26** |

Hanya **C** yang mengubah mean secara berarti. A dan B memberi mean hampir sama karena B menimpa 24 nilai dengan median 2,00:

$$\bar{y}_B = \frac{126(2{,}10) + 24(2{,}00)}{150} \approx 2{,}08$$

Hitung balik rata-rata nilai yang diisi MICE:

$$\bar{y}_{imp} = \frac{150(2{,}18) - 126(2{,}10)}{24} \approx 2{,}60$$

Artinya MICE menduga orang yang hilang membelanjakan sekitar **0,5 juta lebih tinggi** daripada yang teramati. Untuk FS: $(62 \times 2{,}26 - 45 \times 2{,}10)/17 \approx 2{,}68$.

### 2. Potensi distorsi dan penyebab

| Hasil | Distorsi | Penyebab |
|---|---|---|
| **A** | Mean cenderung **terlalu rendah**, terutama FS | Yang hilang punya `uang_saku` lebih tinggi (E5) dan korelasi 0,80, jadi pengeluaran tinggi ikut terbuang (complete-case bias) |
| **B** | Mean **tertarik ke 2,00**, SD menyusut (0,52), SE terlalu kecil | Semua nilai imputasi identik, tanpa memakai `uang_saku`; kelihatan seperti n = 150 padahal informasi hanya 126 |
| **C** | Bisa **terlalu tinggi** jika model salah; FS bergantung pada **satu prediktor kuat** | Hanya valid bila MAR bersyarat pada `uang_saku`/`beasiswa`; jika penerima beasiswa menyembunyikan pengeluaran karena alasan lain (MNAR), model tidak menangkapnya |

Catatan: rentang antar-imputasi C sempit (2,16–2,19 keseluruhan; FS 2,24–2,28), jadi perbedaan dengan A **bukan noise** imputasi acak. Ia berasal dari **asumsi** model.

### 3. Keputusan paling dapat dipertanggungjawabkan
**Pakai C (MICE) sebagai estimasi utama**, dilaporkan per fakultas dengan interval Rubin, dan tampilkan A dan B sebagai analisis sensitivitas. Alasannya hanya C yang sejalan dengan evidence (missing bergantung pada `uang_saku`, korelasi 0,80) dan menjaga ketidakpastian.

Batas kesimpulan:
- **FS**: titik estimasi berkisar 2,07–2,26 menurut metode. Laporkan sebagai rentang, jangan satu angka. FS **mungkin** lebih tinggi daripada FTI dan FTIK hanya jika asumsi MAR berlaku.
- **FTI dan FTIK**: tidak ada perbedaan bermakna antarmetode, tetapi **FTIK n = 30 teramati** (SE ≈ 0,10, CI 95% kasar ≈ 1,93–2,33), jadi **tidak layak diranking** terhadap FTI.
- Tidak ada klaim kausal, dan MAR tidak dapat diuji dari data.

### 4. Keputusan harus berubah jika
1. Ada bukti **MNAR**: tindak lanjut/data administratif menunjukkan yang hilang membelanjakan jauh berbeda dari yang diprediksi `uang_saku` (analisis tipping point δ membalik kesimpulan).
2. Hubungan `uang_saku` → `pengeluaran` berbeda per fakultas atau status beasiswa (interaksi), sehingga model imputasi perlu diubah.
3. Diagnostik MICE buruk (nilai imputasi di luar rentang 0,81–4,57, distribusi tidak masuk akal, konvergensi gagal).
4. Ambang subsidi berada di antara 2,10 dan 2,26, sehingga pilihan metode mengubah siapa yang menerima.
5. Data tambahan menurunkan missing sehingga perbedaan antarmetode mengecil.

---

# Latihan 5 — Missing-data decision memo

## Soal
Layanan konseling ingin melaporkan proporsi klien yang membaik (PHQ-9 turun ≥ 5) pada 2025. N = 620. E1–E6: `phq9_akhir` hilang 38%; hilang 15% pada klien "selesai" (n = 380) vs 74% pada "berhenti" (n = 240); rata-rata `phq9_awal` 11,0 (akhir ada) vs 14,2 (kosong); alasan berhenti tidak tercatat; `alasan_rujukan` Jan–Mar hilang total; klien tidak boleh dihubungi, waktu 2 minggu, pimpinan minta "satu angka". Deliverable: memo ≤ ±500 kata, 10 bagian, dan tanggapan permintaan "satu angka".

## Kode

```python
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

df = pd.read_csv("L5_konseling.csv")
N = len(df)

print(df.isna().mean().round(3))
print(df.groupby("status_layanan").phq9_akhir.apply(lambda s: s.isna().mean()))
df["miss_akhir"] = df.phq9_akhir.isna().astype(int)
print(df.groupby("miss_akhir").phq9_awal.mean())

df["turun"] = df.phq9_awal - df.phq9_akhir
df["membaik"] = np.where(df.turun.notna(), (df.turun >= 5).astype(float), np.nan)

obs = df.membaik.dropna()
p_obs = obs.mean()
n_obs, n_mis = len(obs), N - len(obs)

lower = p_obs * n_obs / N
upper = (p_obs * n_obs + n_mis) / N
print(f"Batas tanpa asumsi: [{lower:.3f}, {upper:.3f}]  lebar = {upper - lower:.3f}")

strata = df.groupby("status_layanan").agg(
    N_s=("membaik", "size"),
    n_obs=("membaik", "count"),
    p_hat=("membaik", "mean"))
strata["r_s"] = strata.n_obs / strata.N_s
p_strat = (strata.N_s / N * strata.p_hat).sum()
print(strata); print("Estimasi terstrata:", round(p_strat, 3))

def p_delta(delta_berhenti, delta_selesai=0.0):
    tot = 0
    for s, d in [("selesai", delta_selesai), ("berhenti", delta_berhenti)]:
        r = strata.loc[s]
        p_mis = np.clip(r.p_hat + d, 0, 1)
        tot += r.N_s * (r.r_s * r.p_hat + (1 - r.r_s) * p_mis)
    return tot / N

grid = np.round(np.arange(-0.4, 0.41, 0.1), 1)
sens = pd.DataFrame({"delta_berhenti": grid, "p_total": [p_delta(d) for d in grid]})
print(sens)

X = pd.concat([df[["phq9_awal", "phq9_akhir", "jumlah_sesi"]],
               pd.get_dummies(df[["status_layanan", "jenis_layanan"]], dtype=float)], axis=1)

m, ps, vs = 20, [], []
for s in range(m):
    imp = IterativeImputer(estimator=BayesianRidge(), sample_posterior=True,
                           max_iter=20, random_state=s)
    Xi = pd.DataFrame(imp.fit_transform(X), columns=X.columns)
    b = ((Xi.phq9_awal - Xi.phq9_akhir) >= 5).astype(float)
    ps.append(b.mean()); vs.append(b.var(ddof=1) / N)

Qbar, W, B = np.mean(ps), np.mean(vs), np.var(ps, ddof=1)
T = W + (1 + 1 / m) * B
print(f"MI: p = {Qbar:.3f}, 95% CI = ({Qbar - 1.96*np.sqrt(T):.3f}, {Qbar + 1.96*np.sqrt(T):.3f})")
```

Rumus yang dipakai:

- Batas tanpa asumsi (Manski), dengan $\hat{p}$ = proporsi membaik pada yang teramati:

$$\frac{n_{obs}\,\hat{p}}{N}\;\le\;p\;\le\;\frac{n_{obs}\,\hat{p}+n_{mis}}{N},\qquad \text{lebar}=\frac{n_{mis}}{N}\approx 0{,}38$$

- Estimasi terstrata dengan penyesuaian $\delta_s$ (pattern-mixture), $r_s$ = tingkat teramati pada strata $s$:

$$p(\boldsymbol{\delta})=\sum_{s}\frac{N_s}{N}\Big[r_s\,\hat{p}_s+(1-r_s)\big(\hat{p}_s+\delta_s\big)\Big]$$

Tingkat teramati: selesai $r = 0{,}85$ (323 dari 380), berhenti $r = 0{,}26$ (sekitar 62 dari 240).

## Jawaban: Memo keputusan (± 480 kata)

**Kepada:** Pimpinan Layanan Konseling | **Perihal:** Proporsi klien membaik 2025

**1. Masalah.** Melaporkan proporsi klien yang membaik (PHQ-9 turun ≥ 5) dari 620 klien, padahal skor akhir kosong pada 38%.

**2. Evidence.** E1: `phq9_akhir` hilang 38%. E2: hilang 15% pada "selesai" (n = 380) vs 74% pada "berhenti" (n = 240). E3: rata-rata skor awal 11,0 (skor akhir ada) vs 14,2 (kosong). E4: alasan berhenti beragam dan tidak tercatat. E5: `alasan_rujukan` Jan–Mar hilang akibat migrasi. E6: tidak boleh menghubungi ulang, 2 minggu, diminta satu angka.

**3. Dugaan mekanisme.** Bukan MCAR (E2, E3). Dugaan: (a) MAR bersyarat pada status layanan dan skor awal, keyakinan **tinggi**; (b) MNAR di dalam kelompok "berhenti", sebab alasan berhenti terkait hasil dengan arah berlawanan (E4), keyakinan **sedang**. `alasan_rujukan` hilang karena migrasi sistem, kira-kira MCAR.

**4. Treatment.** Estimasi utama: *multiple imputation* (MICE, m = 20; prediktor skor awal, status, jenis layanan, jumlah sesi) dengan pooling Rubin. Dilengkapi analisis sensitivitas δ pada kelompok "berhenti" dan batas tanpa asumsi (Manski).

**5. Alternatif ditolak.** (i) *Complete-case*: hanya 62% data, didominasi klien "selesai" dengan skor awal lebih rendah, jadi bias. (ii) Missing dianggap "tidak membaik": itu batas bawah, bukan estimasi. (iii) Imputasi rata-rata/median: mengabaikan status dan menyusutkan varians. (iv) Mengimputasi `alasan_rujukan`: tidak dibutuhkan outcome dan tidak dapat dipulihkan.

**6. Alasan.** MI memakai informasi E2–E3 dan mempertahankan ketidakpastian. Bagian yang tidak teridentifikasi dari data (MNAR) dijawab oleh sensitivitas, bukan disembunyikan.

**7. Risiko.** Bila MNAR, estimasi MI bias dengan arah tidak diketahui (E4 campuran). Pada kelompok "berhenti" 74% skor hilang, sehingga hasilnya hampir sepenuhnya bergantung model. Batas tanpa asumsi selebar sekitar 38 poin persentase.

**8. Assumption log.** A1: membaik dihitung dari selisih awal–akhir ≥ 5. A2: MAR bersyarat pada variabel di model. A3: status selesai/berhenti tercatat akurat. A4: skor PHQ-9 valid (0–27). A5: migrasi sistem tidak memengaruhi PHQ-9. A6: skala diterapkan konsisten antarkonselor.

**9. Validasi.** (i) Audit catatan konselor pada sampel acak klien "berhenti" (tanpa menghubungi klien) untuk menaksir δ. (ii) Diagnostik MICE (distribusi, konvergensi). (iii) Estimasi per jenis layanan. (iv) Bandingkan dengan 2024. (v) Periksa 2% `phq9_awal` hilang.

**10. Kondisi yang mengubah keputusan.** Jika audit menunjukkan alasan berhenti dominan satu arah, δ diganti nilai terukur. Jika mekanisme ternyata MCAR dalam strata, estimasi terstrata cukup. Jika tindak lanjut diizinkan, data baru menggantikan asumsi. Jika seluruh rentang sensitivitas berada di satu sisi ambang keputusan, kesimpulannya aman.

### Tanggapan atas "satu angka" (E6)

Bisa dipenuhi **sebagian**. Satu *angka utama* (estimasi MI) boleh disebutkan sebagai perkiraan terbaik di bawah asumsi tertulis, tetapi **hanya jujur bila selalu ditemani rentangnya** (CI dan hasil sensitivitas). Satu angka tanpa rentang menyiratkan presisi yang tidak ada: 38% skor akhir hilang, dan 74% pada klien yang berhenti. Bahkan dengan data teramati sempurna, batas tanpa asumsi selebar sekitar 38 poin persentase. Usul format: "Perkiraan terbaik X% (rentang masuk akal A–B%), dengan asumsi yang tertulis."

---

## Ringkasan prinsip lintas latihan

1. **Klasifikasikan sebelum menangani.** Kode seperti `0`, `-1`, `999`, `-` bergantung konteks kolom lain.
2. **Mekanisme adalah argumen, bukan label.** Tulis evidence yang mendukung, melemahkan, dan belum diketahui.
3. **Pilih treatment dengan mempertimbangkan constraint** dan sebut asumsi (MCAR / MAR / MNAR).
4. **Imputasi tunggal menyusutkan ketidakpastian**; multiple imputation dengan pooling Rubin mempertahankannya.
5. **Selalu sertakan analisis sensitivitas** karena MNAR tidak dapat diuji dari data teramati.
6. **Laporkan rentang, bukan hanya titik**, terutama saat yang hilang banyak atau sampel kecil.
