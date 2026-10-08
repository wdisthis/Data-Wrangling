# Modul 3 — Penanganan Noise dan Outlier

**Praktikum Data Wrangling (SD25-20007) · Program Studi Sains Data ITERA**

Workbench menyimpan notebook ini di laptop sebagai **`dw-module-03/03_outlier.ipynb`**; seluruh
keluaran modul ditulis ke folder `dw-module-03/` itu. Modul ini membaca keluaran Modul 2 Anda
sendiri dari **`dw-module-02/keluaran/`** (`transaksi_m02.parquet` dan `sensor_m02.parquet`).
Dataset mentah disiapkan lewat panel *Dataset di laptop* halaman modul (Local Runner menyala).
Buku ajar: Bab 2.2.2, 3.4.2, 3.5.2, 4.1, 4.6.6–4.6.7.

| Bagian | Isi notebook ini | Dinilai? |
|---|---|---|
| A–D Prosedur terbimbing | **kode lengkap dan pembahasan** | dijalankan dan lulus checkpoint |
| Latihan 3.1–3.2 | **kode lengkap (sudah dikerjakan)** | **ya** — *Latihan Mandiri di Lab* (20% nilai modul) |
| Tugas Individu | **kode lengkap (sudah dikerjakan)** | **ya** — *Tugas Individu* (80% nilai modul) |

Bagian A–D bukan untuk disalin tanpa dibaca. Latihan dan tugas memakai **cara berpikir yang
sama** pada data lain (filter Hampel pada sensor, beberapa detektor pada ongkir dan total bayar),
dan pertanyaan analisis menanyakan alasan di balik setiap langkah A–D.

**Aturan main**
1. `data/raw/` dan keluaran Modul 2 bersifat *read-only* — jangan menulis ke sana.
2. Jangan menghapus baris. Pencilan diberi status, bukan dibuang.
3. Nilai teramati hanya berubah lewat kolom `<kolom>_metode`; pemeriksa membandingkannya dengan data mentah.
4. Setiap perubahan data dicatat pada `log-keputusan.csv` di akar workspace.
5. Sebelum mengumpulkan: *Kernel → Restart Kernel and Run All Cells*, lalu jalankan Checkpoint.

> **Versi Markdown.** Isi notebook `03_outlier.ipynb` ditulis ulang sebagai teks; setiap sel kode menjadi satu blok ` ```python `, dijalankan berurutan dari atas ke bawah. Keluaran sel tidak disertakan karena dihasilkan saat dijalankan. Bagian *Latihan 3.1–3.2* dan *Tugas Individu* sudah berisi kode lengkap menggantikan kerangka `TODO`. Kode baru memerlukan `scikit-learn` dan `pyarrow`.

## Bagian A — Mengukur ekstrem secara jujur

### A.1 Menyiapkan notebook

```python
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

# Kernel DSWorkbench menjalankan sel di luar thread utama, sehingga backend GUI
# (mis. "macosx") gagal membuat figure. Di luar Jupyter/ipykernel dipakai Agg;
# figure yang terbuka tetap ditampilkan kernel sebagai gambar di bawah sel.
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
    warnings.filterwarnings("ignore", message=r"FigureCanvas\w* is non-interactive")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / "data" / "raw").is_dir())
RAW = ROOT / "data" / "raw"
M02 = ROOT / "dw-module-02" / "keluaran"   # keluaran Modul 2 milik Anda sendiri
MODUL = ROOT / "dw-module-03"      # folder tempat Workbench menyimpan notebook ini
OUT = MODUL / "keluaran"
OUT.mkdir(parents=True, exist_ok=True)
LOG = ROOT / "log-keputusan.csv"
SEED = 2026
pd.set_option("display.width", 140)

kurang = [b for b in ["transaksi_m02.parquet", "sensor_m02.parquet"] if not (M02 / b).exists()]
if kurang:
    raise FileNotFoundError(f"Belum ada di {M02}: {', '.join(kurang)}. "
                            "Selesaikan dan jalankan ulang notebook Modul 2 lebih dulu.")
t2 = pd.read_parquet(M02 / "transaksi_m02.parquet")
s2 = pd.read_parquet(M02 / "sensor_m02.parquet")
mentah = pd.read_csv(RAW / "transaksi.csv", dtype=str, keep_default_na=False)
produk = pd.read_csv(RAW / "produk.csv", dtype=str, keep_default_na=False)
assert len(t2) == len(mentah) and t2.id_pesanan.astype(str).equals(mentah.id_pesanan), \
    "transaksi_m02 tidak sejajar dengan data mentah; periksa ulang Modul 2"
print("Akar repositori :", ROOT.name)
print("transaksi_m02   :", t2.shape, "| sensor_m02:", s2.shape)
```

```python
# Tampilan angka untuk analisis. Kolom pada t2 tidak diubah.
num = pd.DataFrame({k: pd.to_numeric(t2[k]) for k in ["jumlah", "harga_satuan", "diskon_persen", "total_bayar"]})
num["ongkir"] = t2.ongkir.astype(float)
status_bersih = t2.status.str.lower()          # hanya untuk analisis; kolom status tidak diubah


def rp(x: float) -> str:
    """Rupiah berformat Indonesia untuk teks bukti: 1234567 -> 'Rp1.234.567'."""
    return f"Rp{x:,.0f}".replace(",", ".")


def angka(x: float, d: int = 1) -> str:
    """Angka berformat Indonesia untuk teks bukti: 90.66 -> '90,7'."""
    return f"{x:.{d}f}".replace(".", ",")


num.describe().T.round(1)
```

### A.2 Tiga detektor univariat

```python
def deteksi(x: pd.Series, ambang_z: float = 3.0, k_iqr: float = 1.5,
            ambang_rz: float = 3.5) -> pd.DataFrame:
    """Tiga detektor univariat. Mengembalikan mask per metode; data tidak diubah."""
    x = x.astype(float)
    z = (x - x.mean()) / x.std()
    q1, q3 = x.quantile([.25, .75])
    iqr = q3 - q1
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0:
        warnings.warn(f"{x.name}: MAD = 0, robust z tidak terdefinisi")
        rz = pd.Series(np.nan, index=x.index)
    else:
        rz = 0.6745 * (x - med) / mad
    return pd.DataFrame({"z": z.abs() > ambang_z,
                         "iqr": (x < q1 - k_iqr * iqr) | (x > q3 + k_iqr * iqr),
                         "robust_z": rz.abs() > ambang_rz}, index=x.index)


tanda = {k: deteksi(num[k]) for k in ["harga_satuan", "jumlah", "total_bayar"]}
pd.DataFrame({k: f.sum() for k, f in tanda.items()}).T
```

### A.3 Pencilan terhadap kelompoknya: harga katalog

```python
def rupiah(teks: pd.Series) -> pd.Series:
    """Hanya untuk analisis: 'Rp1.330.500', '1.330.500,00', 'Rp 79.000', '74500' -> angka.
    Penyeragaman permanen harga_katalog adalah pekerjaan Modul 4."""
    bersih = (teks.str.replace("Rp", "", regex=False).str.strip()
                  .str.replace(r",00$", "", regex=True).str.replace(".", "", regex=False))
    return pd.to_numeric(bersih, errors="raise")


katalog = pd.Series(rupiah(produk.harga_katalog).to_numpy(), index=produk.kode_produk)
harga_katalog = t2.kode_produk.map(katalog)      # NaN bila kode tidak ada di katalog
rasio = num.harga_satuan / harga_katalog
rasio.round(2).value_counts(dropna=False).head(10)
```

```python
salah_harga = rasio.eq(100)
tanpa_katalog = harga_katalog.isna()
h = num.harga_satuan
B = {}   # angka bukti yang dipakai ulang pada tabel, log, dan pembahasan

B["n_x100"] = int(salah_harga.sum())
B["n_tanpa_katalog"] = int(tanpa_katalog.sum())
print(f"rasio tepat 100: {B['n_x100']} baris | tanpa harga katalog: {B['n_tanpa_katalog']} baris")
print("rasio selain 1 dan 100:", int((rasio.notna() & ~rasio.isin([1, 100])).sum()))

# Masking: kesalahan besar menggelembungkan sd sehingga ambang 3 sd ikut naik
B["sd_dengan"], B["sd_tanpa"] = h.std(), h[~salah_harga].std()
B["batas_dengan"] = h.mean() + 3 * h.std()
B["batas_tanpa"] = h[~salah_harga].mean() + 3 * h[~salah_harga].std()
z_lolos = salah_harga & ~tanda["harga_satuan"]["z"]
B["z_tangkap"] = int((salah_harga & tanda["harga_satuan"]["z"]).sum())
B["z_lolos"] = int(z_lolos.sum())
B["x100_min"], B["sah_max"] = h[salah_harga].min(), h[~salah_harga].max()
B["x100_di_bawah_sah_max"] = int((h[salah_harga] < B["sah_max"]).sum())
z_tanpa = (h[~salah_harga] - h[~salah_harga].mean()) / h[~salah_harga].std()
B["z_tanpa_tanda"] = int((z_tanpa.abs() > 3).sum())
print(f"sd harga: {B['sd_dengan']:,.0f} (dengan x100) vs {B['sd_tanpa']:,.0f} (tanpa)")
print(f"batas 3 sd: Rp{B['batas_dengan']:,.0f} vs Rp{B['batas_tanpa']:,.0f}")
print(f"z-score menangkap {B['z_tangkap']} dan meloloskan {B['z_lolos']} dari {B['n_x100']} harga x100")
print(f"harga x100 terkecil Rp{B['x100_min']:,.0f}; harga sah termahal Rp{B['sah_max']:,.0f}; "
      f"{B['x100_di_bawah_sah_max']} harga x100 di bawahnya")
print(f"z-score pada harga tanpa x100 masih menandai {B['z_tanpa_tanda']} harga sah")

label = {"harga_satuan": salah_harga, "jumlah": num.jumlah.lt(0),
         "total_bayar": salah_harga | num.jumlah.lt(0)}
baris = []
for kolom, f in tanda.items():
    metode = dict(f.items())
    if kolom == "harga_satuan":
        metode["rasio_katalog"] = rasio.notna() & rasio.ne(1)
    if kolom == "jumlah":
        metode["aturan_domain"] = num.jumlah.lt(1)
    for nama, m in metode.items():
        baris.append({"kolom": kolom, "metode": nama, "tertandai": int(m.sum()),
                      "kesalahan_tertangkap": int((m & label[kolom]).sum()),
                      "kesalahan_total": int(label[kolom].sum()),
                      "nilai_sah_tertandai": int((m & ~label[kolom]).sum())})
deteksi_pencilan = pd.DataFrame(baris)
deteksi_pencilan.to_csv(OUT / "deteksi_pencilan_m03.csv", index=False)
deteksi_pencilan
```

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6))
bins = np.linspace(3.5, 8.5, 70)
ax1.hist(np.log10(h[~salah_harga]), bins=bins, color="#9bb7d4", label="rasio katalog 1")
ax1.hist(np.log10(h[salah_harga]), bins=bins, color="#b2182b", label="rasio katalog 100")
for nama, nilai, gaya in [("batas 3 sd", B["batas_dengan"], "--"),
                          ("pagar IQR atas", h.quantile(.75) + 1.5 * (h.quantile(.75) - h.quantile(.25)), ":")]:
    ax1.axvline(np.log10(nilai), color="k", ls=gaya, lw=.9, label=nama)
ax1.set_xlabel("log10 harga_satuan (Rp)"); ax1.set_ylabel("baris"); ax1.set_yscale("log")
ax1.legend(frameon=False, fontsize=8); ax1.set_title("Ambang global vs kesalahan x100")
ada = harga_katalog.notna()
ax2.scatter(harga_katalog[ada & ~salah_harga], h[ada & ~salah_harga], s=4, color="#9bb7d4", label="rasio 1")
ax2.scatter(harga_katalog[salah_harga], h[salah_harga], s=8, color="#b2182b", label="rasio 100")
ax2.set_xscale("log"); ax2.set_yscale("log")
ax2.set_xlabel("harga katalog (Rp)"); ax2.set_ylabel("harga_satuan (Rp)")
ax2.legend(frameon=False, fontsize=8); ax2.set_title("Terhadap kelompoknya: harga katalog")
fig.tight_layout()
fig.savefig(OUT / "01_deteksi_harga.png", dpi=110, bbox_inches="tight")
plt.show()
```

**Pembahasan A**

| `harga_satuan` | Tertandai | Harga ×100 tertangkap (dari 119) | Harga sah ikut ditandai |
|---|---:|---:|---:|
| z-score (\|z\| > 3) | 46 | **46** | 0 |
| IQR (1,5 × IQR) | 3.730 | 119 | **3.611** |
| robust z (MAD, > 3,5) | 3.973 | 119 | **3.854** |
| rasio terhadap harga katalog ≠ 1 | 119 | 119 | **0** |

- **Masking.** 119 harga tepat 100 kali harga katalog menggelembungkan simpangan baku dari
  Rp261.305 menjadi **Rp2.401.479**, sehingga batas 3 sd naik dari Rp947.826 ke **Rp7.453.095**.
  Akibatnya z-score hanya menangkap 46 kesalahan dan **meloloskan 73 dari 119**. Persis contoh
  buku (Bab 3.4.2): pada data `[10, 12, 12, 13, 11, 50]`, nilai 50 hanya ber-z 2,04.
- IQR dan robust z menangkap semuanya, tetapi tenggelam di antara **3,6–3,9 ribu harga sah**
  (kain tenun dan kerajinan memang mahal). Tanpa kesalahan pun, z-score masih menandai 910 harga
  sah. Tidak ada ambang global yang memisahkan: harga ×100 terkecil (Rp1.400.000) lebih murah
  daripada harga sah termahal (Rp1.498.500).
- **Pencilan terhadap kelompoknya** menyelesaikan masalah: rasio `harga_satuan / harga katalog`
  hanya bernilai 1 (24.092 baris) atau tepat **100** (119 baris), tanpa nilai antara. 77 baris
  berkode produk di luar katalog (A-16) tidak dapat dihitung rasionya dan diperiksa pada B.1.
- `jumlah`: z-score menandai 84 baris, **seluruhnya pembelian besar yang sah, dan tidak satu pun
  dari 70 jumlah negatif** (z negatif hanya −0,95 sampai −0,46). IQR/robust z menandai 97 = 84
  pembelian besar + 13 negatif terbesar. Yang menemukan ke-70 negatif hanya aturan domain
  `jumlah ≥ 1`.
- `total_bayar`: z-score menandai 421 baris, hanya 2 di antaranya terkait kesalahan. Total bayar
  **tidak ikut** membesar 100 kali, petunjuk pertama bahwa harga ×100 adalah salah catat harga,
  bukan pembelian mahal.

*Kesalahan lazim:* menyimpulkan "z-score menemukan 46 pencilan harga, hapus" atau memakai IQR
pada seluruh harga lalu mengoreksi 3.730 baris.

## Bagian B — Kesalahan atau pencilan sah?

### B.1 Harga seratus kali lipat

```python
# Identitas kamus data: total_bayar = jumlah x harga_satuan x (1 - diskon/100) + ongkir.
# Diskon pecahan (A-07) dibaca sebagai persen HANYA untuk analisis; kolomnya ditangani Modul 4.
diskon_efektif = num.diskon_persen.where(~num.diskon_persen.between(0, 1, inclusive="neither"),
                                         num.diskon_persen * 100)


def selisih_identitas(jumlah: pd.Series, harga: pd.Series) -> pd.Series:
    """total_bayar dikurangi nilai yang seharusnya menurut identitas; 0 berarti cocok."""
    return num.total_bayar - (jumlah * harga * (1 - diskon_efektif / 100) + num.ongkir)


ongkir_teramati = t2.ongkir_metode.astype(str).eq("teramati")
cocok = lambda s: s.abs() <= 1                      # noqa: E731  (toleransi pembulatan Rp1)
```

```python
sel_asli = selisih_identitas(num.jumlah, h)
sel_bagi = selisih_identitas(num.jumlah, h / 100)
B["x100_cocok_bagi"] = int((salah_harga & cocok(sel_bagi)).sum())
B["x100_cocok_asli"] = int((salah_harga & cocok(sel_asli)).sum())
B["x100_ongkir_isian"] = int((salah_harga & ~ongkir_teramati).sum())
print(f"harga x100: identitas cocok bila harga dibagi 100 = {B['x100_cocok_bagi']}, "
      f"bila apa adanya = {B['x100_cocok_asli']} (dari {B['n_x100']}; ongkir isian {B['x100_ongkir_isian']})")
print(t2.loc[salah_harga & ~cocok(sel_bagi), ["id_pesanan", "kode_produk", "jumlah", "harga_satuan",
                                               "ongkir", "ongkir_metode", "total_bayar"]])
# Kode produk tanpa katalog: periksa lewat identitas, karena rasio tidak dapat dihitung
B["yatim_cocok"] = int((tanpa_katalog & cocok(sel_asli)).sum())
B["yatim_sel"] = sel_asli[tanpa_katalog & ~cocok(sel_asli)].round().tolist()
print(f"tanpa katalog: identitas cocok {B['yatim_cocok']} dari {B['n_tanpa_katalog']}; sisanya selisih {B['yatim_sel']}")

harga_baru = h.where(~salah_harga, h / 100)
harga_metode = pd.Series(np.where(salah_harga, "koreksi_bagi_100", "teramati"), index=t2.index)
harga_status = pd.Series(np.where(salah_harga, "kesalahan_dikoreksi", "wajar"), index=t2.index)
assert (harga_baru[salah_harga] == harga_katalog[salah_harga]).all()
pd.crosstab(harga_status, harga_metode)
```

### B.2 Jumlah negatif

```python
negatif = num.jumlah.lt(0)
print("jumlah negatif:", int(negatif.sum()))
pd.crosstab(negatif, status_bersih, margins=True)
```

```python
# Bila negatif adalah retur, total_bayar akan negatif atau cocok dengan jumlah bertanda.
sel_tanda = selisih_identitas(num.jumlah, harga_baru)
sel_mutlak = selisih_identitas(num.jumlah.abs(), harga_baru)
B["neg"] = int(negatif.sum())
B["neg_total_negatif"] = int((negatif & num.total_bayar.lt(0)).sum())
B["neg_cocok_tanda"] = int((negatif & cocok(sel_tanda)).sum())
B["neg_cocok_mutlak"] = int((negatif & cocok(sel_mutlak)).sum())
B["neg_ongkir_isian"] = int((negatif & ~ongkir_teramati).sum())
# Pada baris yang ongkirnya hasil imputasi Modul 2, identitas tidak dapat menjadi bukti
# (ongkir itu sendiri ditebak). Bukti yang tersisa: ongkir turunan sah (>= 0, kelipatan 500).
ongkir_turunan = num.total_bayar - num.jumlah.abs() * harga_baru * (1 - diskon_efektif / 100)
turunan_sah = ongkir_turunan.ge(0) & ongkir_turunan.round().mod(500).eq(0)
bukti_tanda = negatif & np.where(ongkir_teramati, cocok(sel_mutlak), turunan_sah)
koreksi_tanda = negatif & bukti_tanda
kosongkan = negatif & ~bukti_tanda
B["neg_koreksi"], B["neg_kosong"] = int(koreksi_tanda.sum()), int(kosongkan.sum())
print(f"total_bayar negatif: {B['neg_total_negatif']} | cocok dengan jumlah bertanda: {B['neg_cocok_tanda']} | "
      f"cocok dengan |jumlah|: {B['neg_cocok_mutlak']} dari {B['neg']} (ongkir isian {B['neg_ongkir_isian']})")
print("tidak lolos bukti:")
print(t2.loc[kosongkan, ["id_pesanan", "jumlah", "harga_satuan", "diskon_persen", "ongkir",
                         "ongkir_metode", "total_bayar", "status"]].assign(selisih=sel_mutlak[kosongkan]))

# Apakah negatif = retur? Bandingkan porsi status dikembalikan
tabel = pd.crosstab(negatif, status_bersih.eq("dikembalikan"))
B["neg_retur"] = int((negatif & status_bersih.eq("dikembalikan")).sum())
B["p_retur_neg"] = 100 * B["neg_retur"] / B["neg"]
B["p_retur_lain"] = 100 * status_bersih.eq("dikembalikan")[~negatif].mean()
B["p_fisher"] = stats.fisher_exact(tabel).pvalue
print(f"dikembalikan: {B['p_retur_neg']:.1f}% baris negatif vs {B['p_retur_lain']:.1f}% baris lain, "
      f"Fisher p = {B['p_fisher']:.2f}")

z_j = tanda["jumlah"]["z"]
B["neg_z"] = int((negatif & z_j).sum())
B["neg_iqr"] = int((negatif & tanda["jumlah"]["iqr"]).sum())
print(f"z-score menandai {B['neg_z']} dan IQR {B['neg_iqr']} dari {B['neg']} jumlah negatif")
```

### B.3 Pembelian besar

```python
kategori = t2.kode_produk.map(pd.Series(produk.kategori.str.strip().str.title().to_numpy(),
                                        index=produk.kode_produk))   # hanya untuk analisis
num.jumlah.value_counts().sort_index().to_frame("baris").T
```

```python
besar = num.jumlah.ge(50)            # celah: tidak ada jumlah 6-49
B["grosir"] = int(besar.sum())
B["grosir_min"], B["grosir_max"] = int(num.jumlah[besar].min()), int(num.jumlah[besar].max())
B["jumlah_max_biasa"] = int(num.jumlah[~besar].max())
B["grosir_kategori"] = kategori[besar].value_counts().to_dict()
B["grosir_cocok"] = int((besar & cocok(selisih_identitas(num.jumlah, harga_baru))).sum())
B["grosir_z"] = int((besar & z_j).sum())
B["z_j_total"] = int(z_j.sum())
print(f"jumlah >= 50: {B['grosir']} baris ({B['grosir_min']}-{B['grosir_max']}); jumlah biasa maksimum {B['jumlah_max_biasa']}")
print("kategori:", B["grosir_kategori"], "| identitas cocok:", B["grosir_cocok"])
print(f"z-score jumlah menandai {B['z_j_total']} baris, {B['grosir_z']} di antaranya pembelian besar ini")
# Ongkir per kg pada pembelian besar mengikuti tarif yang sama dengan pembelian biasa?
prov = t2.provinsi_tujuan
print("median ongkir, provinsi Lampung: biasa", num.ongkir[~besar & prov.eq("Lampung")].median(),
      "| besar", num.ongkir[besar & prov.eq("Lampung")].median())
pembelian_grosir = besar.copy()
jumlah_baru = num.jumlah.abs().astype("Int64").mask(kosongkan)
jumlah_metode = pd.Series(np.select([koreksi_tanda, kosongkan], ["koreksi_tanda", "dikosongkan"], "teramati"),
                          index=t2.index)
jumlah_status = pd.Series(np.select([koreksi_tanda, kosongkan, besar],
                                    ["kesalahan_dikoreksi", "kesalahan_dikosongkan", "pencilan_sah"], "wajar"),
                          index=t2.index)
# Urutan langkah: ongkir Modul 2 yang diisi median karena prasyarat deduktif gagal akibat
# harga x100 atau jumlah negatif kini dapat diturunkan (tidak diubah di sini; dicatat untuk pipeline)
isian_median = t2.ongkir_metode.astype(str).eq("median_provinsi")
turunan_baru = num.total_bayar - jumlah_baru.astype("Float64") * harga_baru * (1 - diskon_efektif / 100)
kini_turun = (isian_median & (salah_harga | negatif) & turunan_baru.ge(0)
              & turunan_baru.round().mod(500).eq(0)).fillna(False).astype(bool)
beda_turun = (turunan_baru - num.ongkir)[kini_turun]
B["ongkir_kini_turun"], B["ongkir_beda"] = int(kini_turun.sum()), beda_turun[beda_turun.abs() > 1].round().tolist()
B["grosir_ongkir_isian_selisih"] = selisih_identitas(num.jumlah, harga_baru)[besar & isian_median].round().tolist()
print(f"ongkir isian median Modul 2 yang kini dapat diturunkan: {B['ongkir_kini_turun']}; "
      f"berbeda dari median: {B['ongkir_beda']}")
print("grosir dengan ongkir isian median, selisih identitas:", B["grosir_ongkir_isian_selisih"])
pd.crosstab(jumlah_status, jumlah_metode)
```

### B.4 Lama kirim yang panjang

```python
lama = t2.lama_kirim_hari.astype("Float64")
pagar_atas = lama.quantile(.75) + 1.5 * (lama.quantile(.75) - lama.quantile(.25))
print("pagar IQR atas lama kirim:", pagar_atas, "hari")
pd.crosstab(pd.cut(lama, [0, pagar_atas, 17, 30], labels=["<= pagar", "pagar-17", "18-30"]), t2.kurir)
```

```python
lama_panjang = lama.gt(pagar_atas).fillna(False).astype(bool)
B["lama_pagar"] = float(pagar_atas)
B["lama_tertandai"] = int(lama_panjang.sum())
B["lama_18"] = int(lama.ge(18).fillna(False).sum())
B["lama_18_pos"] = int((lama.ge(18).fillna(False) & t2.kurir.eq("Pos Indonesia")).sum())
print(f"lama kirim > {pagar_atas:g} hari: {B['lama_tertandai']} | >= 18 hari: {B['lama_18']} "
      f"({B['lama_18_pos']} Pos Indonesia)")
print("rating teramati rata-rata, >= 18 hari vs lainnya:",
      round(t2.rating_ulasan[lama.ge(18).fillna(False)].astype(float).mean(), 2),
      round(t2.rating_ulasan[lama.lt(18).fillna(False)].astype(float).mean(), 2))
lama_status = pd.Series(np.select([lama.isna().to_numpy(dtype=bool), lama_panjang], ["tidak_berlaku", "pencilan_sah"],
                                  "wajar"), index=t2.index)
lama_status.value_counts()
```

```python
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
lr = np.log10(rasio.dropna())
axes[0].hist(lr, bins=60, color="#4c72b0"); axes[0].set_yscale("log")
axes[0].set_xlabel("log10 (harga_satuan / harga katalog)"); axes[0].set_ylabel("baris")
axes[0].set_title("Rasio terhadap katalog: 1 atau 100")
axes[1].hist([sel_tanda[negatif] / 1e3, sel_mutlak[negatif] / 1e3], bins=30,
             label=["jumlah bertanda", "|jumlah|"], color=["#b2182b", "#4c72b0"])
axes[1].set_xlabel("selisih identitas total_bayar (ribu Rp)"); axes[1].legend(frameon=False, fontsize=8)
axes[1].set_title("Jumlah negatif: total dihitung dengan |jumlah|")
kr = kategori.isin(["Kopi", "Rempah"])
axes[2].hist([num.jumlah[besar & kr], num.jumlah[besar & ~kr]], bins=20, stacked=True,
             label=["Kopi/Rempah", "lainnya"], color=["#8c6d31", "#bbbbbb"])
axes[2].set_xlabel("jumlah (unit), hanya >= 50"); axes[2].legend(frameon=False, fontsize=8)
axes[2].set_title("Pembelian besar menurut kategori")
fig.tight_layout()
fig.savefig(OUT / "02_bukti_kesalahan.png", dpi=110, bbox_inches="tight")
plt.show()
```

**Pembahasan B**

**B.1 Harga ×100.** Identitas `total_bayar` cocok dengan harga **dibagi 100** pada 118 dari 119
baris, dan **tidak pernah** cocok dengan harga apa adanya. Satu baris sisanya berongkir isian
median Modul 2 (paket berat, ongkir diisi terlalu kecil), sehingga identitasnya tidak dapat
menjadi saksi; rasio katalog tetap bukti utama. Keputusan: koreksi deduktif ÷100
(`harga_satuan_metode = koreksi_bagi_100`, status `kesalahan_dikoreksi`, 119 baris); harga
terkoreksi sama persis dengan harga katalog. Pada 77 baris tanpa katalog, identitas cocok dengan
harga apa adanya pada 76 baris dan satu sisanya meleset tepat Rp10.000 (A-08): **tidak ada harga
×100 tersembunyi** di antara kode yatim.

**B.2 Jumlah negatif (70 baris).**

| Pemeriksaan | Hasil |
|---|---|
| `total_bayar` negatif | **0** |
| identitas cocok dengan jumlah bertanda | **0** |
| identitas cocok dengan \|jumlah\| | **68** (4 baris berongkir isian Modul 2) |
| status `dikembalikan` | 5,7% vs 4,0% pada baris lain, Fisher p = 0,37; 61 dari 70 berstatus selesai |

- Negatif **bukan retur**: retur akan bertotal negatif atau setidaknya berstatus dikembalikan.
  Total dihitung dengan jumlah positif, jadi tandanya salah tulis.
- Pada baris yang ongkirnya **diisi** Modul 2, identitas melingkar (ongkir itu sendiri tebakan).
  Bukti yang dipakai di sana: ongkir turunan `total_bayar − |jumlah| × harga × (1 − d)` sah
  (≥ 0, kelipatan Rp500). Hasil: **69 dikoreksi tanda**, **1 dikosongkan** (identitas meleset tepat
  Rp10.000, A-08 pada baris yang sama; dua kesalahan pada satu baris).
- Mahasiswa yang lupa diskon pecahan (A-07) memperoleh 65 cocok, bukan 68: tiga baris negatif
  berdiskon pecahan. Koreksi 68–70 diterima bila aturannya dinyatakan.

**B.3 Pembelian besar.** Distribusi `jumlah` berhenti di 5, lalu kosong sampai **50**: 84 baris
50–198 unit, **seluruhnya Kopi (45) dan Rempah (39)**. Identitas cocok pada 82 baris (satu +Rp10.000,
satu berongkir isian median yang meleset Rp83.500), dan ongkirnya mengikuti berat (median Lampung
Rp191.000 vs Rp12.000 untuk pembelian biasa). Ini transaksi nyata: status `pencilan_sah` dan bendera
`pembelian_grosir`. Catatan sampingan: 10 ongkir yang diisi median pada Modul 2 karena harga ×100
atau jumlah negatif kini dapat diturunkan; 2 di antaranya berbeda dari median (Rp88.000 dan
Rp20.500 lebih besar). Urutan langkah penting: **koreksi kesalahan nilai sebelum imputasi deduktif**.

**B.4 Lama kirim.** Pagar IQR atas 7 hari menandai 373 baris; 115 di antaranya ≥ 18 hari dan 114
Pos Indonesia (satu J&T tepat 18 hari). Rating teramati pada pengiriman ≥ 18 hari rata-rata **1,09**
vs 4,25 pada lainnya: justru bukti layanan buruk yang harus dilaporkan, bukan dibuang. Status
`pencilan_sah` (373), nilai tidak diubah.

## Bagian C — Noise dan anomali pada deret waktu sensor

### C.1 Peta harian

```python
s = s2.sort_values(["id_gudang", "waktu"]).reset_index(drop=True)
mungkin = s.suhu_c.lt(50)          # 85,0 mustahil di gudang berpendingin (Modul 2)
harian = (s[mungkin].groupby(["id_gudang", s.waktu.dt.floor("D")]).suhu_c
          .agg(["min", "median", "max", "std"]))
fig, ax = plt.subplots(figsize=(11, 3.2))
harian["max"].unstack(0).plot(ax=ax, lw=.9)
ax.set_ylabel("suhu maksimum harian (°C)"); ax.set_xlabel("tanggal")
plt.show()
harian.sort_values("max", ascending=False).head(5)
```

```python
B["hari_puncak"] = harian["max"].idxmax()
B["puncak"] = float(harian["max"].max())
print("hari dengan maksimum tertinggi:", B["hari_puncak"], B["puncak"])
print("simpangan baku harian terendah:")
print(harian.sort_values("std").head(3))
```

### C.2 Menghaluskan: rata-rata bergulir, median bergulir, EWMA

```python
BATAS_ATAS = 8.0                    # batas atas suhu operasional gudang berpendingin
g = s.groupby("id_gudang").suhu_c
s["rata_bergulir"] = g.transform(lambda x: x.rolling(6, min_periods=1).mean())     # 1 jam ke belakang
s["median_bergulir"] = g.transform(lambda x: x.rolling(6, min_periods=1).median())
s["ewma"] = g.transform(lambda x: x.ewm(span=6).mean())
deret = ["suhu_c", "rata_bergulir", "median_bergulir", "ewma"]
pd.DataFrame({k: s[k].gt(BATAS_ATAS).groupby(s.id_gudang).sum() for k in deret})
```

```python
lonjakan = s.suhu_c.eq(85)
hari_k = s.id_gudang.eq(B["hari_puncak"][0]) & s.waktu.dt.floor("D").eq(B["hari_puncak"][1])
alarm = {}
for k in deret:
    a = s[k].gt(BATAS_ATAS)
    alarm[k] = {"titik_alarm": int(a.sum()), "di_luar_hari_kejadian": int((a & ~hari_k).sum()),
                "akibat_lonjakan": int((a & ~hari_k & ~lonjakan).sum())}
    pk = s.loc[hari_k, k]
    alarm[k]["puncak_hari_kejadian"] = round(float(pk.max()), 2)
    alarm[k]["pertama_lewat_batas"] = s.loc[hari_k & a, "waktu"].min()
    alarm[k]["terakhir_lewat_batas"] = s.loc[hari_k & a, "waktu"].max()
alarm = pd.DataFrame(alarm).T
B["alarm"] = alarm
alarm
```

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.6))
i0 = int(np.flatnonzero(lonjakan & s.id_gudang.eq("GDG-BDL"))[2])
w = s.iloc[i0 - 12:i0 + 18]
for k, gaya in zip(deret, ["-", "--", "-", ":"]):
    ax1.plot(w.waktu, w[k].clip(upper=30), gaya, lw=1.2, label=k)
ax1.axhline(BATAS_ATAS, color="gray", lw=.7)
ax1.set_ylim(0, 30); ax1.set_ylabel("suhu (°C), dipotong 30")
ax1.set_title(f"Satu lonjakan 85,0 ({w.id_gudang.iloc[0]})"); ax1.legend(frameon=False, fontsize=8)
ax1.tick_params(axis="x", rotation=30)
w = s[hari_k]
for k, gaya in zip(deret, ["-", "--", "-", ":"]):
    ax2.plot(w.waktu, w[k], gaya, lw=1.2, label=k)
ax2.axhline(BATAS_ATAS, color="gray", lw=.7)
ax2.set_title("Hari kejadian GDG-PLM"); ax2.tick_params(axis="x", rotation=30)
fig.tight_layout()
fig.savefig(OUT / "03_smoothing_sensor.png", dpi=110, bbox_inches="tight")
plt.show()
```

### C.3 Filter Hampel

```python
def hampel(x: pd.Series, jendela: int = 13, k: float = 3.0) -> tuple[pd.Series, pd.Series]:
    """Tandai titik yang menyimpang lebih dari k x 1,4826 x MAD dari median bergulir terpusat."""
    med = x.rolling(jendela, center=True, min_periods=1).median()
    mad = (x - med).abs().rolling(jendela, center=True, min_periods=1).median()
    return (x - med).abs() > k * 1.4826 * mad, med


def hampel_per_gudang(jendela: int = 13, k: float = 3.0) -> tuple[pd.Series, pd.Series]:
    tanda_h, median_h = pd.Series(False, index=s.index), pd.Series(np.nan, index=s.index)
    for _, idx in s.groupby("id_gudang").groups.items():
        tanda_h[idx], median_h[idx] = hampel(s.loc[idx, "suhu_c"], jendela, k)
    return tanda_h, median_h


pintu = s.pintu_terbuka.fillna(0).astype(int)
# titik pintu terbuka dan dua titik sesudahnya (kenaikan sah yang turun dalam tiga titik)
jejak_pintu = pintu.groupby(s.id_gudang).transform(lambda p: p.rolling(3, min_periods=1).max()).eq(1)
tanda_h, median_h = hampel_per_gudang(13, 3.0)
jenis = np.select([s.suhu_c.eq(85), jejak_pintu], ["lonjakan 85", "jejak pintu"], "lainnya")
pd.crosstab(jenis, tanda_h, margins=True)
```

```python
B["h3_tanda"] = int(tanda_h.sum())
B["h3_lonjakan"] = int((tanda_h & lonjakan).sum())
B["h3_pintu"] = int((tanda_h & jejak_pintu & ~lonjakan).sum())
B["h3_lain"] = int((tanda_h & ~jejak_pintu & ~lonjakan).sum())
B["lonjakan_di_jejak_pintu"] = int((lonjakan & jejak_pintu).sum())
B["pintu_titik"] = int(pintu.eq(1).sum())
B["pintu_titik_ditandai"] = int((pintu.eq(1) & tanda_h & ~lonjakan).sum())
print(f"Hampel (13, 3): {B['h3_tanda']} tanda = {B['h3_lonjakan']} lonjakan + {B['h3_pintu']} jejak pintu "
      f"+ {B['h3_lain']} lainnya")
print(f"titik pintu terbuka {B['pintu_titik']}, {B['pintu_titik_ditandai']} di antaranya ditandai; "
      f"{B['lonjakan_di_jejak_pintu']} lonjakan 85 jatuh di jejak pintu")
```

### C.4 Sensor macet dan kenaikan berkelanjutan

```python
def panjang_run(x: pd.Series) -> pd.Series:
    """Panjang deretan nilai identik berturut-turut tempat setiap titik berada."""
    blok = x.ne(x.shift()).cumsum()
    return x.groupby(blok).transform("size")


s["run_identik"] = g.transform(panjang_run)
s["sd_1jam"] = g.transform(lambda x: x.rolling(6).std())
s["baseline"] = g.transform(lambda x: x.rolling(144, center=True, min_periods=72).median())  # 24 jam
s.groupby("id_gudang")[["run_identik"]].max()
```

```python
macet = s.run_identik.ge(6) & s.suhu_c.notna()
B["sd_macet"] = float(s.sd_1jam[macet].min())
print("sd 1 jam tepat 0 (==):", int(s.sd_1jam.eq(0).sum()), "| < 1e-6:", int(s.sd_1jam.lt(1e-6).sum()),
      f"| sd pada deret identik: {B['sd_macet']:.1e}")


def run_mask(m: pd.Series) -> pd.Series:
    """Panjang run True per gudang (0 bila False)."""
    blok = (m.ne(m.shift()) | s.id_gudang.ne(s.id_gudang.shift())).cumsum()
    return m.groupby(blok).transform("sum").where(m, 0)


# Kenaikan berkelanjutan: median 1 jam lebih dari 2 °C di atas baseline 24 jam, selama >= 1 jam;
# lalu diperluas ke titik bersebelahan yang masih > 0,5 °C di atas baseline (median terpusat 7 titik).
inti = (s.median_bergulir - s.baseline).gt(2.0)
inti &= run_mask(inti).ge(6)
med_pusat = g.transform(lambda x: x.rolling(7, center=True, min_periods=1).median())
di_atas = (med_pusat - s.baseline).gt(0.5) & ~lonjakan
blok_atas = (di_atas.ne(di_atas.shift()) | s.id_gudang.ne(s.id_gudang.shift())).cumsum()
kenaikan = di_atas & inti.groupby(blok_atas).transform("any")

kejadian = []
for (gudang, jenis_k), m in [(("GDG-BDL", "sensor_macet"), macet), (("GDG-PLM", "kegagalan_kompresor"), kenaikan)]:
    for gd, d in s[m].groupby("id_gudang"):
        kejadian.append({"id_gudang": gd, "mulai": d.waktu.min(), "selesai": d.waktu.max(),
                         "jenis": jenis_k, "n_titik": len(d)})
print(pd.DataFrame(kejadian))
```

```python
celah = s.suhu_c.isna() & run_mask(s.suhu_c.isna()).ge(4)
baris_k = []
for _, d in s[macet].groupby("id_gudang"):
    baris_k.append({"id_gudang": d.id_gudang.iloc[0], "mulai": d.waktu.min(), "selesai": d.waktu.max(),
                    "jenis": "sensor_macet", "n_titik": len(d),
                    "bukti": f"{len(d)} titik identik {angka(d.suhu_c.iloc[0], 2)} °C berturut-turut; sd 1 jam praktis 0",
                    "tindakan": "suhu_c_bersih dikosongkan; suhu sebenarnya tidak diketahui"})
for _, d in s[kenaikan].groupby("id_gudang"):
    puncak = d.loc[d.suhu_c.idxmax()]
    baris_k.append({"id_gudang": d.id_gudang.iloc[0], "mulai": d.waktu.min(), "selesai": d.waktu.max(),
                    "jenis": "kegagalan_kompresor", "n_titik": len(d),
                    "bukti": (f"naik berkelanjutan sampai {angka(puncak.suhu_c, 2)} °C pukul {puncak.waktu:%H.%M}; "
                              f"{int(d.suhu_c.gt(BATAS_ATAS).sum())} titik di atas {BATAS_ATAS:g} °C; "
                              f"median 1 jam > baseline + 2 °C selama {int(inti[d.index].sum())} titik"),
                    "tindakan": "anomali_nyata: tidak dihaluskan, tidak dikoreksi; dilaporkan ke operasional"})
for _, d in s[celah].groupby("id_gudang"):
    baris_k.append({"id_gudang": d.id_gudang.iloc[0], "mulai": d.waktu.min(), "selesai": d.waktu.max(),
                    "jenis": "celah_data", "n_titik": len(d),
                    "bukti": f"{len(d)} titik tanpa nilai (baris tidak terkirim, tidak diisi Modul 2)",
                    "tindakan": "tetap kosong"})
for i in np.flatnonzero(lonjakan):
    baris_k.append({"id_gudang": s.id_gudang[i], "mulai": s.waktu[i], "selesai": s.waktu[i],
                    "jenis": "lonjakan", "n_titik": 1,
                    "bukti": f"85,00 °C; median Hampel {angka(median_h[i], 2)} °C; mustahil di gudang berpendingin",
                    "tindakan": "dikoreksi ke median Hampel pada suhu_c_bersih"})
kejadian = (pd.DataFrame(baris_k).sort_values(["id_gudang", "mulai"]).reset_index(drop=True))
kejadian.to_csv(OUT / "kejadian_sensor_m03.csv", index=False)
B["kejadian"] = kejadian
kejadian.groupby("jenis").agg(baris=("n_titik", "size"), titik=("n_titik", "sum"))
```

```python
fig, axes = plt.subplots(1, 3, figsize=(14, 3.4))
for ax, (gudang, mulai, selesai, judul) in zip(axes, [
        ("GDG-PLM", "2026-06-03 06:00", "2026-06-04 06:00", "Kegagalan kompresor"),
        ("GDG-BDL", "2026-06-10 02:00", "2026-06-10 20:00", "Sensor macet"),
        ("GDG-MDN", "2026-05-17 20:00", "2026-05-18 14:00", "Celah data (Modul 2)")]):
    w = s[s.id_gudang.eq(gudang) & s.waktu.between(mulai, selesai)]
    ax.plot(w.waktu, w.suhu_c.where(w.suhu_c < 50), lw=1, label="suhu_c")
    ax.plot(w.waktu, w.baseline, lw=1, ls="--", color="gray", label="baseline 24 jam")
    k = kejadian[kejadian.id_gudang.eq(gudang) & kejadian.jenis.ne("lonjakan")]
    for _, r in k.iterrows():
        if r.selesai >= pd.Timestamp(mulai) and r.mulai <= pd.Timestamp(selesai):
            ax.axvspan(r.mulai, r.selesai, color="#e07b39", alpha=.25)
    ax.set_title(f"{judul} — {gudang}"); ax.tick_params(axis="x", rotation=30)
axes[0].set_ylabel("suhu (°C)"); axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "04_kejadian_sensor.png", dpi=110, bbox_inches="tight")
plt.show()
```

**Pembahasan C**

**C.1.** Maksimum harian tertinggi: **GDG-PLM 3 Juni 2026, 13,46 °C** (hari lain di bawah 8,4 °C).
Simpangan baku harian tidak menonjolkan sensor macet; yang menemukannya panjang deretan nilai
identik di C.4.

**C.2 Smoothing** (jendela 1 jam ke belakang, batas atas operasional 8 °C):

| Deret | Titik > 8 °C | Di luar hari kejadian, bukan titik 85 itu sendiri | Puncak 3 Juni | Pertama > 8 °C |
|---|---:|---:|---:|---|
| `suhu_c` | 74 | 1 | 13,46 | 15.10 |
| rata-rata bergulir | 283 | **210** | 12,94 | 15.30 |
| median bergulir | 31 | **0** | 12,82 | 15.30 |
| EWMA (span 6) | 276 | **202** | 12,77 | 15.30 |

- Rata-rata bergulir dan EWMA **mengubah 42 lonjakan 85,0 menjadi 210 dan 202 titik alarm palsu**
  (satu lonjakan menaikkan rata-rata 1 jam ke ±17,6 °C selama enam titik). Median bergulir
  kebal: tidak satu pun.
- Pada kegagalan kompresor semua penghalus **menurunkan puncak** (13,46 → 12,8–12,9 °C) dan
  **menunda alarm 20 menit**. Menghaluskan kejadian nyata menyembunyikan apa yang justru harus
  dilihat.

**C.3 Hampel (13 titik, k = 3).** 1.710 tanda = 42 lonjakan (seluruhnya) + **1.014 jejak pintu**
+ 654 lainnya. Dari 535 titik pintu terbuka, 528 ditandai: kenaikan +2,5 °C yang sah tampak
seperti lonjakan bagi filter yang tidak tahu konteks. Aturan pengecualian harus memakai kolom
`pintu_terbuka` (titik itu dan dua titik sesudahnya), **tetapi** 4 lonjakan 85,0 jatuh di jejak
pintu dan tetap kesalahan: pengecualian tidak boleh membebaskan nilai yang mustahil.

**C.4 Kejadian** (`kejadian_sensor_m03.csv`, 45 baris):

| Jenis | Gudang | Mulai–selesai | Titik | Bukti |
|---|---|---|---:|---|
| `sensor_macet` | GDG-BDL | 10 Jun 08.00–13.50 | 36 | 36 nilai identik 4,20 °C |
| `kegagalan_kompresor` | GDG-PLM | 3 Jun 13.30–21.50 | 51 | puncak 13,46 °C; 31 titik > 8 °C; median 1 jam > baseline + 2 °C selama 41 titik |
| `celah_data` | GDG-MDN | 18 Mei 02.00–07.50 | 36 | baris tidak terkirim (Modul 2) |
| `lonjakan` | ketiganya | 42 titik tunggal | 42 | BDL 18, MDN 15, PLM 9 |

- **Simpangan baku bergulir dari 36 nilai identik bukan 0, melainkan 2,6 × 10⁻⁷** (galat
  floating point). `sd == 0` menemukan nol titik; pakai ambang kecil (`< 1e-6`) atau panjang
  deretan nilai identik.
- Hampel **tidak** menandai sensor macet (nilai konstan = median). Sebaliknya ia menandai
  titik **di dalam** kegagalan kompresor (10 titik pada jendela 13, k = 5), karena MAD runtuh ke 0
  pada kenaikan yang monoton.
  Kejadian nyata harus dilindungi oleh tabel kejadian, bukan oleh nilai `k`.
- Batas kejadian kompresor ditentukan dari data, sehingga berbeda beberapa titik antar-mahasiswa
  (±30 menit di kedua ujung diterima). Yang dinilai: kenaikan **berkelanjutan** dibedakan dari
  lonjakan tunggal, dan kejadian itu tidak dihaluskan.

## Bagian D — Menangani dan mengevaluasi

### D.1 Empat strategi

```python
j, h = num.jumlah, num.harga_satuan
nilai_barang = lambda jj, hh: jj * hh * (1 - diskon_efektif / 100)        # noqa: E731
pembanding = num.total_bayar - num.ongkir      # nilai barang menurut total_bayar
z_h = (h - h.mean()) / h.std()
z_jml = (j - j.mean()) / j.std()
semua = pd.Series(True, index=t2.index)
strategi = {
    "biarkan": (j, h, semua),
    "hapus_di_luar_3sd": (j, h, (z_h.abs() <= 3) & (z_jml.abs() <= 3)),
    "winsorize_p1_p99": (j.clip(*j.quantile([.01, .99])), h.clip(*h.quantile([.01, .99])), semua),
}
```

```python
strategi["koreksi_dan_bendera"] = (jumlah_baru.astype("Float64"), harga_baru, semua)
baris = []
for nama, (jj, hh, simpan) in strategi.items():
    nb = nilai_barang(jj, hh)[simpan]
    baris.append({"strategi": nama, "baris_tersisa": int(simpan.sum()),
                  "rata_harga": round(float(hh[simpan].mean()), 0),
                  "median_harga": round(float(hh[simpan].median()), 0),
                  "sd_harga": round(float(hh[simpan].std()), 0),
                  "nilai_barang_miliar": round(float(nb.sum()) / 1e9, 3)})
perbandingan = pd.DataFrame(baris)
B["pembanding_miliar"] = float(pembanding.sum()) / 1e9
perbandingan["selisih_dari_total_bayar_persen"] = (
    100 * (perbandingan.nilai_barang_miliar / B["pembanding_miliar"] - 1)).round(2)
perbandingan.to_csv(OUT / "perbandingan_strategi_m03.csv", index=False)
B["perbandingan"] = perbandingan.set_index("strategi")
hapus = ~strategi["hapus_di_luar_3sd"][2]
B["hapus_baris"] = int(hapus.sum())
B["hapus_x100"] = int((hapus & salah_harga).sum())
B["hapus_grosir"] = int((hapus & besar).sum())
B["hapus_x100_lolos"] = int((~hapus & salah_harga).sum())
print(f"pembanding (total_bayar - ongkir): Rp{B['pembanding_miliar']:.3f} miliar")
print(f"hapus 3 sd: {B['hapus_baris']} baris terbuang = {B['hapus_x100']} x100 + {B['hapus_grosir']} grosir sah; "
      f"{B['hapus_x100_lolos']} x100 tetap tinggal")
perbandingan
```

### D.2 Menyimpan `transaksi_m03.parquet`

```python
hasil = t2.copy()
hasil["harga_satuan"] = harga_baru.astype(float)
hasil["harga_satuan_metode"] = harga_metode
hasil["harga_satuan_status_m03"] = harga_status
hasil["jumlah"] = jumlah_baru
hasil["jumlah_metode"] = jumlah_metode
hasil["jumlah_status_m03"] = jumlah_status
hasil["pembelian_grosir"] = pembelian_grosir.astype(bool)
hasil["lama_kirim_hari_status_m03"] = lama_status
```

```python
STATUS_M03 = {"wajar", "pencilan_sah", "kesalahan_dikoreksi", "kesalahan_dikosongkan",
              "anomali_nyata", "tidak_berlaku"}
assert len(hasil) == len(t2) and hasil.id_pesanan.equals(t2.id_pesanan)
for k in ["jumlah", "harga_satuan"]:
    assert set(hasil[f"{k}_status_m03"]) <= STATUS_M03, f"{k}: status di luar nilai baku"
    tetap = hasil[f"{k}_metode"].eq("teramati")
    assert np.allclose(hasil.loc[tetap, k].astype(float), num.loc[tetap, k]), f"{k}: nilai teramati berubah"
assert not (hasil.jumlah.astype("Float64") < 0).any()
hasil.to_parquet(OUT / "transaksi_m03.parquet", index=False)
print(hasil.shape, "->", (OUT / "transaksi_m03.parquet").name)
hasil[["jumlah_status_m03", "harga_satuan_status_m03"]].apply(pd.Series.value_counts).fillna(0).astype(int)
```

### D.3 Log keputusan

```python
import shutil

KOLOM_LOG = ["id", "modul", "sumber_kolom", "temuan", "bukti", "tindakan", "alasan",
             "baris_terdampak", "id_anomali"]


def catat_keputusan(baris: list[dict], path: Path = LOG) -> pd.DataFrame:
    """Tulis ulang baris log dengan id yang sama agar notebook aman dijalankan berulang."""
    lama_log = (pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists()
                else pd.DataFrame(columns=KOLOM_LOG))
    baru = pd.DataFrame(baris, columns=KOLOM_LOG).astype(str)
    log = pd.concat([lama_log[~lama_log.id.isin(baru.id)], baru]).sort_values("id")
    log.to_csv(path, index=False)
    if path.parent != MODUL:
        shutil.copyfile(path, MODUL / path.name)   # salinan untuk ZIP tugas
    return log
```

```python
pb = B["perbandingan"]
log = catat_keputusan([
    {"id": "K-03-001", "modul": 3, "sumber_kolom": "transaksi.harga_satuan",
     "temuan": "harga_satuan tepat 100 kali harga katalog produknya",
     "bukti": (f"{B['n_x100']} baris rasio tepat 100; identitas total_bayar cocok bila harga dibagi 100 pada "
               f"{B['x100_cocok_bagi']} baris dan tidak pernah cocok dengan harga apa adanya"),
     "tindakan": "harga_satuan dibagi 100; harga_satuan_metode=koreksi_bagi_100, harga_satuan_status_m03=kesalahan_dikoreksi",
     "alasan": "koreksi deduktif: nilai benar diketahui dari katalog dan dikonfirmasi total_bayar; menghapus baris membuang transaksi nyata",
     "baris_terdampak": B["n_x100"], "id_anomali": "A-06"},
    {"id": "K-03-002", "modul": 3, "sumber_kolom": "transaksi.harga_satuan",
     "temuan": "detektor global gagal pada harga: masking dan produk mahal yang sah",
     "bukti": (f"z-score meloloskan {B['z_lolos']} dari {B['n_x100']} harga x100 (sd {rp(B['sd_dengan'])} vs "
               f"{rp(B['sd_tanpa'])} tanpa x100); IQR menandai "
               f"{int(tanda['harga_satuan']['iqr'].sum())} baris, robust z {int(tanda['harga_satuan']['robust_z'].sum())}"),
     "tindakan": "pencilan harga ditentukan terhadap harga katalog per kode_produk, bukan ambang global; harga sah tidak diubah",
     "alasan": f"harga x100 terkecil {rp(B['x100_min'])} di bawah harga sah termahal {rp(B['sah_max'])}; tidak ada ambang global yang memisahkan",
     "baris_terdampak": 0, "id_anomali": "A-06"},
    {"id": "K-03-003", "modul": 3, "sumber_kolom": "transaksi.jumlah",
     "temuan": "jumlah negatif adalah salah tanda, bukan retur",
     "bukti": (f"{B['neg']} baris; total_bayar cocok dengan |jumlah| ({B['neg_cocok_mutlak']}) dan tidak pernah dengan "
               f"jumlah bertanda ({B['neg_cocok_tanda']}); dikembalikan {angka(B['p_retur_neg'])}% vs "
               f"{angka(B['p_retur_lain'])}% (Fisher p={angka(B['p_fisher'], 2)})"),
     "tindakan": "tanda dibalik bila identitas cocok (ongkir teramati) atau ongkir turunan sah (ongkir isian); jumlah_metode=koreksi_tanda",
     "alasan": "nilai benar dapat diturunkan dari total_bayar; z-score tidak menandai satu pun jumlah negatif",
     "baris_terdampak": B["neg_koreksi"], "id_anomali": "A-05"},
    {"id": "K-03-004", "modul": 3, "sumber_kolom": "transaksi.jumlah",
     "temuan": "jumlah negatif yang tidak didukung identitas total_bayar",
     "bukti": f"{B['neg_kosong']} baris: identitas meleset {', '.join(rp(x) for x in sel_mutlak[kosongkan])} dengan |jumlah|",
     "tindakan": "jumlah dikosongkan (NA); jumlah_metode=dikosongkan, jumlah_status_m03=kesalahan_dikosongkan",
     "alasan": "dua kolom bermasalah pada satu baris; membalik tanda tanpa bukti sama dengan menebak",
     "baris_terdampak": B["neg_kosong"], "id_anomali": "A-05;A-08"},
    {"id": "K-03-005", "modul": 3, "sumber_kolom": "transaksi.jumlah",
     "temuan": f"pembelian {B['grosir_min']}-{B['grosir_max']} unit: pencilan statistik yang sah",
     "bukti": (f"{B['grosir']} baris, seluruhnya {' dan '.join(B['grosir_kategori'])}; identitas cocok "
               f"{B['grosir_cocok']}; tidak ada jumlah 6-49; z-score menandai {B['z_j_total']} baris, "
               f"{B['grosir_z']} di antaranya pembelian ini"),
     "tindakan": "nilai dipertahankan; jumlah_status_m03=pencilan_sah dan bendera pembelian_grosir",
     "alasan": "menghapus atau memotong menghilangkan omzet nyata kopi dan rempah; bendera membiarkan analisis hilir memilih",
     "baris_terdampak": 0, "id_anomali": "A-17"},
    {"id": "K-03-006", "modul": 3, "sumber_kolom": "transaksi.lama_kirim_hari",
     "temuan": f"lama kirim di atas pagar IQR {angka(B['lama_pagar'], 1)} hari",
     "bukti": f"{B['lama_tertandai']} baris di atas pagar; {B['lama_18']} baris >= 18 hari, {B['lama_18_pos']} Pos Indonesia",
     "tindakan": "nilai dipertahankan; lama_kirim_hari_status_m03=pencilan_sah",
     "alasan": "pengiriman lambat adalah layanan nyata yang justru perlu dilaporkan, bukan kesalahan pencatatan",
     "baris_terdampak": 0, "id_anomali": "-"},
    {"id": "K-03-007", "modul": 3, "sumber_kolom": "transaksi.total_bayar",
     "temuan": "total_bayar dipakai sebagai bukti, tidak dikoreksi",
     "bukti": (f"identitas cocok pada {B['x100_cocok_bagi']} harga x100 setelah dibagi 100 dan {B['neg_cocok_mutlak']} "
               f"jumlah negatif dengan |jumlah|; {B['ongkir_kini_turun']} ongkir isian median Modul 2 kini dapat "
               f"diturunkan, {len(B['ongkir_beda'])} berbeda ({', '.join(rp(x) for x in B['ongkir_beda'])})"),
     "tindakan": "tidak diubah; selisih +Rp10.000 dan diskon pecahan diserahkan ke Modul 4; urutan koreksi-sebelum-imputasi dicatat untuk pipeline",
     "alasan": "kolom yang menjadi saksi tidak boleh ikut dikoreksi dengan aturan yang sama; A-07/A-08 milik Modul 4",
     "baris_terdampak": 0, "id_anomali": "A-07;A-08"},
    {"id": "K-03-008", "modul": 3, "sumber_kolom": "transaksi.harga_satuan;transaksi.jumlah",
     "temuan": "strategi penanganan dibandingkan dengan nilai barang menurut total_bayar",
     "bukti": (f"selisih dari pembanding: biarkan {angka(pb.loc['biarkan', 'selisih_dari_total_bayar_persen'], 2)}%, "
               f"hapus 3 sd {angka(pb.loc['hapus_di_luar_3sd', 'selisih_dari_total_bayar_persen'], 2)}%, "
               f"winsorize {angka(pb.loc['winsorize_p1_p99', 'selisih_dari_total_bayar_persen'], 2)}%, "
               f"koreksi {angka(pb.loc['koreksi_dan_bendera', 'selisih_dari_total_bayar_persen'], 2)}%"),
     "tindakan": "koreksi deduktif dan bendera; tidak ada baris yang dihapus atau dipotong",
     "alasan": (f"hapus 3 sd membuang {B['hapus_grosir']} grosir sah dan meloloskan {B['hapus_x100_lolos']} harga x100; "
                "winsorize memotong harga sah dan mempertahankan kesalahan"),
     "baris_terdampak": 0, "id_anomali": "A-05;A-06;A-17"},
])
log[log.id.str.startswith("K-03")][["id", "sumber_kolom", "tindakan", "baris_terdampak"]]
```

**Pembahasan D**

**D.1 Empat strategi** (pembanding: nilai barang menurut `total_bayar − ongkir` = Rp9,327 miliar):

| Strategi | Baris tersisa | Rata-rata harga | sd harga | Nilai barang (miliar) | Selisih dari pembanding |
|---|---:|---:|---:|---:|---:|
| biarkan | 24.288 | 248.658 | 2.401.479 | 13,082 | **+40,27%** |
| hapus di luar 3 sd | 24.158 | 175.724 | 346.842 | 9,051 | −2,95% |
| winsorize p1–p99 | 24.288 | 169.758 | 273.563 | 8,849 | −5,12% |
| **koreksi dan bendera** | 24.288 | 163.963 | 261.477 | 9,325 | **−0,02%** |

- Median harga Rp63.000 pada keempat strategi: **median tidak melihat masalahnya sama sekali**.
  Rata-rata dan sd yang menunjukkannya.
- "Hapus di luar 3 sd" tampak dekat (−2,95%) karena dua kesalahan saling menutupi: membuang 130
  baris = 46 harga ×100 + **84 pembelian grosir sah**, sementara **73 harga ×100 tetap tinggal**.
- Winsorize memotong harga sah termahal ke persentil 99 (Rp1.405.500) dan mengganti harga ×100
  dengan batas itu, bukan dengan harga yang benar; harga ×100 terkecil (Rp1.400.000, seharusnya
  Rp14.000) bahkan lolos di bawah batas.
- Hanya koreksi deduktif yang mendekati pembanding; sisa −0,02% berasal dari satu jumlah yang
  dikosongkan dan baris +Rp10.000 (A-08).

**D.2** `transaksi_m03.parquet`: 24.288 × 30, urutan sama dengan Modul 2. Status jumlah: 24.134
wajar, 84 pencilan sah, 69 dikoreksi, 1 dikosongkan; status harga: 24.169 wajar, 119 dikoreksi.

**D.3** Log K-03-001 sampai K-03-008. Tiga keputusan **tidak mengubah nilai** (K-03-002, K-03-005,
K-03-006) dan dua mencatat pembanding atau saksi (K-03-007, K-03-008), dengan
`baris_terdampak = 0`. Keputusan sensor (Tugas) dan Latihan 3.2 adalah pekerjaan Anda:
tambahkan sebagai **K-03-009** dan seterusnya.

## Latihan 3.1 — Memilih parameter filter Hampel

**Dinilai.** Kerjakan di laboratorium; jawabannya ditulis pada **`dw-module-03/latihan-hampel.md`** dan
diunggah ke tugas *Latihan Mandiri di Lab — Modul 03*. Keluarannya dipakai lagi pada Tugas.

1. Jalankan `hampel_per_gudang()` untuk beberapa jendela (misalnya 7, 13, dan 25 titik) dan
   beberapa nilai `k` (misalnya 2 sampai 10). Untuk setiap pasangan, hitung titik yang
   ditandai dan pisahkan menurut jenisnya: lonjakan 85,0, jejak pintu terbuka, titik di
   dalam kejadian pada `kejadian_sensor_m03.csv`, dan lainnya.
2. Simpan tabelnya ke `keluaran/parameter_hampel_m03.csv` dan gambar kurva banyak titik
   tertandai terhadap `k` ke `keluaran/05_kurva_hampel.png`.
3. Pilih satu pasangan (jendela, `k`) dan beri alasan dengan angka dari tabel. Pasangan
   yang dipilih harus menangkap seluruh lonjakan. Periksa kolom kejadian nyata: dapatkah
   `k` saja membuatnya nol? Bila tidak, aturan apa yang mencegah kejadian nyata dan
   **pintu terbuka** dikoreksi sebagai kesalahan?

```python
from sklearn.ensemble import IsolationForest


def md_tabel(df: pd.DataFrame) -> str:
    """DataFrame -> tabel Markdown tanpa dependensi tambahan (untuk file jawaban dan laporan)."""
    kepala = "| " + " | ".join(map(str, df.columns)) + " |"
    sekat = "|" + "|".join("---:" if pd.api.types.is_numeric_dtype(df[c]) else "---" for c in df.columns) + "|"
    isi = ["| " + " | ".join(str(v).replace("|", "\\|") for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([kepala, sekat, *isi])


# Langkah 1 -- titik di dalam kejadian nyata = sensor macet dan kegagalan kompresor pada tabel kejadian Bagian C
nyata = pd.Series(False, index=s.index)
for _, r in kejadian[kejadian.jenis.isin(["sensor_macet", "kegagalan_kompresor"])].iterrows():
    nyata |= s.id_gudang.eq(r.id_gudang) & s.waktu.between(r.mulai, r.selesai)
# jenis titik saling lepas, urutan prioritas: lonjakan 85 > kejadian nyata > jejak pintu > lainnya
jenis_titik = pd.Series(np.select([lonjakan, nyata, jejak_pintu],
                                  ["lonjakan 85", "kejadian nyata", "jejak pintu"], "lainnya"), index=s.index)
N_LONJAKAN, N_NYATA, N_PINTU = int(lonjakan.sum()), int(nyata.sum()), int(pintu.eq(1).sum())
print(f"lonjakan 85,0: {N_LONJAKAN} | titik di dalam kejadian nyata: {N_NYATA} | titik pintu terbuka: {N_PINTU}")

JENDELA_UJI, K_UJI = [7, 13, 25], [2, 3, 4, 5, 6, 8, 10]
baris = []
for jd in JENDELA_UJI:
    for kk in K_UJI:
        t, _ = hampel_per_gudang(jd, kk)
        t = t.astype(bool)
        per_jenis = jenis_titik[t].value_counts()
        baris.append({"jendela": jd, "k": kk, "ditandai": int(t.sum()),
                      "lonjakan_85": int(per_jenis.get("lonjakan 85", 0)),
                      "jejak_pintu": int(per_jenis.get("jejak pintu", 0)),
                      "kejadian_nyata": int(per_jenis.get("kejadian nyata", 0)),
                      "lainnya": int(per_jenis.get("lainnya", 0)),
                      "titik_pintu_ditandai": int((t & pintu.eq(1) & ~lonjakan).sum())})
parameter_hampel = pd.DataFrame(baris)
parameter_hampel["lonjakan_lolos"] = N_LONJAKAN - parameter_hampel.lonjakan_85

# Langkah 2 -- simpan tabel dan kurva banyak titik tertandai terhadap k
parameter_hampel.to_csv(OUT / "parameter_hampel_m03.csv", index=False)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6))
for jd, d in parameter_hampel.groupby("jendela"):
    ax1.plot(d.k, d.ditandai, marker="o", lw=1.1, label=f"jendela {jd}")
    ax2.plot(d.k, d.kejadian_nyata, marker="o", lw=1.1, label=f"jendela {jd}")
ax1.axhline(N_LONJAKAN, color="gray", ls="--", lw=.8, label=f"{N_LONJAKAN} lonjakan 85,0")
ax1.set_yscale("log"); ax1.set_xlabel("k"); ax1.set_ylabel("titik ditandai Hampel")
ax1.set_title("Semua titik tertandai"); ax1.legend(frameon=False, fontsize=8)
ax2.set_xlabel("k"); ax2.set_ylabel("titik ditandai")
ax2.set_title(f"Hanya titik di dalam kejadian nyata (n = {N_NYATA})"); ax2.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "05_kurva_hampel.png", dpi=110, bbox_inches="tight")
plt.show()

# Langkah 3 -- pasangan terpilih: menangkap seluruh lonjakan, lalu paling sedikit titik yang ditandai
layak = parameter_hampel[parameter_hampel.lonjakan_lolos.eq(0)]
assert len(layak), "tidak ada pasangan yang menangkap seluruh lonjakan; perlebar K_UJI"
pilih = layak.sort_values(["ditandai", "jendela", "k"]).iloc[0]
JENDELA_H, K_H = int(pilih.jendela), int(pilih.k)
print(f"{len(layak)} dari {len(parameter_hampel)} pasangan menangkap seluruh {N_LONJAKAN} lonjakan")
print(f"pilihan: JENDELA_H = {JENDELA_H}, K_H = {K_H} -> {int(pilih.ditandai)} titik ditandai "
      f"({int(pilih.jejak_pintu)} jejak pintu, {int(pilih.kejadian_nyata)} kejadian nyata, {int(pilih.lainnya)} lainnya)")
parameter_hampel
```

**Jawaban Latihan 3.1** — ditulis ke `latihan-hampel.md` oleh sel berikut: tabel lengkap dari `parameter_hampel_m03.csv`,
pasangan (jendela, `k`) pilihan, alasan dengan angka, jawaban atas pertanyaan *dapatkah `k` saja membuat kejadian nyata nol*,
dan aturan untuk pintu terbuka. Seluruh angka dalam teks berasal dari tabel, bukan diketik tangan.

```python
sel_p = parameter_hampel[(parameter_hampel.jendela == JENDELA_H) & (parameter_hampel.k == K_H)].iloc[0]
k_kecil = parameter_hampel[(parameter_hampel.jendela == JENDELA_H) & (parameter_hampel.k == min(K_UJI))].iloc[0]
min_nyata = int(layak.kejadian_nyata.min())
nol_nyata = layak[layak.kejadian_nyata.eq(0)]
if len(nol_nyata) == 0:
    k_saja = (f"**Tidak.** Pada seluruh {len(layak)} pasangan yang menangkap semua lonjakan, kejadian nyata "
              f"tetap ditandai paling sedikit {min_nyata} titik (pasangan terpilih: {int(sel_p.kejadian_nyata)} titik). "
              "Seperti pada C.4, MAD jendela runtuh pada kenaikan yang monoton, sehingga titik di dalamnya tampak "
              "menyimpang pada rentang k yang diuji.")
else:
    pasangan_nol = ", ".join(f"({int(a)}, {int(b)})" for a, b in nol_nyata[["jendela", "k"]].to_numpy())
    k_saja = (f"**Pada grid ini bisa, tetapi tidak dapat diandalkan.** Pasangan {pasangan_nol} menolkan kejadian nyata "
              f"sambil tetap menangkap {N_LONJAKAN} lonjakan, namun itu terjadi karena k dinaikkan sampai ambang melampaui "
              "simpangan kejadian, bukan karena filter memahami kejadiannya. Kejadian dengan simpangan lebih kecil atau "
              "lonjakan yang lebih landai akan ikut lolos; perlindungan tidak boleh bergantung pada k.")

jawaban_hampel = f"""# Latihan 3.1 — Memilih parameter filter Hampel

Praktikum Data Wrangling (SD25-20007) · Modul 3. Seluruh angka berasal dari `keluaran/parameter_hampel_m03.csv`
(sel Latihan 3.1 pada `03_outlier.ipynb`).

## 1. Tabel parameter

Grid: jendela {JENDELA_UJI} titik, k {K_UJI}. Data sensor memuat {N_LONJAKAN} lonjakan 85,0, {N_NYATA} titik di dalam
kejadian nyata (sensor macet dan kegagalan kompresor, `kejadian_sensor_m03.csv`), dan {N_PINTU} titik pintu terbuka.
Jenis titik saling lepas dengan prioritas lonjakan 85 → kejadian nyata → jejak pintu (titik pintu terbuka dan dua titik
sesudahnya) → lainnya.

{md_tabel(parameter_hampel)}

Kurva: `keluaran/05_kurva_hampel.png`.

## 2. Pasangan terpilih: jendela = {JENDELA_H} titik, k = {K_H}

- Menangkap **{int(sel_p.lonjakan_85)} dari {N_LONJAKAN}** lonjakan 85,0 (lolos {int(sel_p.lonjakan_lolos)}).
- Dari {len(layak)} pasangan yang menangkap seluruh lonjakan, pasangan ini menandai titik paling sedikit:
  **{int(sel_p.ditandai)}** titik (rentang {int(layak.ditandai.min())}–{int(layak.ditandai.max())}).
  Rinciannya {int(sel_p.jejak_pintu)} jejak pintu, {int(sel_p.kejadian_nyata)} kejadian nyata, {int(sel_p.lainnya)} lainnya.
- Pada jendela yang sama dengan k = {min(K_UJI)}, filter menandai {int(k_kecil.ditandai)} titik
  ({angka(k_kecil.ditandai / sel_p.ditandai)} kali lebih banyak) tanpa menangkap lonjakan tambahan: k kecil hanya menambah
  kandidat palsu.
- Hampel hanya **menandai kandidat**; nilai baru berubah bila aturan di bawah terpenuhi.

## 3. Kejadian nyata dan pintu terbuka

**Dapatkah k saja membuat kejadian nyata nol?** {k_saja}

Aturan yang dipakai pada Tugas Individu:

1. **Nilai mustahil** (85,0 °C di gudang berpendingin) yang ditandai Hampel adalah kesalahan: `kesalahan_dikoreksi`,
   `suhu_c_bersih` = median Hampel, `suhu_c` tidak diubah. Aturan ini berlaku juga bagi lonjakan yang jatuh di jejak pintu
   ({B['lonjakan_di_jejak_pintu']} titik): pengecualian pintu tidak boleh membebaskan nilai yang mustahil.
2. **Pintu terbuka bukan kesalahan.** Titik ber`pintu_terbuka` = 1 dan dua titik sesudahnya yang ditandai Hampel, tetapi
   tidak mustahil, berstatus `pencilan_sah` dan tidak diubah ({int(sel_p.titik_pintu_ditandai)} dari {N_PINTU} titik pintu
   terbuka ditandai pada pasangan terpilih).
3. **Kejadian nyata dilindungi oleh tabel kejadian, bukan oleh k.** Titik di dalam `kegagalan_kompresor` berstatus
   `anomali_nyata` (tidak dihaluskan, tidak dikoreksi, dilaporkan ke operasional) dan titik di dalam `sensor_macet`
   berstatus `kesalahan_dikosongkan`.
4. Titik lain yang ditandai Hampel tanpa bukti kesalahan tetap `pencilan_sah` dengan nilai dipertahankan.
"""
(MODUL / "latihan-hampel.md").write_text(jawaban_hampel, encoding="utf-8")
print("ditulis:", (MODUL / "latihan-hampel.md").relative_to(ROOT))
```

## Latihan 3.2 — Membandingkan detektor pada ongkir dan total bayar

**Dinilai.** Jawaban ditulis pada **`dw-module-03/latihan-multimetode.md`**.

Untuk `ongkir` dan `total_bayar`, **per provinsi tujuan**, jalankan empat detektor: z-score,
IQR, robust z (MAD), dan `IsolationForest` dari scikit-learn (`random_state=SEED`, dua fitur
log10 ongkir dan log10 total bayar). `provinsi_tujuan` belum diseragamkan (Modul 4): buat
pemetaan sementara khusus analisis. Susun **tabel kesepakatan antarmetode** (misalnya indeks
Jaccard antara himpunan tanda dua metode) dan simpan ke `keluaran/kesepakatan_metode_m03.csv`.
Jelaskan mengapa metode-metode itu tidak sepakat, apa yang terjadi pada robust z di kolom
ongkir, dan berapa banyak pembelian grosir sah yang ikut ditandai. Catat keputusan Anda pada log.

**Langkah 1 — pemetaan sementara provinsi.**

```python
import difflib
import itertools
import re
import unicodedata

# Langkah 1 -- pemetaan sementara provinsi_tujuan (hanya analisis; penyeragaman permanen adalah pekerjaan Modul 4)
PROVINSI = ["Aceh", "Sumatera Utara", "Sumatera Barat", "Riau", "Kepulauan Riau", "Jambi", "Sumatera Selatan",
            "Bangka Belitung", "Bengkulu", "Lampung", "DKI Jakarta", "Jawa Barat", "Banten", "Jawa Tengah",
            "DI Yogyakarta", "Jawa Timur", "Bali", "Nusa Tenggara Barat", "Nusa Tenggara Timur", "Kalimantan Barat",
            "Kalimantan Tengah", "Kalimantan Selatan", "Kalimantan Timur", "Kalimantan Utara", "Sulawesi Utara",
            "Gorontalo", "Sulawesi Tengah", "Sulawesi Barat", "Sulawesi Selatan", "Sulawesi Tenggara", "Maluku",
            "Maluku Utara", "Papua", "Papua Barat", "Papua Barat Daya", "Papua Selatan", "Papua Tengah",
            "Papua Pegunungan"]
ALIAS = {"nad": "Aceh", "nanggroeacehdarussalam": "Aceh", "sumut": "Sumatera Utara", "sumbar": "Sumatera Barat",
         "sumsel": "Sumatera Selatan", "kepri": "Kepulauan Riau", "kepriau": "Kepulauan Riau",
         "babel": "Bangka Belitung", "kepbabel": "Bangka Belitung", "kepbangkabelitung": "Bangka Belitung",
         "kepulauanbangkabelitung": "Bangka Belitung", "jakarta": "DKI Jakarta", "dki": "DKI Jakarta",
         "jkt": "DKI Jakarta", "jakartaraya": "DKI Jakarta", "jabar": "Jawa Barat", "jateng": "Jawa Tengah",
         "jatim": "Jawa Timur", "yogyakarta": "DI Yogyakarta", "jogja": "DI Yogyakarta", "jogjakarta": "DI Yogyakarta",
         "diy": "DI Yogyakarta", "daerahistimewayogyakarta": "DI Yogyakarta", "istimewayogyakarta": "DI Yogyakarta",
         "ntb": "Nusa Tenggara Barat", "ntt": "Nusa Tenggara Timur", "kalbar": "Kalimantan Barat",
         "kalteng": "Kalimantan Tengah", "kalsel": "Kalimantan Selatan", "kaltim": "Kalimantan Timur",
         "kaltara": "Kalimantan Utara", "sulut": "Sulawesi Utara", "sulteng": "Sulawesi Tengah",
         "sulsel": "Sulawesi Selatan", "sulbar": "Sulawesi Barat", "sultra": "Sulawesi Tenggara",
         "malut": "Maluku Utara", "pabar": "Papua Barat"}


def kompak(x) -> str:
    """Kunci pencocokan: huruf kecil tanpa tanda baca, spasi, dan kata 'provinsi' ('D.K.I. Jakarta' -> 'dkijakarta')."""
    t = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode().lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    t = re.sub(r"\b(provinsi|propinsi|prov)\b", " ", t)
    return re.sub(r"[^a-z0-9]", "", t)


KANON = {kompak(p): p for p in PROVINSI}


def petakan_provinsi(x) -> str:
    k = kompak(x)
    if k in KANON:
        return KANON[k]
    if k in ALIAS:
        return ALIAS[k]
    mirip = difflib.get_close_matches(k, list(KANON), n=1, cutoff=0.85)      # salah ketik ringan
    return KANON[mirip[0]] if mirip else "TIDAK_DIKENAL"


prov_mentah = t2.provinsi_tujuan.fillna("").astype(str)
peta = {v: petakan_provinsi(v) for v in prov_mentah.unique()}
prov_bersih = prov_mentah.map(peta)                 # hanya untuk analisis; t2.provinsi_tujuan tidak diubah
varian = (pd.DataFrame({"mentah": list(peta), "provinsi": list(peta.values())})
          .assign(baris=lambda d: d.mentah.map(prov_mentah.value_counts()))
          .sort_values("baris", ascending=False))
print(f"{len(peta)} penulisan berbeda -> {prov_bersih.nunique()} kelompok; "
      f"tidak dikenal: {int(prov_bersih.eq('TIDAK_DIKENAL').sum())} baris")
varian[varian.mentah.ne(varian.provinsi)].head(12)
```

**Langkah 2 — empat detektor per provinsi.**

```python
# Langkah 2 -- per provinsi: z, IQR, robust z (deteksi() Bagian A) pada ongkir dan total_bayar,
#           ditambah IsolationForest pada dua fitur log10 ongkir dan log10 total bayar
METODE = ["z", "iqr", "robust_z", "iforest"]
KOLOM_UJI = ["ongkir", "total_bayar"]
MIN_IF = 20                                          # kelompok lebih kecil dari ini tidak diberi IsolationForest
n_ongkir_nol = int((num.ongkir <= 0).sum())
fitur = pd.DataFrame({"log10_ongkir": np.log10(num.ongkir.clip(lower=1)),
                      "log10_total_bayar": np.log10(num.total_bayar.clip(lower=1))})
print(f"ongkir <= 0: {n_ongkir_nol} baris (dipotong ke Rp1 pada log10)")

tanda32 = {c: pd.DataFrame(False, index=t2.index, columns=METODE) for c in KOLOM_UJI}
mad_prov = []
for prov, idx in prov_bersih.groupby(prov_bersih).groups.items():
    for c in KOLOM_UJI:
        x = num.loc[idx, c]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")           # deteksi() memperingatkan bila MAD = 0; dihitung di bawah
            d = deteksi(x)
        tanda32[c].loc[idx, ["z", "iqr", "robust_z"]] = d[["z", "iqr", "robust_z"]].to_numpy()
        mad_prov.append({"provinsi": prov, "kolom": c, "n": len(x), "median": float(x.median()),
                         "mad": float((x - x.median()).abs().median())})
    if len(idx) >= MIN_IF:
        iso = IsolationForest(n_estimators=200, random_state=SEED).fit(fitur.loc[idx])
        bendera = iso.predict(fitur.loc[idx]) == -1
        for c in KOLOM_UJI:
            tanda32[c].loc[idx, "iforest"] = bendera
mad_prov = pd.DataFrame(mad_prov)

# banyak tanda per metode dan berapa pembelian grosir sah (B.3) yang ikut ditandai
ringkas = pd.DataFrame([{"kolom": c, "metode": m, "ditandai": int(f[m].sum()),
                         "persen_baris": round(100 * float(f[m].mean()), 2),
                         "grosir_ditandai": int((f[m] & besar).sum()), "grosir_total": int(besar.sum()),
                         "bukan_grosir_ditandai": int((f[m] & ~besar).sum())}
                        for c, f in tanda32.items() for m in METODE])
ringkas.to_csv(OUT / "tanda_metode_provinsi_m03.csv", index=False)
ringkas
```

**Langkah 3–4 — tabel kesepakatan, robust z pada ongkir, dan pembelian grosir.**

```python
# Langkah 3 -- tabel kesepakatan antarmetode: indeks Jaccard pada himpunan baris yang ditandai
def jaccard(a: pd.Series, b: pd.Series) -> float:
    gabungan = int((a | b).sum())
    return round(float((a & b).sum()) / gabungan, 4) if gabungan else np.nan


kesepakatan = pd.DataFrame([{"kolom": c, "metode_a": a, "metode_b": b,
                             "n_a": int(f[a].sum()), "n_b": int(f[b].sum()),
                             "irisan": int((f[a] & f[b]).sum()), "gabungan": int((f[a] | f[b]).sum()),
                             "jaccard": jaccard(f[a], f[b])}
                            for c, f in tanda32.items() for a, b in itertools.combinations(METODE, 2)])
kesepakatan.to_csv(OUT / "kesepakatan_metode_m03.csv", index=False)
for c in KOLOM_UJI:
    mat = pd.DataFrame(1.0, index=METODE, columns=METODE)
    for _, r in kesepakatan[kesepakatan.kolom.eq(c)].iterrows():
        mat.loc[r.metode_a, r.metode_b] = mat.loc[r.metode_b, r.metode_a] = r.jaccard
    print(f"Jaccard antarmetode, {c}:")
    print(mat.round(2), "\n")

# Langkah 4 -- robust z pada ongkir: kelompok provinsi dengan MAD = 0
mo = mad_prov[mad_prov.kolom.eq("ongkir")]
mad_nol_prov = mo.loc[mo.mad.eq(0), "provinsi"]
di_mad_nol = prov_bersih.isin(mad_nol_prov)
B["ong_prov"], B["ong_prov_mad0"], B["ong_baris_mad0"] = len(mo), len(mad_nol_prov), int(di_mad_nol.sum())
B["ong_rz_total"] = int(tanda32["ongkir"]["robust_z"].sum())
B["ong_rz_di_mad0"] = int(tanda32["ongkir"]["robust_z"][di_mad_nol].sum())
B["ong_iqr_di_mad0"] = int(tanda32["ongkir"]["iqr"][di_mad_nol].sum())
B["ong_rel_mad"] = float((mo.mad / mo["median"]).median())
print(f"ongkir: {B['ong_prov_mad0']} dari {B['ong_prov']} kelompok ber-MAD = 0 ({B['ong_baris_mad0']} baris); "
      f"robust z menandai {B['ong_rz_di_mad0']} baris di sana, IQR {B['ong_iqr_di_mad0']}")
print(f"robust z menandai {B['ong_rz_total']} baris ongkir secara keseluruhan; median MAD/median per provinsi "
      f"{B['ong_rel_mad']:.2f}")
ringkas[["kolom", "metode", "ditandai", "grosir_ditandai", "grosir_total"]]
```

**Jawaban Latihan 3.2** — ditulis ke `latihan-multimetode.md` oleh sel berikut: banyak tanda per metode, tabel
kesepakatan, penyebab ketidaksepakatan, apa yang terjadi pada robust z untuk ongkir, berapa grosir sah yang ikut ditandai,
dan keputusan. Keputusan dicatat pada log sebagai K-03-013.

```python
rr = ringkas.set_index(["kolom", "metode"])
jk = kesepakatan.set_index(["kolom", "metode_a", "metode_b"]).jaccard
terendah = {c: kesepakatan[kesepakatan.kolom.eq(c)].nsmallest(1, "jaccard").iloc[0] for c in KOLOM_UJI}
tertinggi = {c: kesepakatan[kesepakatan.kolom.eq(c)].nlargest(1, "jaccard").iloc[0] for c in KOLOM_UJI}
condong = {c: float(stats.skew(num[c])) for c in KOLOM_UJI}
if B["ong_prov_mad0"]:
    teks_rz = (f"Pada **{B['ong_prov_mad0']} dari {B['ong_prov']}** kelompok provinsi, MAD ongkir bernilai 0 (lebih dari separuh "
               f"ongkir berharga sama). `deteksi()` memberi robust z = NaN sehingga **tidak satu pun** dari "
               f"{B['ong_baris_mad0']} baris di kelompok itu ditandai, padahal IQR menandai {B['ong_iqr_di_mad0']} baris di "
               f"kelompok yang sama. Metode ini buta justru di kelompok yang ongkirnya seragam.")
else:
    teks_rz = (f"Tidak ada kelompok provinsi dengan MAD ongkir = 0, tetapi MAD sangat kecil dibanding median (median rasio "
               f"MAD/median per provinsi {angka(B['ong_rel_mad'], 2)}). Akibatnya ambang 3,5 sangat dekat ke median dan robust z "
               f"menandai {B['ong_rz_total']} baris ongkir ({angka(rr.loc[('ongkir', 'robust_z'), 'persen_baris'], 2)}% baris), "
               f"{int(rr.loc[('ongkir', 'robust_z'), 'grosir_ditandai'])} di antaranya pembelian grosir sah yang ongkirnya "
               f"memang mengikuti berat.")
def _urut(c):
    d = ringkas[ringkas.kolom.eq(c)].set_index("metode").ditandai
    return d.idxmin(), int(d.min()), d.idxmax(), int(d.max())


teks_banyak = []
for c in KOLOM_UJI:
    mn, nmn, mx, nmx = _urut(c)
    teks_banyak.append(f"{c}: paling sedikit {mn} ({nmn} baris), paling banyak {mx} ({nmx} baris); "
                       f"kemiringan sebaran {angka(condong[c], 1)}.")
teks_banyak = " ".join(teks_banyak)
if all(_urut(c)[0] == "z" for c in KOLOM_UJI) and all(condong[c] > 1 for c in KOLOM_UJI):
    teks_banyak += (" Sebaran menjulur ke kanan: ekor kanan menaikkan rata-rata dan simpangan baku sehingga ambang z-score "
                    "ikut naik (masking, Bagian A.3), sedangkan IQR dan robust z berpusat di median.")
grosir_baris = "\n".join(
    f"| {c} | " + " | ".join(f"{int(rr.loc[(c, m), 'grosir_ditandai'])}" for m in METODE) + " |" for c in KOLOM_UJI)
jawaban_multi = f"""# Latihan 3.2 — Membandingkan detektor pada ongkir dan total bayar

Praktikum Data Wrangling (SD25-20007) · Modul 3. Angka berasal dari `keluaran/tanda_metode_provinsi_m03.csv` dan
`keluaran/kesepakatan_metode_m03.csv` (sel Latihan 3.2 pada `03_outlier.ipynb`).

## 1. Cara kerja

- `provinsi_tujuan` dipetakan sementara (huruf kecil, tanpa tanda baca dan kata "provinsi", singkatan umum, salah ketik
  ringan): {len(peta)} penulisan menjadi {prov_bersih.nunique()} kelompok; {int(prov_bersih.eq('TIDAK_DIKENAL').sum())} baris
  tidak dikenal. Kolom aslinya tidak diubah.
- Per provinsi: z-score (|z| > 3), IQR (1,5 × IQR), robust z (MAD, > 3,5), dan `IsolationForest`
  (`random_state={SEED}`, 200 pohon, fitur log10 ongkir dan log10 total bayar). Kelompok kurang dari {MIN_IF} baris tidak
  diberi IsolationForest.
- IsolationForest bersifat bivariat, jadi himpunan tandanya sama untuk kolom `ongkir` dan `total_bayar`.

## 2. Banyak tanda per metode

{md_tabel(ringkas)}

## 3. Kesepakatan antarmetode (indeks Jaccard)

{md_tabel(kesepakatan)}

Kesepakatan tertinggi: ongkir {tertinggi['ongkir'].metode_a} dan {tertinggi['ongkir'].metode_b}
({angka(tertinggi['ongkir'].jaccard, 2)}); total_bayar {tertinggi['total_bayar'].metode_a} dan
{tertinggi['total_bayar'].metode_b} ({angka(tertinggi['total_bayar'].jaccard, 2)}). Terendah: ongkir
{terendah['ongkir'].metode_a} dan {terendah['ongkir'].metode_b} ({angka(terendah['ongkir'].jaccard, 2)}); total_bayar
{terendah['total_bayar'].metode_a} dan {terendah['total_bayar'].metode_b} ({angka(terendah['total_bayar'].jaccard, 2)}).

## 4. Mengapa metode tidak sepakat

- **Definisi "ekstrem" berbeda.** z-score mengukur jarak dari rata-rata dalam satuan simpangan baku; IQR dan robust z
  mengukur jarak dari bagian tengah sebaran memakai kuantil dan median; IsolationForest mengukur seberapa mudah baris
  dipisahkan di ruang dua fitur. Tiga yang pertama univariat per kolom, yang keempat melihat pasangan ongkir × total.
- **Banyak tanda berbeda jauh.** {teks_banyak}
- **Ambang tidak sebanding.** 3 sd, 1,5 × IQR, dan 3,5 robust z bukan tingkat kepercayaan yang sama; Jaccard rendah
  sebagian hanya berasal dari beda ambang, bukan beda "kebenaran".
- IsolationForest menandai baris yang tidak lazim sebagai **kombinasi** (mis. ongkir tinggi untuk total kecil), yang
  tidak terlihat oleh detektor satu kolom.

## 5. Robust z pada ongkir

{teks_rz}

## 6. Pembelian grosir sah yang ikut ditandai

Pembelian grosir (jumlah ≥ 50) berjumlah **{int(besar.sum())}** baris dan seluruhnya sah (B.3; identitas `total_bayar`
cocok pada {B['grosir_cocok']} baris).

| kolom | {" | ".join(METODE)} |
|---|{"---:|" * len(METODE)}
{grosir_baris}

Detektor hanya menandai kandidat; tidak satu pun tanda di atas merupakan bukti kesalahan.

## 7. Keputusan

Tidak ada nilai `ongkir` atau `total_bayar` yang diubah dan tidak ada baris yang dihapus. Tanda keempat metode dicatat
sebagai bahan pembanding, bukan perintah koreksi; pembelian grosir tetap `pencilan_sah` dengan bendera
`pembelian_grosir`. Bila analisis hilir membutuhkan satu daftar kandidat, pakai irisan dua metode atau lebih dan tetap
periksa terhadap identitas `total_bayar` dan kolom `pembelian_grosir`. Keputusan dicatat sebagai K-03-013.
"""
(MODUL / "latihan-multimetode.md").write_text(jawaban_multi, encoding="utf-8")

log_32 = catat_keputusan([
    {"id": "K-03-013", "modul": 3, "sumber_kolom": "transaksi.ongkir;transaksi.total_bayar",
     "temuan": "Latihan 3.2: empat detektor (z, IQR, robust z, IsolationForest) per provinsi pada ongkir dan total_bayar tidak sepakat",
     "bukti": (f"tanda ongkir z/IQR/robust z/IF = {'/'.join(str(int(rr.loc[('ongkir', m), 'ditandai'])) for m in METODE)}; "
               f"total_bayar = {'/'.join(str(int(rr.loc[('total_bayar', m), 'ditandai'])) for m in METODE)}; "
               f"grosir sah ditandai (ongkir) {'/'.join(str(int(rr.loc[('ongkir', m), 'grosir_ditandai'])) for m in METODE)} "
               f"dari {int(besar.sum())}; robust z ongkir: {B['ong_prov_mad0']} dari {B['ong_prov']} provinsi ber-MAD 0"),
     "tindakan": "tidak ada nilai diubah; tanda detektor tidak dipakai sebagai dasar koreksi; kesepakatan Jaccard disimpan ke kesepakatan_metode_m03.csv",
     "alasan": "pembelian grosir sah ikut ditandai dan tidak ada tanda yang didukung bukti kesalahan; menghapus atau memotongnya membuang omzet nyata",
     "baris_terdampak": 0, "id_anomali": "A-17"},
])
log_32[log_32.id.eq("K-03-013")][["id", "sumber_kolom", "tindakan", "baris_terdampak"]]
```

## Tugas Individu

Dikerjakan di luar sesi. Folder `dw-module-03/` (notebook, keluaran, laporan, jawaban latihan, dan
salinan `log-keputusan.csv` yang ditulis otomatis oleh `catat_keputusan()`) dikompres menjadi
**`dw-module-03.zip`** dan diunggah ke tugas *Tugas Individu — Modul 03*; empat pertanyaan analisis
dijawab langsung pada formulir tugas yang sama.

**1. Sensor.** Simpan `keluaran/sensor_m03.parquet`: grid Modul 2 apa adanya (seluruh kolomnya tetap),
ditambah `suhu_c_status_m03` (nilai baku: `wajar`, `pencilan_sah`, `kesalahan_dikoreksi`,
`kesalahan_dikosongkan`, `anomali_nyata`, `tidak_berlaku`), `suhu_c_bersih`, dan
`suhu_c_bersih_metode`. Pakai parameter Hampel dari Latihan 3.1 dan tabel kejadian Bagian C.
Kejadian nyata **tidak dihaluskan dan tidak dikoreksi**; pintu terbuka bukan kesalahan; `suhu_c`
tetap berisi nilai Modul 2. Catat keputusannya pada log (`sumber_kolom` memuat `sensor_gudang`).

*Urutan prioritas status bila satu titik memenuhi lebih dari satu aturan:* tidak ada nilai → nilai mustahil 85,0 → sensor macet → kenaikan kompresor → tanda Hampel lain → wajar. Lonjakan 85,0 yang jatuh di jejak pintu atau di dalam kejadian nyata tetap dikoreksi, karena pengecualian tidak boleh membebaskan nilai yang mustahil.

```python
# Isi dengan id temuan register Modul 1 Anda (modul_penanganan = 3) yang terkait tiap kejadian sensor
ID_ANOMALI = {"lonjakan": "-", "macet": "-", "kompresor": "-"}

# 1. Hampel dengan parameter Latihan 3.1 pada seluruh titik
tanda_p, median_p = hampel_per_gudang(JENDELA_H, K_H)
tanda_p = tanda_p.astype(bool)

# 2. Status per titik. Urutan prioritas (aturan pertama yang terpenuhi menang):
#    tidak ada nilai > nilai mustahil 85,0 > sensor macet > kenaikan kompresor > tanda Hampel lain > wajar
#    - lonjakan di atas kejadian nyata atau jejak pintu tetap dikoreksi: pengecualian tidak membebaskan nilai mustahil
#    - titik pintu terbuka dan pencilan Hampel lain tanpa bukti kesalahan -> pencilan_sah (nilai dipertahankan)
s["suhu_c_status_m03"] = np.select(
    [s.suhu_c.isna(), lonjakan, macet, kenaikan, tanda_p],
    ["tidak_berlaku", "kesalahan_dikoreksi", "kesalahan_dikosongkan", "anomali_nyata", "pencilan_sah"], "wajar")

# 3. suhu_c_bersih: hanya lonjakan dikoreksi (median Hampel) dan sensor macet dikosongkan; sisanya apa adanya
s["suhu_c_bersih"] = s.suhu_c.where(~(lonjakan | macet), np.nan)
s.loc[lonjakan, "suhu_c_bersih"] = median_p[lonjakan]
s["suhu_c_bersih_metode"] = np.select([lonjakan, macet], ["koreksi_median_hampel", "dikosongkan"], "teramati")

# 4. Grid Modul 2 apa adanya (urutan baris dan semua kolom), ditambah tiga kolom baru
tambahan = ["suhu_c_status_m03", "suhu_c_bersih", "suhu_c_bersih_metode"]
assert not set(tambahan) & set(s2.columns), "kolom baru bentrok dengan kolom Modul 2"
sensor_m03 = s2.merge(s[["id_gudang", "waktu", *tambahan]], on=["id_gudang", "waktu"], how="left",
                      validate="one_to_one")
assert len(sensor_m03) == len(s2)
assert sensor_m03[list(s2.columns)].equals(s2.reset_index(drop=True)), "kolom Modul 2 berubah"
assert set(sensor_m03.suhu_c_status_m03) <= STATUS_M03
tetap = sensor_m03.suhu_c_bersih_metode.eq("teramati")
assert np.allclose(sensor_m03.suhu_c_bersih[tetap].astype(float), sensor_m03.suhu_c[tetap].astype(float),
                   equal_nan=True), "nilai teramati berubah"
assert sensor_m03.suhu_c_bersih.dropna().lt(50).all(), "masih ada nilai mustahil pada suhu_c_bersih"
sensor_m03.to_parquet(OUT / "sensor_m03.parquet", index=False)
print(sensor_m03.shape, "->", "sensor_m03.parquet")
pd.crosstab(sensor_m03.suhu_c_status_m03, sensor_m03.id_gudang, margins=True)
```

*Log keputusan sensor (K-03-009 sampai K-03-012):*

```python
n_lon, n_mac, n_kom = int(lonjakan.sum()), int(macet.sum()), int(kenaikan.sum())
kj_mac = kejadian[kejadian.jenis.eq("sensor_macet")].iloc[0]
kj_kom = kejadian[kejadian.jenis.eq("kegagalan_kompresor")].iloc[0]
per_gudang = s[lonjakan].id_gudang.value_counts().sort_index()
ringkas_sensor = (sensor_m03.groupby("suhu_c_status_m03").agg(titik=("suhu_c_status_m03", "size"))
                  .assign(persen=lambda d: (100 * d.titik / len(sensor_m03)).round(2)))
B["sensor_status"] = ringkas_sensor

log_sensor = catat_keputusan([
    {"id": "K-03-009", "modul": 3, "sumber_kolom": "sensor_gudang.suhu_c",
     "temuan": f"parameter filter Hampel dan aturan pintu terbuka (Latihan 3.1): jendela {JENDELA_H}, k {K_H}",
     "bukti": (f"menangkap {int((tanda_p & lonjakan).sum())} dari {n_lon} lonjakan 85,0; menandai {int(tanda_p.sum())} titik "
               f"({int((tanda_p & jejak_pintu & ~lonjakan).sum())} di jejak pintu); {int((pintu.eq(1) & tanda_p & ~lonjakan).sum())} "
               f"dari {int(pintu.eq(1).sum())} titik pintu terbuka ditandai"),
     "tindakan": "Hampel hanya menandai kandidat; jejak pintu dan pencilan Hampel lain berstatus pencilan_sah, nilai tidak diubah",
     "alasan": "kenaikan +2,5 C saat pintu terbuka adalah peristiwa sah yang turun dalam tiga titik; pengecualian tidak berlaku bagi nilai mustahil",
     "baris_terdampak": 0, "id_anomali": ID_ANOMALI["lonjakan"]},
    {"id": "K-03-010", "modul": 3, "sumber_kolom": "sensor_gudang.suhu_c",
     "temuan": "lonjakan 85,0 C di gudang berpendingin",
     "bukti": (f"{n_lon} titik tepat 85,0 C ({', '.join(f'{g} {n}' for g, n in per_gudang.items())}); median Hampel "
               f"{angka(median_p[lonjakan].min(), 2)}-{angka(median_p[lonjakan].max(), 2)} C; {B['lonjakan_di_jejak_pintu']} "
               "jatuh di jejak pintu"),
     "tindakan": "suhu_c_bersih = median Hampel; suhu_c_bersih_metode=koreksi_median_hampel; suhu_c_status_m03=kesalahan_dikoreksi; suhu_c tetap 85,0",
     "alasan": "nilai mustahil secara fisik; median bergulir terpusat adalah penaksir yang tidak terpengaruh lonjakan tunggal",
     "baris_terdampak": n_lon, "id_anomali": ID_ANOMALI["lonjakan"]},
    {"id": "K-03-011", "modul": 3, "sumber_kolom": "sensor_gudang.suhu_c",
     "temuan": "sensor macet: nilai identik berturut-turut",
     "bukti": f"{kj_mac.id_gudang} {kj_mac.mulai:%Y-%m-%d %H.%M}-{kj_mac.selesai:%H.%M}: {kj_mac.bukti}",
     "tindakan": "suhu_c_bersih dikosongkan; suhu_c_bersih_metode=dikosongkan; suhu_c_status_m03=kesalahan_dikosongkan",
     "alasan": "suhu sebenarnya tidak diketahui; mengisi dengan nilai lain sama dengan menebak. Hampel tidak menandainya (konstan = median)",
     "baris_terdampak": n_mac, "id_anomali": ID_ANOMALI["macet"]},
    {"id": "K-03-012", "modul": 3, "sumber_kolom": "sensor_gudang.suhu_c",
     "temuan": "kegagalan kompresor: kenaikan suhu berkelanjutan",
     "bukti": f"{kj_kom.id_gudang} {kj_kom.mulai:%Y-%m-%d %H.%M}-{kj_kom.selesai:%H.%M}: {kj_kom.bukti}",
     "tindakan": "nilai dipertahankan; suhu_c_status_m03=anomali_nyata; tidak dihaluskan, tidak dikoreksi; dilaporkan ke operasional",
     "alasan": "kejadian nyata; penghalusan menurunkan puncak dan menunda alarm 20 menit (C.2)",
     "baris_terdampak": 0, "id_anomali": ID_ANOMALI["kompresor"]},
])
log_sensor[log_sensor.id.isin(["K-03-009", "K-03-010", "K-03-011", "K-03-012"])][["id", "sumber_kolom", "tindakan", "baris_terdampak"]]
```

**2. Dampak penanganan pada nilai barang bulanan.** Hitung nilai barang per bulan (WIB) untuk keempat
strategi Bagian D dan pembandingnya (`total_bayar − ongkir`). Simpan sebagai
`keluaran/dampak_penanganan_m03.csv` (satu baris per bulan, satu kolom per strategi) dan grafik
`keluaran/06_dampak_omzet.png`.

```python
# Kolom waktu pesanan: dicari otomatis; isi KOLOM_WAKTU sendiri bila tebakan salah
KOLOM_WAKTU = None
if KOLOM_WAKTU is None:
    calon = [c for c in t2.columns if re.search(r"(waktu|tanggal|tgl|date|time)", c, re.I)
             and not re.search(r"(kirim|terima|ulasan|bayar|update)", c, re.I)]
    KOLOM_WAKTU = next((c for c in calon if pd.to_datetime(t2[c], errors="coerce").notna().mean() > .99), None)
    if KOLOM_WAKTU is None:
        raise ValueError(f"kolom waktu pesanan tidak ditemukan di {list(t2.columns)}; isi KOLOM_WAKTU")
w = pd.to_datetime(t2[KOLOM_WAKTU], errors="coerce")
if w.dt.tz is not None:
    wib, cara = w.dt.tz_convert("Asia/Jakarta").dt.tz_localize(None), "dikonversi dari zona waktu pada data"
elif re.search("utc", KOLOM_WAKTU, re.I):
    wib, cara = w.dt.tz_localize("UTC").dt.tz_convert("Asia/Jakarta").dt.tz_localize(None), "UTC -> WIB"
else:
    wib, cara = w, "tanpa zona waktu, dianggap sudah WIB (periksa kamus data Modul 2)"
bulan = wib.dt.to_period("M")
print(f"kolom waktu: {KOLOM_WAKTU} ({cara}); tanpa tanggal valid: {int(bulan.isna().sum())} baris; "
      f"{bulan.min()} sampai {bulan.max()}")

# nilai barang per bulan untuk keempat strategi Bagian D (strategi dan nilai_barang dari D.1) dan pembandingnya
PEMBANDING = "pembanding_total_bayar_dikurangi_ongkir"
per_strategi = {}
for nama, (jj, hh, simpan) in strategi.items():
    nb = nilai_barang(jj, hh).astype(float)
    per_strategi[nama] = nb[simpan].groupby(bulan[simpan]).sum()
    # rekonsiliasi: jumlah semua bulan = nilai barang total strategi pada baris bertanggal valid
    assert np.isclose(per_strategi[nama].sum(), nb[simpan & bulan.notna()].sum()), f"{nama}: jumlah bulanan tidak cocok"
per_strategi[PEMBANDING] = pembanding.groupby(bulan).sum()
assert np.isclose(per_strategi[PEMBANDING].sum(), pembanding[bulan.notna()].sum())
dampak = pd.DataFrame(per_strategi).fillna(0).round(0)
dampak.index = dampak.index.astype(str)
dampak = dampak.rename_axis("bulan").reset_index()
dampak.to_csv(OUT / "dampak_penanganan_m03.csv", index=False)      # satuan: rupiah

NAMA_STRATEGI = list(strategi)
selisih = pd.DataFrame({"bulan": dampak.bulan,
                        **{n: (100 * (dampak[n] / dampak[PEMBANDING] - 1)).round(2) for n in NAMA_STRATEGI}})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.8))
for n in NAMA_STRATEGI:
    ax1.plot(dampak.bulan, dampak[n] / 1e9, marker="o", lw=1.1, label=n)
    ax2.plot(selisih.bulan, selisih[n], marker="o", lw=1.1, label=n)
ax1.plot(dampak.bulan, dampak[PEMBANDING] / 1e9, "k--", lw=1.4, label="pembanding (total_bayar - ongkir)")
ax2.axhline(0, color="k", lw=.8, ls="--")
ax1.set_ylabel("nilai barang (miliar Rp)"); ax1.set_title("Nilai barang per bulan (WIB)")
ax2.set_ylabel("selisih dari pembanding (%)"); ax2.set_title("Selisih relatif terhadap pembanding")
for ax in (ax1, ax2):
    ax.tick_params(axis="x", rotation=30)
ax1.legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "06_dampak_omzet.png", dpi=110, bbox_inches="tight")
plt.show()
dampak.assign(**{f"{n}_selisih_persen": selisih[n] for n in NAMA_STRATEGI}).round(2)
```

**3. Log keputusan** memuat sekurang-kurangnya **sepuluh** baris `K-03-*`: delapan dari Bagian D ditambah
keputusan Anda untuk sensor (`sumber_kolom` memuat `sensor_gudang`) dan untuk Latihan 3.2. Setiap temuan
register Modul 1 Anda yang `modul_penanganan`-nya 3 dirujuk oleh sekurang-kurangnya satu baris log.

```python
log_semua = pd.read_csv(LOG, dtype=str, keep_default_na=False)
k03 = log_semua[log_semua.id.str.startswith("K-03")]
assert len(k03) >= 10, f"log K-03 baru {len(k03)} baris; minimal 10"
assert k03.sumber_kolom.str.contains("sensor_gudang").any(), "tidak ada keputusan sensor"
assert k03.temuan.str.contains("Latihan 3.2").any(), "tidak ada keputusan Latihan 3.2"
assert set(k03.columns) == set(KOLOM_LOG)
assert (MODUL / "log-keputusan.csv").exists(), "salinan log untuk ZIP belum ada"
print(f"{len(k03)} baris K-03: {', '.join(k03.id)}")

# Pemeriksaan cepat terhadap register Modul 1 (bila berkasnya ditemukan di workspace)
rujukan = sorted({a.strip() for x in k03.id_anomali for a in x.split(";") if a.strip().startswith("A-")})
print("id temuan yang dirujuk log K-03:", ", ".join(rujukan))
register = None
for p in [*ROOT.glob("*.csv"), *ROOT.glob("dw-module-01/**/*.csv")]:
    try:
        kolom = pd.read_csv(p, nrows=0).columns
    except Exception:
        continue
    if "modul_penanganan" in kolom:
        register = pd.read_csv(p, dtype=str, keep_default_na=False)
        print("register Modul 1:", p.relative_to(ROOT))
        break
if register is not None:
    kol_id = next((c for c in register.columns if c.lower() in {"id", "id_anomali", "id_temuan"}), register.columns[0])
    modul3 = register[register.modul_penanganan.astype(str).str.strip().eq("3")]
    belum = sorted(set(modul3[kol_id]) - set(rujukan))
    print("temuan modul_penanganan = 3:", len(modul3), "| belum dirujuk log:", belum or "tidak ada")
else:
    print("register Modul 1 tidak ditemukan otomatis; cocokkan sendiri id_anomali pada log dengan register Anda")
k03[["id", "sumber_kolom", "baris_terdampak", "id_anomali"]]
```

**4. `laporan.md`** mengikuti `template-laporan.md`: tabel deteksi, bukti kesalahan dan pencilan sah untuk
setiap kolom, tabel kejadian sensor, ringkasan `sensor_m03.parquet` per status, perbandingan strategi dan
dampak bulanan, hasil Latihan 3.1–3.2, rekonsiliasi jumlah baris transaksi **dan** sensor, serta
keterbatasan yang menyebut sekurang-kurangnya satu risiko sisa beserta cara memperkirakan besarnya.
Setiap angka harus dapat dilacak ke sel notebook.


Sel berikut menyusun kerangka `laporan.md` dari objek hasil di atas; sesuaikan judul bagian dengan `template-laporan.md` (sel mencetak judul templat bila berkasnya ditemukan).

```python
# Kerangka laporan.md: seluruh tabel dan angka diambil dari objek hasil sel di atas (sumber disebut pada tiap bagian)
TEMPLATE = next(iter(ROOT.rglob("template-laporan.md")), None)
if TEMPLATE:
    print("Judul bagian pada", TEMPLATE.relative_to(ROOT), "-- samakan susunan laporan.md dengan ini:")
    print(*[ln for ln in TEMPLATE.read_text(encoding="utf-8").splitlines() if ln.startswith("#")], sep="\n")
else:
    print("template-laporan.md tidak ditemukan otomatis; sesuaikan judul bagian di bawah dengan templat dari dosen")

# rekonsiliasi jumlah baris
rek = pd.DataFrame([
    {"tabel": "transaksi", "baris_masuk (Modul 2)": len(t2), "baris_keluar (Modul 3)": len(hasil),
     "baris_dihapus": len(t2) - len(hasil), "jumlah_per_status": int(hasil.jumlah_status_m03.value_counts().sum())},
    {"tabel": "sensor", "baris_masuk (Modul 2)": len(s2), "baris_keluar (Modul 3)": len(sensor_m03),
     "baris_dihapus": len(s2) - len(sensor_m03), "jumlah_per_status": int(sensor_m03.suhu_c_status_m03.value_counts().sum())},
])
assert (rek["baris_dihapus"] == 0).all() and (rek["baris_keluar (Modul 3)"] == rek["jumlah_per_status"]).all()

# risiko sisa yang dapat diperkirakan: harga x100 tersembunyi pada kode produk tanpa katalog
p_x100 = B["n_x100"] / (len(t2) - B["n_tanpa_katalog"])
harap_x100 = p_x100 * B["n_tanpa_katalog"]
status_jumlah = hasil.jumlah_status_m03.value_counts().rename_axis("status").reset_index(name="baris")
status_harga = hasil.harga_satuan_status_m03.value_counts().rename_axis("status").reset_index(name="baris")
tabel_kejadian = kejadian[kejadian.jenis.ne("lonjakan")][["id_gudang", "mulai", "selesai", "jenis", "n_titik", "bukti"]]
ringkas_lonjakan = kejadian[kejadian.jenis.eq("lonjakan")].groupby("id_gudang").n_titik.sum().rename_axis("id_gudang").reset_index(name="titik_lonjakan")
pb = B["perbandingan"].reset_index()

laporan = f"""# Laporan Modul 3 — Penanganan Noise dan Outlier

Praktikum Data Wrangling (SD25-20007) · Program Studi Sains Data ITERA. Setiap angka berasal dari objek atau berkas
keluaran yang disebut pada akhir bagiannya; seluruh sel dijalankan ulang dari *Restart Kernel and Run All Cells*.

## 1. Tabel deteksi (Bagian A.2–A.3)

{md_tabel(deteksi_pencilan)}

Sumber: `keluaran/deteksi_pencilan_m03.csv`. z-score menangkap {B['z_tangkap']} dari {B['n_x100']} harga ×100 dan
meloloskan {B['z_lolos']} (masking: sd {rp(B['sd_dengan'])} vs {rp(B['sd_tanpa'])} tanpa kesalahan).

## 2. Bukti kesalahan dan pencilan sah per kolom (Bagian B)

- **`harga_satuan`** — kesalahan: {B['n_x100']} harga tepat 100 × harga katalog; identitas `total_bayar` cocok bila dibagi
  100 pada {B['x100_cocok_bagi']} baris dan tidak pernah cocok apa adanya ({B['x100_cocok_asli']}). Koreksi ÷100.
  Harga sah mahal (kain tenun, kerajinan) dipertahankan. Sel B.1.
- **`jumlah`** — kesalahan: {B['neg']} jumlah negatif, bukan retur (total_bayar negatif: {B['neg_total_negatif']}; cocok dengan
  |jumlah|: {B['neg_cocok_mutlak']}; dikembalikan {angka(B['p_retur_neg'])}% vs {angka(B['p_retur_lain'])}%, Fisher
  p = {angka(B['p_fisher'], 2)}). {B['neg_koreksi']} dikoreksi tanda, {B['neg_kosong']} dikosongkan. Pencilan sah:
  {B['grosir']} pembelian {B['grosir_min']}–{B['grosir_max']} unit, seluruhnya {' dan '.join(B['grosir_kategori'])}. Sel B.2–B.3.
- **`lama_kirim_hari`** — pencilan sah: {B['lama_tertandai']} baris di atas pagar IQR {angka(B['lama_pagar'], 1)} hari,
  {B['lama_18']} di antaranya ≥ 18 hari ({B['lama_18_pos']} Pos Indonesia). Sel B.4.
- **`ongkir`, `total_bayar`** — tidak diubah; `total_bayar` menjadi saksi identitas. Perbandingan empat detektor pada
  Latihan 3.2 (bagian 6).

Status hasil `transaksi_m03.parquet` (sel D.2):

{md_tabel(status_jumlah.assign(kolom="jumlah"))}

{md_tabel(status_harga.assign(kolom="harga_satuan"))}

## 3. Kejadian sensor (Bagian C.4)

{md_tabel(tabel_kejadian)}

Ditambah lonjakan 85,0 °C menurut gudang:

{md_tabel(ringkas_lonjakan)}

Sumber: `keluaran/kejadian_sensor_m03.csv`.

## 4. Ringkasan `sensor_m03.parquet` per status (Tugas 1)

{md_tabel(ringkas_sensor.reset_index())}

Parameter Hampel: jendela {JENDELA_H}, k {K_H} (Latihan 3.1). Urutan prioritas status: tidak ada nilai → nilai mustahil
→ sensor macet → kenaikan kompresor → tanda Hampel lain → wajar. Kejadian nyata tidak dihaluskan dan tidak dikoreksi;
pintu terbuka bukan kesalahan.

## 5. Perbandingan strategi dan dampak bulanan (Bagian D.1, Tugas 2)

{md_tabel(pb[['strategi', 'baris_tersisa', 'rata_harga', 'median_harga', 'sd_harga', 'nilai_barang_miliar', 'selisih_dari_total_bayar_persen']])}

Pembanding `total_bayar − ongkir` = Rp{angka(B['pembanding_miliar'], 3)} miliar. Dampak per bulan (rupiah, WIB):

{md_tabel(dampak)}

Sumber: `keluaran/perbandingan_strategi_m03.csv`, `keluaran/dampak_penanganan_m03.csv`, grafik `06_dampak_omzet.png`.

## 6. Latihan 3.1–3.2

- **Latihan 3.1** (`latihan-hampel.md`): pasangan terpilih jendela {JENDELA_H}, k {K_H}; menangkap
  {int(sel_p.lonjakan_85)} dari {N_LONJAKAN} lonjakan dengan {int(sel_p.ditandai)} titik ditandai; kejadian nyata ditandai
  {int(sel_p.kejadian_nyata)} titik sehingga perlindungannya bertumpu pada tabel kejadian.
- **Latihan 3.2** (`latihan-multimetode.md`): tanda ongkir z/IQR/robust z/IsolationForest =
  {'/'.join(str(int(rr.loc[('ongkir', m), 'ditandai'])) for m in METODE)}; total_bayar =
  {'/'.join(str(int(rr.loc[('total_bayar', m), 'ditandai'])) for m in METODE)}; grosir sah ikut ditandai pada ongkir
  {'/'.join(str(int(rr.loc[('ongkir', m), 'grosir_ditandai'])) for m in METODE)} dari {int(besar.sum())}.

## 7. Rekonsiliasi jumlah baris

{md_tabel(rek)}

Tidak ada baris yang dihapus pada transaksi maupun sensor; seluruh baris memiliki tepat satu status.

## 8. Keterbatasan

- **Harga ×100 pada kode produk tanpa katalog.** {B['n_tanpa_katalog']} baris tidak dapat diperiksa lewat rasio katalog;
  identitas `total_bayar` cocok pada {B['yatim_cocok']} baris. Besar risiko sisa dapat diperkirakan dari proporsi
  pada baris berkatalog: {B['n_x100']} dari {len(t2) - B['n_tanpa_katalog']} ({angka(100 * p_x100, 2)}%), sehingga
  diharapkan sekitar **{angka(harap_x100, 1)} baris** tersembunyi di antara {B['n_tanpa_katalog']} baris itu.
- **Batas kejadian kompresor** ditentukan dari data (median bergulir terhadap baseline 24 jam), sehingga dapat
  bergeser beberapa titik bila ambang diubah; perkirakan dengan mengulang sel C.4 pada ambang 1,5 °C dan 2,5 °C.
- **Pencilan Hampel lain berstatus `pencilan_sah` tanpa bukti independen**; besarnya
  ({int((sensor_m03.suhu_c_status_m03 == 'pencilan_sah').sum())} titik) dapat dibandingkan dengan log perawatan gudang bila ada.
- **Urutan koreksi sebelum imputasi** ({B['ongkir_kini_turun']} ongkir isian median Modul 2 kini dapat diturunkan, {len(B['ongkir_beda'])}
  berbeda dari median) belum diterapkan ulang; dicatat untuk pipeline akhir.
"""
(MODUL / "laporan.md").write_text(laporan, encoding="utf-8")
print("ditulis:", (MODUL / "laporan.md").relative_to(ROOT), f"({len(laporan.splitlines())} baris)")
```

**Sebelum mengunggah:** *Restart Kernel and Run All Cells*, jalankan **Checkpoint** di Workbench sampai tidak
ada butir GAGAL, lalu kompres direktori `dw-module-03/` menjadi `dw-module-03.zip`.
