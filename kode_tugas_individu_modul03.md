# Kode Tugas Individu — Modul 3 (Penanganan Noise dan Outlier)

Letakkan tiap blok kode di **sel baru** setelah sel Latihan 3.2 di `03_outlier.ipynb`, sesuai urutan.
Blok ini memakai variabel dari Bagian A–D dan Latihan 3.1–3.2 (`s`, `s2`, `t2`, `num`, `lonjakan`, `macet`,
`kenaikan`, `jejak_pintu`, `hampel_per_gudang`, `strategi`, `perbandingan`, `tanda_semua`, `ringkas`,
`kesepakatan`, `catat_keputusan`, `B`, `OUT`, dll.), jadi **jalankan notebook dari atas** (Restart Kernel and Run All) lebih dulu.

---

## Nomor 1 — Sensor: `sensor_m03.parquet`

### 1a. Status, nilai bersih, dan metode per titik

Urutan prioritas bila satu titik memenuhi lebih dari satu aturan (yang paling atas menang):

| Prioritas | Kondisi | `suhu_c_status_m03` | `suhu_c_bersih` | `suhu_c_bersih_metode` |
|---|---|---|---|---|
| 1 | `suhu_c` kosong (celah data Modul 2) | `tidak_berlaku` | kosong | `tidak_ada_data` |
| 2 | nilai 85,0 (mustahil, termasuk yang jatuh di jejak pintu) | `kesalahan_dikoreksi` | median Hampel | `koreksi_median_hampel` |
| 3 | sensor macet (deretan identik ≥ 6 titik) | `kesalahan_dikosongkan` | kosong | `dikosongkan_sensor_macet` |
| 4 | kegagalan kompresor (kenaikan berkelanjutan) | `anomali_nyata` | = `suhu_c` | `teramati` |
| 5 | ditandai Hampel dan berada di jejak pintu terbuka | `pencilan_sah` | = `suhu_c` | `teramati` |
| 6 | ditandai Hampel, di luar kondisi di atas | `pencilan_sah` | = `suhu_c` | `teramati` |
| 7 | lainnya | `wajar` | = `suhu_c` | `teramati` |

Alasan urutan: nilai yang mustahil (85,0) harus dikoreksi walaupun jatuh di jejak pintu; kejadian nyata
(kompresor) dilindungi dari koreksi dan penghalusan; pintu terbuka adalah kenaikan sah, bukan kesalahan.

```python
# ============ TUGAS 1a: status, suhu_c_bersih, suhu_c_bersih_metode ============
# Parameter Hampel dari Latihan 3.1 (sesuaikan bila pilihan Anda berbeda)
JENDELA_H, K_H = 13, 5
th_h, mh_h = hampel_per_gudang(JENDELA_H, K_H)   # tanda & median Hampel, indeks sejajar dengan s

STATUS_SUHU = ["wajar", "pencilan_sah", "kesalahan_dikoreksi", "kesalahan_dikosongkan",
               "anomali_nyata", "tidak_berlaku"]

# Pemeriksaan awal: semua nilai mustahil (>= 50) memang hanya titik 85,0
assert s.suhu_c.ge(50).sum() == lonjakan.sum(), "ada nilai >= 50 yang bukan 85,0"

kosong = s.suhu_c.isna()
kondisi = [kosong,
           lonjakan,
           macet,
           kenaikan,
           th_h & jejak_pintu,
           th_h]
status_suhu = np.select(kondisi,
                        ["tidak_berlaku", "kesalahan_dikoreksi", "kesalahan_dikosongkan",
                         "anomali_nyata", "pencilan_sah", "pencilan_sah"],
                        default="wajar")

metode_suhu = np.select([kosong, lonjakan, macet],
                        ["tidak_ada_data", "koreksi_median_hampel", "dikosongkan_sensor_macet"],
                        default="teramati")

suhu_bersih = s.suhu_c.astype(float).copy()
suhu_bersih[lonjakan] = mh_h[lonjakan].round(2)   # koreksi: median Hampel di sekitar titik
suhu_bersih[macet] = np.nan                       # suhu sebenarnya tidak diketahui
# kosong (celah data) sudah NaN; kejadian nyata & pintu dibiarkan apa adanya

tambahan = s[["id_gudang", "waktu"]].copy()
tambahan["suhu_c_status_m03"] = status_suhu
tambahan["suhu_c_bersih"] = suhu_bersih.to_numpy()
tambahan["suhu_c_bersih_metode"] = metode_suhu

pd.crosstab(tambahan.suhu_c_status_m03, tambahan.suhu_c_bersih_metode, margins=True)
```

### 1b. Gabungkan ke grid Modul 2 apa adanya, validasi, dan simpan

`s` sudah diurutkan ulang di C.1, jadi hasil digabung kembali ke `s2` lewat kunci `(id_gudang, waktu)`
agar urutan dan seluruh kolom Modul 2 tetap utuh.

```python
# ============ TUGAS 1b: gabung ke s2, validasi, simpan ============
kunci = ["id_gudang", "waktu"]
assert not s2.duplicated(kunci).any(), "kunci (id_gudang, waktu) tidak unik"

sensor_m03 = s2.merge(tambahan, on=kunci, how="left", validate="one_to_one")

# --- validasi ---
assert len(sensor_m03) == len(s2), "jumlah baris berubah"
assert list(sensor_m03.columns[:len(s2.columns)]) == list(s2.columns), "kolom Modul 2 berubah urutan"
for k in s2.columns:                                   # seluruh kolom Modul 2 tidak berubah
    assert sensor_m03[k].equals(s2[k]) or (
        sensor_m03[k].isna().equals(s2[k].isna())
        and np.allclose(pd.to_numeric(sensor_m03[k], errors="coerce").fillna(0),
                        pd.to_numeric(s2[k], errors="coerce").fillna(0))
    ) or sensor_m03[k].astype(str).equals(s2[k].astype(str)), f"kolom {k} berubah"
assert set(sensor_m03.suhu_c_status_m03) <= set(STATUS_SUHU), "status di luar nilai baku"
assert sensor_m03.suhu_c_status_m03.notna().all()

# nilai teramati harus sama persis dengan suhu_c; hanya metode non-teramati yang boleh beda
tetap = sensor_m03.suhu_c_bersih_metode.eq("teramati")
assert np.allclose(sensor_m03.loc[tetap, "suhu_c_bersih"].astype(float),
                   sensor_m03.loc[tetap, "suhu_c"].astype(float), equal_nan=True), \
    "nilai teramati berubah"
# kejadian nyata tidak dihaluskan dan tidak dikoreksi
nyata = sensor_m03.suhu_c_status_m03.eq("anomali_nyata")
assert sensor_m03.loc[nyata, "suhu_c_bersih_metode"].eq("teramati").all()
# tidak ada nilai mustahil yang tersisa pada suhu_c_bersih
assert sensor_m03.suhu_c_bersih.dropna().lt(50).all(), "masih ada nilai mustahil"

sensor_m03.to_parquet(OUT / "sensor_m03.parquet", index=False)
print(sensor_m03.shape, "->", "sensor_m03.parquet")

ringkasan_sensor = (sensor_m03.groupby("suhu_c_status_m03")
                    .agg(titik=("suhu_c", "size"),
                         suhu_c_median=("suhu_c", "median"),
                         bersih_kosong=("suhu_c_bersih", lambda x: int(x.isna().sum())))
                    .reindex(STATUS_SUHU).fillna(0))
ringkasan_sensor
```

> Perkiraan hasil bila parameter Hampel (13, 5): `kesalahan_dikoreksi` = 42, `kesalahan_dikosongkan` = 36,
> `anomali_nyata` ≈ 51, `tidak_berlaku` ≥ 36 (celah data GDG-MDN, ditambah baris kosong lain bila ada).
> Batas kejadian kompresor boleh berbeda beberapa titik antarmahasiswa.

### 1c. Catat keputusan sensor ke log

`id_anomali` untuk sensor: isi dengan ID dari **register Modul 1 Anda** (di sini diberi placeholder `"-"`).

```python
# ============ TUGAS 1c: log keputusan sensor (K-03-009 s.d. K-03-012, K-03-014) ============
ID_LONJAKAN = "-"   # ganti dengan ID register Modul 1 untuk lonjakan 85,0 (mis. "A-xx")
ID_MACET    = "-"   # ID register untuk sensor macet
ID_KOMP     = "-"   # ID register untuk kegagalan kompresor / kenaikan berkelanjutan
ID_CELAH    = "-"   # ID register untuk celah data sensor

st = sensor_m03.suhu_c_status_m03.value_counts()
n_lonj   = int(st.get("kesalahan_dikoreksi", 0))
n_macet  = int(st.get("kesalahan_dikosongkan", 0))
n_nyata  = int(st.get("anomali_nyata", 0))
n_celah  = int(st.get("tidak_berlaku", 0))
n_pintu  = int(((tambahan.suhu_c_status_m03 == "pencilan_sah").to_numpy() & jejak_pintu.to_numpy()).sum())
n_pencilan = int(st.get("pencilan_sah", 0))

log = catat_keputusan([
    {"id": "K-03-009", "modul": 3, "sumber_kolom": "sensor_gudang.suhu_c",
     "temuan": "lonjakan 85,0 °C: mustahil di gudang berpendingin",
     "bukti": (f"{n_lonj} titik bernilai 85,0; median Hampel di sekitarnya "
               f"{angka(float(mh_h[lonjakan].min()), 2)}–{angka(float(mh_h[lonjakan].max()), 2)} °C; "
               f"{B['lonjakan_di_jejak_pintu']} di antaranya jatuh di jejak pintu terbuka; "
               f"rata-rata/EWMA bergulir mengubahnya menjadi {int(B['alarm'].loc['rata_bergulir','akibat_lonjakan'])}/"
               f"{int(B['alarm'].loc['ewma','akibat_lonjakan'])} alarm palsu"),
     "tindakan": (f"suhu_c tetap; suhu_c_bersih = median Hampel (jendela {JENDELA_H}, k={K_H}); "
                  "suhu_c_bersih_metode=koreksi_median_hampel; suhu_c_status_m03=kesalahan_dikoreksi"),
     "alasan": "nilai benar dapat diperkirakan dari tetangga; median tidak terpengaruh lonjakan, rata-rata dan EWMA terpengaruh",
     "baris_terdampak": n_lonj, "id_anomali": ID_LONJAKAN},

    {"id": "K-03-010", "modul": 3, "sumber_kolom": "sensor_gudang.suhu_c",
     "temuan": "sensor macet: nilai identik berturut-turut",
     "bukti": (f"{n_macet} titik identik pada GDG-BDL; sd bergulir 1 jam ≈ {B['sd_macet']:.1e} "
               "(bukan 0 karena galat floating point; sd == 0 menemukan nol titik)"),
     "tindakan": "suhu_c_bersih dikosongkan; suhu_c_bersih_metode=dikosongkan_sensor_macet; suhu_c_status_m03=kesalahan_dikosongkan",
     "alasan": "suhu sebenarnya tidak diketahui; mengisi dengan tebakan sama dengan mengarang data",
     "baris_terdampak": n_macet, "id_anomali": ID_MACET},

    {"id": "K-03-011", "modul": 3, "sumber_kolom": "sensor_gudang.suhu_c",
     "temuan": "kegagalan kompresor: kenaikan suhu berkelanjutan pada GDG-PLM",
     "bukti": (f"puncak {angka(B['puncak'], 2)} °C pada {B['hari_puncak'][1]:%d %b %Y}; {n_nyata} titik; "
               "median 1 jam > baseline 24 jam + 2 °C; penghalus menurunkan puncak ke 12,8–12,9 °C dan menunda alarm 20 menit"),
     "tindakan": "tidak dihaluskan, tidak dikoreksi; suhu_c_status_m03=anomali_nyata; dilaporkan ke operasional",
     "alasan": "kejadian nyata harus terlihat apa adanya; k Hampel saja tidak dapat melindunginya (MAD runtuh pada kenaikan monoton), jadi dilindungi oleh tabel kejadian",
     "baris_terdampak": n_nyata, "id_anomali": ID_KOMP},

    {"id": "K-03-012", "modul": 3, "sumber_kolom": "sensor_gudang.suhu_c;sensor_gudang.pintu_terbuka",
     "temuan": "pintu terbuka dan noise ringan ditandai Hampel padahal sah",
     "bukti": (f"Hampel ({JENDELA_H}, {K_H}) menandai {int(th_h.sum())} titik: {int((th_h & lonjakan).sum())} lonjakan, "
               f"{n_pintu} di jejak pintu (titik pintu + dua titik sesudahnya), sisanya noise; "
               f"hanya {n_pencilan} titik berstatus pencilan_sah"),
     "tindakan": "nilai dipertahankan; suhu_c_status_m03=pencilan_sah; tidak dikoreksi",
     "alasan": "kenaikan ±2,5 °C saat pintu terbuka adalah perilaku sah; pengecualian memakai kolom pintu_terbuka, tetapi tidak membebaskan nilai mustahil (85,0 di jejak pintu tetap dikoreksi)",
     "baris_terdampak": 0, "id_anomali": "-"},

    {"id": "K-03-014", "modul": 3, "sumber_kolom": "sensor_gudang.suhu_c",
     "temuan": "celah data sensor dari Modul 2",
     "bukti": f"{n_celah} titik suhu_c kosong (baris tidak terkirim; tidak diisi Modul 2)",
     "tindakan": "tetap kosong; suhu_c_status_m03=tidak_berlaku; suhu_c_bersih_metode=tidak_ada_data",
     "alasan": "tidak ada bukti untuk memperkirakan nilai; mengisi akan menyamarkan kegagalan pengiriman",
     "baris_terdampak": 0, "id_anomali": ID_CELAH},
])
log[log.id.isin(["K-03-009", "K-03-010", "K-03-011", "K-03-012", "K-03-014"])][
    ["id", "sumber_kolom", "tindakan", "baris_terdampak"]]
```

---

## Nomor 2 — Dampak penanganan pada nilai barang bulanan

Nilai barang per baris = `jumlah × harga × (1 − diskon_efektif/100)` untuk tiap strategi D.1; pada strategi
"hapus di luar 3 sd" baris yang dibuang bernilai 0. Pembanding = `total_bayar − ongkir`.
Bulan ditentukan menurut **WIB (Asia/Jakarta)**.

> **Sesuaikan** `KOLOM_TGL` bila nama kolom tanggal transaksi Anda berbeda. Kode mencoba menebaknya.

```python
# ============ TUGAS 2: nilai barang bulanan per strategi ============
# --- 1) kolom tanggal transaksi ---
kandidat = [c for c in t2.columns if any(w in c.lower() for w in ["tanggal", "waktu", "tgl"])
            and "kirim" not in c.lower() and not c.lower().endswith("_metode")]
print("kandidat kolom tanggal:", kandidat)
KOLOM_TGL = kandidat[0]          # <-- ubah manual bila tebakan salah
print("dipakai:", KOLOM_TGL, "| contoh:", t2[KOLOM_TGL].iloc[0])

# --- 2) ke WIB ---
ts = pd.to_datetime(t2[KOLOM_TGL], errors="coerce")
if ts.dt.tz is None:
    # Bila Modul 2 menyimpan waktu naive dalam UTC, ganti menjadi:
    #   ts = ts.dt.tz_localize("UTC").dt.tz_convert("Asia/Jakarta")
    ts = ts.dt.tz_localize("Asia/Jakarta")     # naive dianggap sudah WIB
else:
    ts = ts.dt.tz_convert("Asia/Jakarta")
bulan = ts.dt.strftime("%Y-%m")
print("tanggal tidak terbaca:", int(ts.isna().sum()), "| rentang:", bulan.min(), "s.d.", bulan.max())

# --- 3) nilai barang per baris untuk tiap strategi ---
def ke_float(x) -> pd.Series:
    """Ubah Series (termasuk Int64/Float64 bertipe nullable) ke float64 dengan NA -> NaN."""
    return pd.Series(pd.array(x, dtype="Float64").to_numpy(dtype=float, na_value=np.nan), index=x.index)

nilai_per_baris = {}
for nama, (jj, hh, simpan) in strategi.items():
    nb = nilai_barang(ke_float(jj), ke_float(hh))
    nilai_per_baris[nama] = nb.where(simpan, 0.0)          # baris dihapus bernilai 0
nilai_per_baris["pembanding_total_bayar_minus_ongkir"] = pembanding

tabel_baris = pd.DataFrame(nilai_per_baris)
dampak = (tabel_baris.groupby(bulan, dropna=True).sum(min_count=1)
          .rename_axis("bulan_wib").reset_index())

# --- 4) rekonsiliasi dengan tabel D.1 (total keseluruhan harus sama) ---
total_bulanan = dampak.drop(columns="bulan_wib").sum() / 1e9
for nama in strategi:
    ref = float(B["perbandingan"].loc[nama, "nilai_barang_miliar"])
    assert abs(total_bulanan[nama] - ref) < 0.01, f"{nama}: total bulanan {total_bulanan[nama]:.3f} != D.1 {ref:.3f}"
print("rekonsiliasi dengan D.1 lolos (miliar Rp):")
print(total_bulanan.round(3).to_string())

dampak.round(0).to_csv(OUT / "dampak_penanganan_m03.csv", index=False)
dampak.assign(**{c: (dampak[c] / 1e6).round(1) for c in dampak.columns[1:]}).head(12)  # tampilan juta Rp
```

```python
# ============ TUGAS 2: grafik dampak bulanan ============
pemb = "pembanding_total_bayar_minus_ongkir"
warna = {"biarkan": "#b2182b", "hapus_di_luar_3sd": "#e07b39",
         "winsorize_p1_p99": "#8c6d31", "koreksi_dan_bendera": "#2166ac"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
x = np.arange(len(dampak))
ax1.plot(x, dampak[pemb] / 1e6, color="k", lw=2, ls="--", label="pembanding (total_bayar − ongkir)")
for nama, c in warna.items():
    ax1.plot(x, dampak[nama] / 1e6, marker="o", ms=3, lw=1.2, color=c, label=nama)
ax1.set_xticks(x); ax1.set_xticklabels(dampak.bulan_wib, rotation=45, ha="right")
ax1.set_ylabel("nilai barang (juta Rp)"); ax1.set_title("Nilai barang bulanan menurut strategi")
ax1.legend(frameon=False, fontsize=8)

for nama, c in warna.items():
    sel = 100 * (dampak[nama] / dampak[pemb] - 1)
    ax2.plot(x, sel, marker="o", ms=3, lw=1.2, color=c, label=nama)
ax2.axhline(0, color="gray", lw=.8)
ax2.set_xticks(x); ax2.set_xticklabels(dampak.bulan_wib, rotation=45, ha="right")
ax2.set_ylabel("selisih terhadap pembanding (%)"); ax2.set_title("Selisih bulanan terhadap pembanding")
ax2.set_yscale("symlog", linthresh=10)   # 'biarkan' jauh lebih besar dari strategi lain
ax2.legend(frameon=False, fontsize=8)

fig.tight_layout()
fig.savefig(OUT / "06_dampak_omzet.png", dpi=120, bbox_inches="tight")
plt.show()

# ringkasan untuk laporan: bulan dengan selisih terbesar per strategi
for nama in strategi:
    sel = 100 * (dampak[nama] / dampak[pemb] - 1)
    i = sel.abs().idxmax()
    print(f"{nama:22s} selisih bulanan terbesar: {dampak.bulan_wib[i]} ({sel[i]:+.2f}%)")
```

---

## Nomor 3 — Log keputusan (≥ 10 baris `K-03-*`)

Sudah ada K-03-001…008 (Bagian D) + K-03-009…012 dan K-03-014 (sensor, di Nomor 1c). Tinggal **K-03-013 (Latihan 3.2)**
lalu pemeriksaan jumlah baris dan rujukan ke register Modul 1.

```python
# ============ TUGAS 3a: log Latihan 3.2 (K-03-013) ============
r = ringkas   # dari Latihan 3.2
jac = kesepakatan.set_index(["metode_a", "metode_b"]).jaccard
log = catat_keputusan([
    {"id": "K-03-013", "modul": 3, "sumber_kolom": "transaksi.ongkir;transaksi.total_bayar",
     "temuan": "empat detektor per provinsi tidak sepakat pada ongkir dan total_bayar; robust z ongkir tidak terdefinisi",
     "bukti": (f"tanda: z ongkir {int(r.loc['z_ongkir','tanda'])}, IQR ongkir {int(r.loc['iqr_ongkir','tanda'])}, "
               f"robust z ongkir {int(r.loc['robust_z_ongkir','tanda'])} (MAD = 0 di {len(mad_nol['ongkir'])} dari "
               f"{prov_bersih.nunique()} provinsi; ±70% ongkir sama dengan median provinsinya), "
               f"IsolationForest {int(r.loc['isolation_forest','tanda'])}; "
               f"Jaccard IQR ongkir vs IsolationForest {angka(jac[('iqr_ongkir','isolation_forest')], 2)}, "
               f"IQR vs robust z total_bayar {angka(jac[('iqr_total_bayar','robust_z_total_bayar')], 2)}; "
               f"pembelian grosir ikut ditandai (dari {int(besar.sum())}): z ongkir {int(r.loc['z_ongkir','grosir_ditandai'])}, "
               f"IQR ongkir {int(r.loc['iqr_ongkir','grosir_ditandai'])}, IQR total {int(r.loc['iqr_total_bayar','grosir_ditandai'])}, "
               f"robust z total {int(r.loc['robust_z_total_bayar','grosir_ditandai'])}, "
               f"IsolationForest {int(r.loc['isolation_forest','grosir_ditandai'])}"),
     "tindakan": "tidak ada nilai ongkir/total_bayar diubah; pembelian grosir tetap pencilan_sah dan tidak dihapus; robust z tidak dipakai untuk ongkir",
     "alasan": "ongkir berupa tarif diskret sehingga MAD = 0; IQR dan IsolationForest menandai ribuan nilai sah (ongkir paket berat, grosir); satu ambang global atau satu metode tidak cukup, perlu konteks (berat, provinsi, jumlah)",
     "baris_terdampak": 0, "id_anomali": "A-17"},
])

k03 = log[log.id.str.startswith("K-03")]
print("baris K-03-*:", len(k03), "(minimum 10)")
print("baris yang memuat sensor_gudang:", int(k03.sumber_kolom.str.contains("sensor_gudang").sum()))
assert len(k03) >= 10, "log K-03-* kurang dari 10 baris"
assert k03.sumber_kolom.str.contains("sensor_gudang").any(), "belum ada keputusan sensor"
k03[["id", "sumber_kolom", "baris_terdampak", "id_anomali"]]
```

```python
# ============ TUGAS 3b: setiap temuan register Modul 1 (modul_penanganan = 3) dirujuk log ============
# Cari file register Modul 1 Anda; ubah REGISTER bila tidak ketemu.
cari = sorted(ROOT.glob("**/*register*.csv"))
print("kandidat register:", [str(p.relative_to(ROOT)) for p in cari])
REGISTER = cari[0] if cari else None          # <-- ubah manual bila perlu, mis. ROOT/"dw-module-01"/"register.csv"

if REGISTER is None:
    print("File register tidak ditemukan: periksa manual tabel register Modul 1 terhadap kolom id_anomali di log.")
else:
    reg = pd.read_csv(REGISTER, dtype=str, keep_default_na=False)
    kol_mod = next(c for c in reg.columns if "modul_penanganan" in c.lower())
    kol_id = next(c for c in reg.columns if c.lower().startswith(("id", "kode")))
    wajib = reg.loc[reg[kol_mod].str.contains(r"\b3\b", regex=True), kol_id].tolist()

    dirujuk = set()
    for sel_id in k03.id_anomali:
        dirujuk |= {x.strip() for x in sel_id.replace(",", ";").split(";")}
    belum = [a for a in wajib if a not in dirujuk]

    print("temuan register modul_penanganan = 3:", wajib)
    print("belum dirujuk oleh log K-03-*       :", belum if belum else "tidak ada — lengkap")
    # Bila ada yang belum, tambahkan baris K-03-0xx untuk temuan itu (mis. K-03-015) lewat catat_keputusan([...]).
```

---

## Pemeriksaan akhir sebelum mengumpulkan (opsional, membantu laporan)

```python
# Rekonsiliasi jumlah baris (untuk bagian rekonsiliasi di laporan.md)
rekon = pd.DataFrame({
    "tabel": ["transaksi", "sensor"],
    "baris_modul2": [len(t2), len(s2)],
    "baris_modul3": [len(hasil), len(sensor_m03)],
})
rekon["selisih"] = rekon.baris_modul3 - rekon.baris_modul2
assert (rekon.selisih == 0).all(), "ada baris yang hilang/bertambah"
print(rekon.to_string(index=False))

berkas = ["deteksi_pencilan_m03.csv", "transaksi_m03.parquet", "kejadian_sensor_m03.csv",
          "perbandingan_strategi_m03.csv", "parameter_hampel_m03.csv", "kesepakatan_metode_m03.csv",
          "sensor_m03.parquet", "dampak_penanganan_m03.csv",
          "01_deteksi_harga.png", "02_bukti_kesalahan.png", "03_smoothing_sensor.png",
          "04_kejadian_sensor.png", "05_kurva_hampel.png", "06_dampak_omzet.png"]
for b in berkas:
    print(("OK    " if (OUT / b).exists() else "HILANG"), b)
```

Setelah itu: **Kernel → Restart Kernel and Run All Cells**, jalankan **Checkpoint** di Workbench, tulis
`latihan-hampel.md`, `latihan-multimetode.md`, dan `laporan.md` (ikuti `template-laporan.md`), lalu kompres
`dw-module-03/` menjadi `dw-module-03.zip`.

### Catatan penting
- **Kolom tanggal** (Nomor 2) dan **file register** (Nomor 3b) ditebak otomatis; cek keluaran `print` dan ubah
  `KOLOM_TGL` / `REGISTER` bila perlu.
- **`id_anomali` sensor** pada log (`ID_LONJAKAN`, `ID_MACET`, `ID_KOMP`, `ID_CELAH`) harus diisi dari register Modul 1 Anda.
- Parameter Hampel `JENDELA_H, K_H = 13, 5` mengikuti sel Latihan 3.1; ganti bila Anda memilih pasangan lain
  (dan perbarui alasan di `latihan-hampel.md`).
- Saya tidak dapat menjalankan kode ini terhadap data Anda, jadi bandingkan hasilnya dengan angka perkiraan di atas
  dan perbaiki bila ada `assert` yang gagal.
