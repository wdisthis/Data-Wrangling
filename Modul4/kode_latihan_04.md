# Kode Latihan 4.1 dan 4.2 — Modul 4 (04_duplikat.ipynb)

Setiap blok kode di bawah ini adalah satu sel notebook (satu TODO). Jalankan berurutan **setelah** Bagian A–D selesai dijalankan, karena kode memakai variabel dari sel-sel sebelumnya (`wu`, `tgl_web`, `tk1`, `app`, `produk`, `produk_m04`, `OUT`, `kunci_teks`, `kunci_wilayah`, `PROVINSI`, `catat_keputusan`, dan seterusnya).

---

## Latihan 4.1 — Parser tanggal daftar pelanggan web

### TODO 1

```python
# TODO 1: pola penulisan tgl_daftar (wu) dan banyak barisnya
# Pola dibentuk dengan mengganti setiap angka menjadi 9 dan setiap huruf menjadi a.
pola_tgl = (wu.tgl_daftar.str.replace(r"\d", "9", regex=True)
            .str.replace(r"[A-Za-z]", "a", regex=True))

ringkas_pola = (pd.DataFrame({"pola": pola_tgl, "contoh": wu.tgl_daftar})
                .groupby("pola")
                .agg(n_baris=("contoh", "size"), contoh=("contoh", "first"))
                .reset_index())

print("baris web tanpa salinan eksak:", len(wu))
print(ringkas_pola.to_string(index=False))
```

### TODO 2

```python
# TODO 2: pola dd/mm atau mm/dd? bukti dari nilai > 12
# Untuk pola bergaris miring: nilai maksimum bagian pertama dan kedua,
# serta banyak baris yang bagian pertamanya (dan kedua) > 12.
garis = pola_tgl.str.contains("/")
bagian = wu.tgl_daftar[garis].str.split("/", expand=True).astype(int)
bagian.columns = ["bagian_1", "bagian_2", "bagian_3"]

bukti_urutan = (bagian.groupby(pola_tgl[garis])
                .agg(maks_bagian_1=("bagian_1", "max"),
                     maks_bagian_2=("bagian_2", "max"),
                     n_bagian_1_gt12=("bagian_1", lambda s: int(s.gt(12).sum())),
                     n_bagian_2_gt12=("bagian_2", lambda s: int(s.gt(12).sum()))))

print(bukti_urutan.to_string())
print("\nKesimpulan: bila bagian pertama pernah > 12 dan bagian kedua tidak pernah > 12,")
print("urutannya pasti hari/bulan (dd/mm); bila sebaliknya, mm/dd.")
```

### TODO 3

```python
# TODO 3: format="mixed" dan format="mixed", dayfirst=True vs tgl_web: beda per pola
mixed = pd.to_datetime(wu.tgl_daftar, format="mixed")
mixed_dayfirst = pd.to_datetime(wu.tgl_daftar, format="mixed", dayfirst=True)

beda_mixed = mixed.ne(tgl_web)
beda_mixed_dayfirst = mixed_dayfirst.ne(tgl_web)

beda_per_pola = pd.DataFrame({
    "n_baris": pola_tgl.value_counts(),
    "beda_mixed": beda_mixed.groupby(pola_tgl).sum().astype(int),
    "beda_mixed_dayfirst": beda_mixed_dayfirst.groupby(pola_tgl).sum().astype(int),
})
print(beda_per_pola.to_string())

# contoh baris yang berbeda, untuk menjelaskan penyebabnya (tidak ada galat, tetapi nilai berbeda)
contoh_beda = pd.DataFrame({"teks": wu.tgl_daftar, "pola": pola_tgl, "eksplisit": tgl_web,
                            "mixed": mixed, "mixed_dayfirst": mixed_dayfirst})
print(contoh_beda[beda_mixed | beda_mixed_dayfirst].groupby("pola").head(3).to_string())
```

### TODO 4

```python
# TODO 4: dampak pada tanggal daftar sama di tautan tingkat 1 (tk1)
# Tanggal daftar web dari tiga parser x tanggal daftar aplikasi dalam WIB dan dalam UTC.
parser_web = {"eksplisit (tgl_web)": tgl_web,
              "mixed": mixed,
              "mixed_dayfirst": mixed_dayfirst}

# tanggal aplikasi: WIB (hasil konversi D.1) dan UTC (teks apa adanya)
app_wib = tk1.tanggal_daftar_app.dt.normalize().to_numpy()
utc_app_tk1 = pd.to_datetime(app.registered_at, format="%Y-%m-%dT%H:%M:%SZ") \
                .set_axis(app.customer_id)
app_utc = utc_app_tk1[tk1.id_pelanggan_app].dt.normalize().to_numpy()

baris_dampak = []
for nama_parser, seri in parser_web.items():
    web_per_id = pd.Series(seri.to_numpy(), index=wu.id_pelanggan.to_numpy())
    web_tk1 = web_per_id[tk1.id_pelanggan_web].dt.normalize().to_numpy()
    baris_dampak.append({
        "parser_web": nama_parser,
        "persen_sama_app_WIB": round(100 * float((web_tk1 == app_wib).mean()), 1),
        "persen_sama_app_UTC": round(100 * float((web_tk1 == app_utc).mean()), 1)})

dampak_tk1 = pd.DataFrame(baris_dampak)
print(f"tautan tingkat 1: {len(tk1)} pasangan")
print(dampak_tk1.to_string(index=False))
```

### TODO 5

```python
# TODO 5: simpan OUT / "parser_tanggal_m04.csv"
# Kolom: pola, n_baris, contoh, maks_bagian_1, maks_bagian_2, beda_mixed, beda_mixed_dayfirst
# maks_bagian_1/2 hanya bermakna untuk pola bergaris miring; pola lain dibiarkan kosong (NaN).
parser_tanggal = (ringkas_pola
                  .merge(bukti_urutan[["maks_bagian_1", "maks_bagian_2"]],
                         left_on="pola", right_index=True, how="left")
                  .merge(beda_per_pola[["beda_mixed", "beda_mixed_dayfirst"]],
                         left_on="pola", right_index=True, how="left"))

parser_tanggal = parser_tanggal[["pola", "n_baris", "contoh", "maks_bagian_1",
                                 "maks_bagian_2", "beda_mixed", "beda_mixed_dayfirst"]]
parser_tanggal.to_csv(OUT / "parser_tanggal_m04.csv", index=False)

assert parser_tanggal.n_baris.sum() == len(wu), "jumlah baris per pola harus sama dengan jumlah pelanggan web"
print(parser_tanggal.to_string(index=False))
```

---

## Latihan 4.2 — Duplikat produk dan toko?

### TODO 1

```python
# TODO 1: keunikan kode_produk, nama_produk; kandidat duplikat dekat produk
print("kode_produk unik :", produk.kode_produk.is_unique)
print("nama_produk unik :", produk.nama_produk.is_unique,
      "| baris dengan nama ganda:", int(produk.nama_produk.duplicated(keep=False).sum()))

# duplikat dekat: identik pada seluruh kolom selain kode_produk dan nama_produk
# (memakai kolom kanonik produk_m04, bukan kolom *_asli)
KOLOM_ISI = ["id_toko", "nama_toko", "kategori", "harga_katalog_rp", "berat_gram",
             "provinsi_toko", "perlu_pendingin", "rating_toko", "rating_toko_status"]
dekat_produk = produk_m04.duplicated(subset=KOLOM_ISI, keep=False)
print("baris dalam kelompok kandidat duplikat dekat:", int(dekat_produk.sum()))
print(produk_m04[dekat_produk].sort_values(KOLOM_ISI)[["kode_produk", "nama_produk"] + KOLOM_ISI]
      .head(10).to_string(index=False))
```

### TODO 2

```python
# TODO 2: bagian-bagian nama_produk vs kategori, provinsi_toko, nomor kode
# nama_produk berbentuk <kategori> <wilayah> <nomor>
pm = produk_m04.copy()

# nomor = angka di ujung nama; kepala = sisanya (kategori + wilayah)
ekstrak = pm.nama_produk.str.extract(r"^(?P<kepala>.*?)\s+(?P<nomor_nama>\d+)\s*$")
pm["kepala"], pm["nomor_nama"] = ekstrak.kepala, ekstrak.nomor_nama

# bagian 1: kategori pada nama dibandingkan dengan kategori kanonik
pm["kategori_cocok"] = [isinstance(k, str) and kunci_teks(k).startswith(kunci_teks(kat))
                        for k, kat in zip(pm.kepala, pm.kategori)]

# bagian 2: wilayah = kepala tanpa awalan kategori; bandingkan dengan provinsi_toko
pm["wilayah_nama"] = [k[len(kat):].strip() if isinstance(k, str) and kc else pd.NA
                      for k, kat, kc in zip(pm.kepala, pm.kategori, pm.kategori_cocok)]
pm["wilayah_cocok"] = [isinstance(w, str) and kunci_wilayah(w) == kunci_wilayah(p)
                       for w, p in zip(pm.wilayah_nama, pm.provinsi_toko)]

# bagian 3: nomor pada nama dibandingkan dengan nomor pada kode_produk (LS-KOP-0001 -> 1)
nomor_kode = pm.kode_produk.str.extract(r"(\d+)$")[0]
pm["nomor_cocok"] = (pm.nomor_nama.notna() & nomor_kode.notna()
                     & (pd.to_numeric(pm.nomor_nama, errors="coerce")
                        == pd.to_numeric(nomor_kode, errors="coerce")))

# bagian nama yang tidak dapat diurai (ambigu)
pm["nama_ambigu"] = pm.nomor_nama.isna() | pm.wilayah_nama.isna()

print(pm[["kategori_cocok", "wilayah_cocok", "nomor_cocok", "nama_ambigu"]].sum().astype(int).to_string())
print("\nsemua bagian konsisten:", int((pm.kategori_cocok & pm.wilayah_cocok & pm.nomor_cocok).sum()),
      "dari", len(pm))
print("\ncontoh yang tidak konsisten atau ambigu:")
tak_konsisten = ~(pm.kategori_cocok & pm.wilayah_cocok & pm.nomor_cocok)
print(pm[tak_konsisten][["kode_produk", "nama_produk", "kategori", "provinsi_toko",
                         "wilayah_nama", "kategori_cocok", "wilayah_cocok", "nomor_cocok"]]
      .head(10).to_string(index=False))
```

### TODO 3

```python
# TODO 3: konsistensi atribut per id_toko -> OUT / "konsistensi_toko_m04.csv"
# Per id_toko: banyak produk, banyak nama_toko, banyak provinsi_toko, rentang rating_toko (tanpa penanda).
# rating_toko pada produk_m04 sudah NaN untuk penanda, sehingga min/max otomatis mengabaikannya.
konsistensi_toko = (produk_m04.groupby("id_toko")
                    .agg(n_produk=("kode_produk", "size"),
                         n_nama_toko=("nama_toko", "nunique"),
                         n_provinsi_toko=("provinsi_toko", "nunique"),
                         n_rating_berbeda=("rating_toko", "nunique"),
                         rating_min=("rating_toko", "min"),
                         rating_maks=("rating_toko", "max"),
                         n_rating_penanda=("rating_toko_status", lambda s: int(s.eq("penanda").sum())))
                    .reset_index())
konsistensi_toko["rentang_rating"] = (konsistensi_toko.rating_maks - konsistensi_toko.rating_min).round(2)

konsistensi_toko.to_csv(OUT / "konsistensi_toko_m04.csv", index=False)

print("jumlah toko:", len(konsistensi_toko), "| total produk:", int(konsistensi_toko.n_produk.sum()))
print("toko dengan >1 nama_toko    :", int(konsistensi_toko.n_nama_toko.gt(1).sum()))
print("toko dengan >1 provinsi_toko:", int(konsistensi_toko.n_provinsi_toko.gt(1).sum()))
print("toko dengan rentang rating > 0:", int(konsistensi_toko.rentang_rating.gt(0).sum()))
print(konsistensi_toko.describe().loc[["min", "50%", "max"]].to_string())
```

### TODO 4

```python
# TODO 4: keputusan dan log
# Dapatkah tabel toko (satu baris per toko) diturunkan dari katalog? Keputusan didasarkan pada angka TODO 3.
k = konsistensi_toko
n_toko = len(k)
n_nama_tak_konsisten = int(k.n_nama_toko.gt(1).sum())
n_prov_tak_konsisten = int(k.n_provinsi_toko.gt(1).sum())
n_rating_tak_konsisten = int(k.rentang_rating.gt(0).sum())
n_toko_hanya_penanda = int(k.rating_min.isna().sum())

dapat_diturunkan = (n_nama_tak_konsisten == 0) and (n_prov_tak_konsisten == 0)
rating_seragam = (n_rating_tak_konsisten == 0)

if dapat_diturunkan:
    keputusan = ("tabel toko diturunkan dari katalog: satu baris per id_toko dengan nama_toko dan provinsi_toko "
                 + ("serta rating_toko" if rating_seragam else "(rating_toko tidak seragam, tidak dibawa ke tabel toko)"))
else:
    keputusan = "tabel toko tidak diturunkan otomatis dari katalog: atribut toko tidak konsisten antarproduk"

print(f"{n_toko} toko | nama_toko tak konsisten {n_nama_tak_konsisten} | provinsi_toko tak konsisten "
      f"{n_prov_tak_konsisten} | rating tak seragam {n_rating_tak_konsisten} | hanya penanda {n_toko_hanya_penanda}")
print("keputusan:", keputusan)

log = catat_keputusan([
    {"id": "K-04-013", "modul": 4, "sumber_kolom": "produk.id_toko;produk.nama_toko;produk.provinsi_toko;produk.rating_toko",
     "temuan": "pemeriksaan duplikat produk dan konsistensi atribut toko pada katalog",
     "bukti": (f"kode_produk unik {produk.kode_produk.is_unique}, nama_produk unik {produk.nama_produk.is_unique}; "
               f"{n_toko} toko; nama_toko tak konsisten {n_nama_tak_konsisten}, provinsi_toko tak konsisten "
               f"{n_prov_tak_konsisten}, rentang rating > 0 pada {n_rating_tak_konsisten} toko, "
               f"toko hanya berpenanda rating {n_toko_hanya_penanda}"),
     "tindakan": keputusan,
     "alasan": "tabel toko hanya sah diturunkan bila setiap atribut toko bernilai tunggal pada seluruh produknya",
     "baris_terdampak": n_toko, "id_anomali": "A-19"},
])
log[log.id.eq("K-04-013")][["id", "sumber_kolom", "baris_terdampak", "id_anomali"]]
```

---

## Catatan

- Jawaban tertulis tetap harus kamu isi sendiri di `dw-module-04/latihan-tanggal.md` dan `dw-module-04/latihan-produk.md`, memakai angka yang keluar dari sel-sel di atas.
- Log `K-04-013` dipakai untuk keputusan Latihan 4.2. Bila kamu menambahkan log untuk Latihan 4.1 dan aturan entitas tanpa HP (Tugas Individu no. 4), pakai id berikutnya (`K-04-014`, `K-04-015`, dst.) agar tidak bentrok.
- Nilai `id_anomali` pada log `K-04-013` (`A-19`) saya ambil dari K-04-005 yang menangani rating toko dan atribut katalog; sesuaikan dengan register Modul 1 milikmu bila kodenya berbeda.
