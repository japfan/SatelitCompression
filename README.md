# SVD Satellite Image Processor

SVD Satellite Image Processor adalah program Python sederhana untuk pengolahan dan kompresi citra satelit menggunakan konsep Singular Value Decomposition (SVD) dari Aljabar Linier.

Program membaca citra satelit, mengubah piksel menjadi matriks, melakukan SVD, lalu merekonstruksi citra menggunakan sejumlah nilai singular terbesar. Program mendukung mode `Grayscale` dan `RGB`, termasuk format citra umum seperti `.jpg`, `.jpeg`, `.png`, `.tif`, dan `.tiff`.

## Konsep Singkat SVD

Citra satelit grayscale dapat dilihat sebagai matriks `A`, dengan setiap elemen berisi intensitas piksel dari 0 sampai 255.

Dengan SVD, matriks gambar diuraikan menjadi:

```python
U, S, VT = np.linalg.svd(A, full_matrices=False)
```

Untuk kompresi rank-k, program hanya memakai `k` nilai singular terbesar:

```python
A_k = U[:, :k] @ np.diag(S[:k]) @ VT[:k, :]
```

Semakin kecil nilai `k`, semakin sedikit data yang disimpan, tetapi detail citra dapat menurun. Semakin besar nilai `k`, hasil citra biasanya semakin mirip dengan citra asli.

## Struktur Folder

```text
svd-satellite-image-processor/
|-- main.py
|-- sample_images/
|   |-- contoh.png
|-- output/
|   |-- citra_satelit_svd_k50.png
|-- README.md
|-- requirements.txt
```

Folder `output/` akan diisi otomatis ketika program menyimpan hasil pengolahan SVD dari mode terminal.

## Instalasi

Pastikan Python sudah terpasang, lalu install library yang dibutuhkan:

```bash
pip install -r requirements.txt
```

Untuk membuat file `.exe` portable di Windows, install PyInstaller jika belum ada:

```bash
pip install pyinstaller
```

Lalu jalankan:

```bash
build_portable.bat
```

Atau pakai command langsung:

```bash
python -m PyInstaller --onefile --windowed --clean --name "SVDImageCompressor" main.py
```

Hasil build ada di:

```text
dist/SVDImageCompressor.exe
```

## Cara Menjalankan GUI

Jalankan program:

```bash
python main.py
```

Di jendela aplikasi:

- Tema awal mengikuti tema Windows. Klik ikon bulan/matahari jika ingin mengganti tema manual.
- Klik tombol `Pilih Citra`.
- Pilih satuan kompresi: `Nilai k` atau `Persen`.
- Masukkan nilai kompresi, misalnya `50` untuk nilai k atau `25` untuk 25%.
- Pilih kanal citra: `Grayscale` atau `RGB`.
- Klik tombol `Proses SVD`.
- Setelah proses selesai, klik `Simpan Citra` untuk memilih lokasi penyimpanan.

Program akan langsung menampilkan preview citra asli, resolusi, dan ukuran file asli ketika file dimasukkan. Setelah proses SVD, program menampilkan MSE, estimasi rasio data, dan waktu proses. Ukuran file kompres ditampilkan setelah citra disimpan, karena ukuran aktual baru diketahui setelah file dibuat. Dialog simpan menggunakan `.tif` sebagai format default.

Nilai `k` juga bisa diisi beberapa angka sekaligus:

```text
[10, 30, 50, 100, 200]
```

Jika beberapa nilai `k` dimasukkan, preview GUI menampilkan hasil untuk setiap nilai `k` dalam panel hasil. Tombol `Simpan Citra` menyimpan hasil dari nilai `k` terakhir.

Mode `Persen` mengubah input persen menjadi nilai `k` berdasarkan ukuran maksimal matriks. Contoh citra satelit `4000 x 6000` dengan input `25%` menghasilkan `k = 1000`, karena `25% x min(4000, 6000) = 1000`.

Mode `Grayscale` mengolah satu kanal intensitas. Mode `RGB` mempertahankan warna dengan menjalankan SVD pada kanal merah, hijau, dan biru secara terpisah.

## Cara Menjalankan Mode Terminal

Mode terminal masih tersedia dengan perintah:

```bash
python main.py --cli
```

Masukkan path citra satelit ketika diminta:

```text
sample_images/contoh.png
```

Masukkan nilai `k`:

```text
50
```

Program juga bisa menerima beberapa nilai `k` sekaligus:

```text
[10, 30, 50, 100, 200]
```

Jika nilai `k` lebih besar dari `min(m, n)`, program akan menyesuaikannya ke nilai valid terbesar.

## Contoh Output Teks

```text
Ukuran citra satelit: 512 x 512

Nilai k: 50
Jumlah data asli: 262144
Jumlah data setelah kompresi rank-50: 51250
Rasio representasi data matriks: 19.55%
MSE: 12.34
Ukuran file asli: 80.12 KB
Ukuran file hasil kompresi: 42.30 KB
Rasio ukuran file hasil/asli: 52.80%
Citra satelit hasil kompresi disimpan: output/citra_satelit_svd_k50.png
```

## Output Visual

Program menampilkan perbandingan:

- Citra satelit asli
- Citra satelit hasil SVD untuk setiap nilai `k` dalam satu figure matplotlib
- Rasio representasi data matriks dengan rumus `k * (m + n + 1) / (m * n) * 100%`
- Informasi ukuran file asli, ukuran hasil kompresi, dan MSE

Jika beberapa nilai `k` dimasukkan, semua hasilnya akan ditampilkan dalam satu jendela `matplotlib`.

Pada GUI, preview hasil kompresi menampilkan semua nilai `k` yang dimasukkan.

## Validasi Error

Program menangani beberapa error umum:

- File citra tidak ditemukan.
- Format file bukan `.jpg`, `.jpeg`, `.png`, `.tif`, atau `.tiff`.
- Nilai `k` kurang dari atau sama dengan 0.
- Nilai `k` lebih besar dari `min(jumlah baris, jumlah kolom)`.
