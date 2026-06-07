# SVD Game Image Compressor

SVD Game Image Compressor adalah program Python sederhana untuk kompresi gambar game menggunakan konsep Singular Value Decomposition (SVD) dari Aljabar Linier.

Program membaca gambar game, mengubah piksel menjadi matriks, melakukan SVD, lalu merekonstruksi gambar menggunakan sejumlah nilai singular terbesar. Program mendukung mode `Grayscale`, `RGB`, dan `RGBA`, termasuk format gambar umum seperti `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, `.webp`, `.bmp`, dan `.tga`.

Mode `RGBA` berguna untuk asset game yang memiliki transparansi, seperti sprite, ikon, UI, item, atau tileset. Jika gambar diproses sebagai `RGBA`, kanal alpha dipertahankan agar transparansi tidak rusak. Hasilnya disimpan sebagai WEBP secara default karena mendukung alpha dan biasanya lebih kecil dari PNG untuk hasil SVD. PNG dan TIFF tetap tersedia jika butuh format lossless. Jika gambar bukan `RGBA`, dialog simpan tetap memakai JPG sebagai default.

## Konsep Singkat SVD

Gambar grayscale dapat dilihat sebagai matriks `A`, dengan setiap elemen berisi intensitas piksel dari 0 sampai 255.

Dengan SVD, matriks gambar diuraikan menjadi:

```python
U, S, VT = np.linalg.svd(A, full_matrices=False)
```

Untuk kompresi rank-k, program hanya memakai `k` nilai singular terbesar:

```python
A_k = U[:, :k] @ np.diag(S[:k]) @ VT[:k, :]
```

Semakin kecil nilai `k`, semakin sedikit data yang disimpan, tetapi detail gambar dapat menurun. Semakin besar nilai `k`, hasil gambar biasanya semakin mirip dengan gambar asli.

Untuk mode `RGB`, SVD dijalankan pada setiap kanal warna secara terpisah. Untuk mode `RGBA`, SVD dijalankan pada kanal R, G, dan B, sementara kanal alpha dipertahankan dari gambar asli.

## Struktur Folder

```text
svd-game-image-compressor/
|-- main.py
|-- gui.py
|-- sample_images/
|   |-- contoh.png
|-- output/
|   |-- gambar_game_svd_k50.png
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

- Klik tombol `Pilih Gambar`.
- Masukkan nilai `k`, misalnya `50`.
- Pilih kanal gambar: `Grayscale`, `RGB`, atau `RGBA`.
- Klik tombol `Proses`.
- Setelah proses selesai, klik `Simpan` untuk memilih lokasi penyimpanan.

Program akan langsung menampilkan preview gambar asli, resolusi, dan ukuran file asli ketika file dimasukkan. Setelah proses SVD, program menampilkan PSNR, estimasi rasio data, waktu proses, dan estimasi ukuran file hasil tanpa harus menyimpan lebih dulu.

Nilai `k` juga bisa diisi beberapa angka sekaligus:

```text
[10, 30, 50, 100, 200]
```

Jika beberapa nilai `k` dimasukkan, preview GUI menampilkan hasil untuk setiap nilai `k` dalam panel hasil.

## Cara Menjalankan Mode Terminal

Mode terminal masih tersedia dengan perintah:

```bash
python main.py --cli
```

Masukkan path gambar game ketika diminta:

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

Jika nilai `k` lebih besar dari `min(jumlah baris, jumlah kolom)`, program akan menyesuaikannya ke nilai valid terbesar.

## Contoh Output Teks

```text
Ukuran gambar game: 512 x 512

Nilai k: 50
Jumlah data asli: 262144
Jumlah data setelah kompresi rank-50: 51250
Rasio representasi data matriks: 19.55%
PSNR: 37.22 dB
Ukuran file asli: 80.12 KB
Ukuran file hasil kompresi: 42.30 KB
Rasio ukuran file hasil/asli: 52.80%
Gambar game hasil kompresi disimpan: output/gambar_game_svd_k50.png
```

## Catatan Untuk Gambar Game

SVD cocok untuk texture, background, ilustrasi, dan asset yang detailnya relatif halus. Untuk pixel art, ikon kecil, font bitmap, atau UI dengan tepi tajam, nilai `k` yang terlalu kecil dapat membuat hasil terlihat blur.

Ukuran file PNG hasil SVD tidak selalu lebih kecil dari PNG asli. PNG adalah format lossless, sedangkan rekonstruksi SVD dapat membuat banyak variasi warna baru yang lebih sulit dikompresi. Untuk gambar tanpa transparansi, gunakan mode `RGB` agar hasil default disimpan sebagai JPG. Untuk gambar transparan, gunakan `RGBA` dan simpan sebagai WEBP untuk ukuran lebih kecil, atau PNG jika harus lossless.

Gunakan mode `RGBA` untuk asset transparan. Jika tidak membutuhkan transparansi, gunakan `RGB` agar hasil default tetap disimpan sebagai JPG.

## Validasi Error

Program menangani beberapa error umum:

- File gambar tidak ditemukan.
- Format file bukan format yang didukung.
- Nilai `k` kurang dari atau sama dengan 0.
- Nilai `k` lebih besar dari `min(jumlah baris, jumlah kolom)`.
