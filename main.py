from pathlib import Path
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox
try:
    import winreg
except ImportError:
    winreg = None

import customtkinter as ctk
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk, UnidentifiedImageError


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
SUPPORTED_FORMAT_TEXT = ".jpg, .jpeg, .png, .tif, atau .tiff"
OUTPUT_DIR = Path("output")


def windows_prefers_dark_mode() -> bool:
    if winreg is None:
        return False

    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
        ) as key:
            apps_use_light_theme, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            return apps_use_light_theme == 0
    except OSError:
        return False


def read_image_path() -> Path:
    image_path = Path(input("Masukkan path citra satelit: ").strip().strip('"'))
    if not image_path.exists():
        raise FileNotFoundError("File citra tidak ditemukan. Periksa kembali nama file atau path citra.")
    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Format file tidak valid. Gunakan file {SUPPORTED_FORMAT_TEXT}.")
    return image_path


def load_grayscale_matrix(image_path: Path) -> np.ndarray:
    try:
        image = Image.open(image_path).convert("L")
    except UnidentifiedImageError as exc:
        raise ValueError(f"Format file tidak valid. Gunakan file {SUPPORTED_FORMAT_TEXT}.") from exc

    return np.array(image, dtype=float)


def load_image_matrix(image_path: Path, color_mode: str) -> np.ndarray:
    try:
        if color_mode == "RGB":
            image = Image.open(image_path).convert("RGB")
        else:
            image = Image.open(image_path).convert("L")
    except UnidentifiedImageError as exc:
        raise ValueError(f"Format file tidak valid. Gunakan file {SUPPORTED_FORMAT_TEXT}.") from exc

    return np.array(image, dtype=float)


def validate_image_path(image_path: Path) -> None:
    if not image_path.exists():
        raise FileNotFoundError("File citra tidak ditemukan. Periksa kembali nama file atau path citra.")
    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Format file tidak valid. Gunakan file {SUPPORTED_FORMAT_TEXT}.")


def parse_k_values(raw_value: str, max_k: int) -> list[int]:
    if not raw_value.strip():
        raise ValueError("Nilai k harus lebih dari 0.")

    k_values: list[int] = []
    # Input dibuat fleksibel agar saat presentasi bisa memakai "10,30,50"
    # atau format seperti list Python: "[10, 30, 50, 100, 200]".
    cleaned_value = raw_value.strip().strip("[]")
    for item in cleaned_value.split(","):
        item = item.strip()
        if not item:
            continue

        try:
            k = int(item)
        except ValueError as exc:
            raise ValueError("Nilai k harus berupa bilangan bulat.") from exc

        if k <= 0:
            # Nilai tidak valid diabaikan agar batch beberapa k tetap bisa berjalan.
            print(f"Nilai k = {k} diabaikan karena harus lebih dari 0.")
            continue
        if k > max_k:
            # Rank SVD tidak boleh melebihi min(m, n), jadi k disesuaikan.
            print(f"Nilai k = {k} melebihi batas. Disesuaikan menjadi {max_k}.")
            k = max_k
        k_values.append(k)

    if not k_values:
        raise ValueError("Tidak ada nilai k yang valid.")

    return k_values


def parse_compression_values(raw_value: str, max_k: int, unit: str) -> list[int]:
    if unit == "Persen":
        if not raw_value.strip():
            raise ValueError("Persen kompresi harus lebih dari 0.")

        k_values = []
        for item in raw_value.split(","):
            try:
                percentage = float(item.strip())
            except ValueError as exc:
                raise ValueError("Persen kompresi harus berupa angka.") from exc

            if percentage <= 0:
                raise ValueError("Persen kompresi harus lebih dari 0.")
            if percentage > 100:
                raise ValueError("Persen kompresi maksimal adalah 100.")

            k_values.append(max(1, min(max_k, round((percentage / 100) * max_k))))
        return k_values

    return parse_k_values(raw_value, max_k)


def read_k_values(max_k: int) -> list[int]:
    raw_value = input("Masukkan nilai k (contoh: 50 atau [10, 30, 50, 100, 200]): ").strip()
    return parse_k_values(raw_value, max_k)


def compress_with_rank_k(u: np.ndarray, s: np.ndarray, vt: np.ndarray, k: int) -> np.ndarray:
    reconstructed = u[:, :k] @ np.diag(s[:k]) @ vt[:k, :]
    return np.clip(reconstructed, 0, 255)


def compress_matrix_with_rank_k(matrix: np.ndarray, k: int) -> np.ndarray:
    if matrix.ndim == 2:
        u, s, vt = np.linalg.svd(matrix, full_matrices=False)
        return compress_with_rank_k(u, s, vt, k)

    channels = []
    for channel_index in range(matrix.shape[2]):
        channel = matrix[:, :, channel_index]
        u, s, vt = np.linalg.svd(channel, full_matrices=False)
        channels.append(compress_with_rank_k(u, s, vt, k))
    return np.stack(channels, axis=2)


def storage_stats(matrix_shape: tuple[int, ...], k: int) -> tuple[int, int, float]:
    rows, cols = matrix_shape[:2]
    channels = matrix_shape[2] if len(matrix_shape) == 3 else 1
    original_data = rows * cols * channels
    compressed_data = k * (rows + cols + 1) * channels
    stored_ratio = (compressed_data / original_data) * 100
    return original_data, compressed_data, stored_ratio


def mean_squared_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    # MSE mengukur rata-rata kuadrat selisih piksel asli dan hasil rekonstruksi.
    # Semakin kecil MSE, hasil SVD semakin dekat dengan citra asli.
    return float(np.mean((original - reconstructed) ** 2))


def print_stats(matrix_shape: tuple[int, int], compressed_images: dict[int, np.ndarray],
                original: np.ndarray) -> None:
    rows, cols = matrix_shape
    print(f"\nUkuran citra satelit: {rows} x {cols}")

    for k, image_matrix in compressed_images.items():
        original_data, compressed_data, stored_ratio = storage_stats(matrix_shape, k)
        mse = mean_squared_error(original, image_matrix)
        print(f"\nNilai k: {k}")
        print(f"Jumlah data asli: {original_data}")
        print(f"Jumlah data setelah kompresi rank-{k}: {compressed_data}")
        print(f"Rasio representasi data matriks: {stored_ratio:.2f}%")
        print(f"MSE: {mse:.2f}")


def print_file_size_stats(original_path: Path, output_paths: dict[int, Path]) -> None:
    for k, output_path in output_paths.items():
        original_size, compressed_size, file_ratio = file_size_stats(original_path, output_path)
        print(f"\nPerbandingan ukuran file untuk k = {k}")
        print(f"Ukuran file asli: {format_file_size(original_size)}")
        print(f"Ukuran file hasil kompresi: {format_file_size(compressed_size)}")
        print(f"Rasio ukuran file hasil/asli: {file_ratio:.2f}%")


def save_compressed_images(images: dict[int, np.ndarray]) -> dict[int, Path]:
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_paths = {}
    for k, image_matrix in images.items():
        output_path = OUTPUT_DIR / f"citra_satelit_svd_k{k}.png"
        Image.fromarray(image_matrix.astype(np.uint8)).save(output_path)
        output_paths[k] = output_path
        print(f"Citra satelit hasil kompresi disimpan: {output_path}")
    return output_paths


def format_file_size(size_in_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(size_in_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size_in_bytes} B"


def file_size_stats(original_path: Path, output_path: Path) -> tuple[int, int, float]:
    original_size = original_path.stat().st_size
    compressed_size = output_path.stat().st_size
    ratio = (compressed_size / original_size) * 100 if original_size else 0
    return original_size, compressed_size, ratio


def show_comparison(original: np.ndarray, compressed_images: dict[int, np.ndarray]) -> None:
    total_images = 1 + len(compressed_images)
    columns = min(3, total_images)
    rows = int(np.ceil(total_images / columns))

    plt.figure(figsize=(5 * columns, 4 * rows))

    # Subplot pertama selalu citra asli sebagai pembanding visual.
    plt.subplot(rows, columns, 1)
    plt.imshow(original, cmap="gray")
    plt.title("Citra Satelit Asli")
    plt.axis("off")

    # Setiap nilai k ditampilkan dalam subplot sendiri lengkap dengan rasio dan MSE.
    for index, (k, image_matrix) in enumerate(compressed_images.items(), start=2):
        _, _, stored_ratio = storage_stats(original.shape, k)
        mse = mean_squared_error(original, image_matrix)
        plt.subplot(rows, columns, index)
        plt.imshow(image_matrix, cmap="gray")
        plt.title(f"SVD k={k}\nRasio: {stored_ratio:.2f}% | MSE: {mse:.2f}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def main() -> None:
    try:
        image_path = read_image_path()
        matrix = load_grayscale_matrix(image_path)

        rows, cols = matrix.shape
        print(f"Ukuran matriks citra satelit: {rows} x {cols}")
        k_values = read_k_values(min(rows, cols))

        print("\nMelakukan Singular Value Decomposition...")
        start_time = time.perf_counter()
        u, s, vt = np.linalg.svd(matrix, full_matrices=False)

        # Dictionary menjaga pasangan k dan citra hasil kompresi agar mudah
        # dipakai untuk statistik teks, penyimpanan file, dan subplot matplotlib.
        compressed_images = {
            k: compress_with_rank_k(u, s, vt, k)
            for k in k_values
        }
        process_time = time.perf_counter() - start_time

        print_stats(matrix.shape, compressed_images, matrix)
        print(f"\nWaktu proses kompresi: {process_time:.2f} detik")
        output_paths = save_compressed_images(compressed_images)
        print_file_size_stats(image_path, output_paths)
        show_comparison(matrix, compressed_images)
    except (FileNotFoundError, ValueError) as error:
        print(error)



if __name__ == "__main__":
    import sys
    if "--cli" in sys.argv:
        main()
    else:
        from gui import run_gui
        run_gui()
