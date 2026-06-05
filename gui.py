"""Simple tkinter GUI for SVD-based satellite image compression."""
from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from io import BytesIO
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from main import (
    SUPPORTED_FORMAT_TEXT,
    compress_matrix_with_rank_k,
    format_file_size,
    load_image_matrix,
    parse_compression_values,
    storage_stats,
    validate_image_path,
)


class ImagePreview(ttk.Frame):
    def __init__(self, master, placeholder_text: str = "Belum ada citra") -> None:
        super().__init__(master)
        self.placeholder_text = placeholder_text
        self.pil_image: Image.Image | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.drag_x = 0
        self.drag_y = 0
        self.fitted = True

        self.canvas = tk.Canvas(
            self,
            bg="#e5e7eb",
            highlightthickness=1,
            highlightbackground="#c0c0c0",
        )
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<MouseWheel>", self._on_scroll)
        self.canvas.bind("<Button-4>", self._on_scroll)
        self.canvas.bind("<Button-5>", self._on_scroll)
        self.canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<Double-Button-1>", self._reset_fit)

    def set_image(self, image: Image.Image) -> None:
        self.pil_image = image.copy()
        self.fitted = True
        self.after(20, self._fit_and_redraw)

    def clear(self) -> None:
        self.pil_image = None
        self.photo = None
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self._redraw()

    def _fit_and_redraw(self) -> None:
        self._fit()
        self._redraw()

    def _fit(self) -> None:
        if self.pil_image is None:
            return

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            return

        image_width, image_height = self.pil_image.size
        self.scale = min(canvas_width / image_width, canvas_height / image_height) * 0.95
        self.offset_x = (canvas_width - image_width * self.scale) / 2
        self.offset_y = (canvas_height - image_height * self.scale) / 2

    def _redraw(self) -> None:
        self.canvas.delete("all")
        if self.pil_image is None:
            self.canvas.create_text(
                self.canvas.winfo_width() / 2,
                self.canvas.winfo_height() / 2,
                text=self.placeholder_text,
                fill="#555555",
                font=("Segoe UI", 11),
            )
            return

        image_width, image_height = self.pil_image.size
        draw_width = max(1, int(image_width * self.scale))
        draw_height = max(1, int(image_height * self.scale))
        resized = self.pil_image.resize((draw_width, draw_height), Image.BILINEAR)
        self.photo = ImageTk.PhotoImage(resized)
        self.canvas.create_image(self.offset_x, self.offset_y, image=self.photo, anchor="nw")

    def _on_resize(self, _event) -> None:
        if self.fitted:
            self._fit()
        self._redraw()

    def _on_scroll(self, event) -> str:
        if self.pil_image is None:
            return "break"

        delta = getattr(event, "delta", 0)
        zoom_in = delta > 0 or getattr(event, "num", None) == 4
        factor = 1.15 if zoom_in else 1 / 1.15
        new_scale = max(0.05, min(20.0, self.scale * factor))
        ratio = new_scale / self.scale
        self.offset_x = event.x - (event.x - self.offset_x) * ratio
        self.offset_y = event.y - (event.y - self.offset_y) * ratio
        self.scale = new_scale
        self.fitted = False
        self._redraw()
        return "break"

    def _on_drag_start(self, event) -> None:
        self.drag_x = event.x
        self.drag_y = event.y

    def _on_drag(self, event) -> None:
        if self.pil_image is None:
            return

        self.offset_x += event.x - self.drag_x
        self.offset_y += event.y - self.drag_y
        self.drag_x = event.x
        self.drag_y = event.y
        self.fitted = False
        self._redraw()

    def _reset_fit(self, _event=None) -> None:
        self.fitted = True
        self._fit_and_redraw()


class SVDCompressorGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Kompresi Citra Satelit SVD")
        self.root.geometry("1080x760")
        self.root.minsize(900, 650)

        self.image_path: Path | None = None
        self.original_pil: Image.Image | None = None
        self.compressed_pil: Image.Image | None = None
        self.compressed_images_by_k: dict[int, np.ndarray] = {}
        self.last_compressed_image: np.ndarray | None = None
        self.last_compressed_k: int | None = None
        self.last_original_path: Path | None = None
        self.last_matrix_shape: tuple[int, ...] = ()
        self.last_k_values: list[int] = []
        self.last_process_time = 0.0
        self.last_mse: float | None = None
        self.is_compressing = False
        self.worker_queue: queue.Queue[tuple] = queue.Queue()
        self.result_widgets: list[tk.Widget] = []

        self.path_var = tk.StringVar(value="Belum ada citra satelit dipilih")
        self.k_var = tk.StringVar(value="50")
        self.unit_var = tk.StringVar(value="Nilai k")
        self.color_mode_var = tk.StringVar(value="Grayscale")
        self.status_var = tk.StringVar(value="Pilih citra, masukkan nilai k, lalu klik Proses.")

        self.metric_vars = {
            "Resolusi": tk.StringVar(value="- x -"),
            "Ukuran Asli": tk.StringVar(value="-"),
            "Ukuran Kompres": tk.StringVar(value="-"),
            "MSE": tk.StringVar(value="-"),
            "Nilai k": tk.StringVar(value="-"),
            "Rasio Data": tk.StringVar(value="-"),
            "Waktu Proses": tk.StringVar(value="-"),
        }

        self._setup_style()
        self._build_layout()
        self._poll_worker_queue()

    def _setup_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        self.root.configure(bg="#f3f4f6")
        style.configure(".", font=("Segoe UI", 10))
        style.configure("TFrame", background="#f3f4f6")
        style.configure("Panel.TFrame", background="#ffffff", relief="solid", borderwidth=1)
        style.configure("TLabel", background="#f3f4f6", foreground="#111827")
        style.configure("Panel.TLabel", background="#ffffff", foreground="#111827")
        style.configure("Muted.TLabel", background="#ffffff", foreground="#4b5563")
        style.configure("Title.TLabel", background="#f3f4f6", font=("Segoe UI", 16, "bold"))
        style.configure("Section.TLabel", background="#ffffff", font=("Segoe UI", 10, "bold"))
        style.configure("Accent.TButton", padding=(10, 5))
        style.configure("TButton", padding=(8, 4))

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)

        header = ttk.Frame(self.root, padding=(16, 12, 16, 6))
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text="Kompresi Citra Satelit dengan SVD", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Program sederhana untuk membandingkan citra asli dan hasil kompresi.",
            foreground="#4b5563",
        ).pack(anchor="w", pady=(2, 0))

        controls = ttk.Frame(self.root, style="Panel.TFrame", padding=12)
        controls.grid(row=1, column=0, padx=16, pady=(4, 10), sticky="ew")
        controls.columnconfigure(1, weight=1)

        ttk.Button(controls, text="Pilih Citra", command=self.choose_image).grid(row=0, column=0, sticky="w")
        ttk.Label(controls, textvariable=self.path_var, style="Muted.TLabel").grid(
            row=0, column=1, columnspan=5, padx=(10, 0), sticky="ew"
        )

        ttk.Label(controls, text="Nilai SVD", style="Muted.TLabel").grid(row=1, column=0, pady=(12, 2), sticky="w")
        ttk.Label(controls, text="Satuan", style="Muted.TLabel").grid(row=1, column=1, padx=(12, 0), pady=(12, 2), sticky="w")
        ttk.Label(controls, text="Kanal Citra", style="Muted.TLabel").grid(row=1, column=2, padx=(12, 0), pady=(12, 2), sticky="w")

        ttk.Entry(controls, textvariable=self.k_var, width=22).grid(row=2, column=0, sticky="ew")
        unit_box = ttk.Combobox(
            controls,
            textvariable=self.unit_var,
            values=("Nilai k", "Persen"),
            width=12,
            state="readonly",
        )
        unit_box.grid(row=2, column=1, padx=(12, 0), sticky="w")
        unit_box.bind("<<ComboboxSelected>>", self._update_input_hint)

        color_box = ttk.Combobox(
            controls,
            textvariable=self.color_mode_var,
            values=("Grayscale", "RGB"),
            width=12,
            state="readonly",
        )
        color_box.grid(row=2, column=2, padx=(12, 0), sticky="w")
        color_box.bind("<<ComboboxSelected>>", self._on_color_mode_changed)

        self.compress_button = ttk.Button(controls, text="Proses", command=self.compress_image)
        self.compress_button.grid(row=2, column=3, padx=(18, 0), sticky="w")

        self.save_button = ttk.Button(controls, text="Simpan", command=self.save_photo, state="disabled")
        self.save_button.grid(row=2, column=4, padx=(8, 0), sticky="w")

        preview_area = ttk.Frame(self.root)
        preview_area.grid(row=2, column=0, padx=16, sticky="nsew")
        preview_area.columnconfigure(0, weight=1)
        preview_area.columnconfigure(1, weight=1)
        preview_area.rowconfigure(0, weight=1)

        original_panel = ttk.Frame(preview_area, style="Panel.TFrame", padding=10)
        original_panel.grid(row=0, column=0, padx=(0, 8), sticky="nsew")
        original_panel.columnconfigure(0, weight=1)
        original_panel.rowconfigure(1, weight=1)

        original_header = ttk.Frame(original_panel, style="Panel.TFrame")
        original_header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        original_header.columnconfigure(0, weight=1)
        ttk.Label(original_header, text="Citra Satelit Asli", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Button(original_header, text="Lihat", command=lambda: self._open_fullscreen("original")).grid(row=0, column=1)
        self.original_preview = ImagePreview(original_panel)
        self.original_preview.grid(row=1, column=0, sticky="nsew")

        result_panel = ttk.Frame(preview_area, style="Panel.TFrame", padding=10)
        result_panel.grid(row=0, column=1, padx=(8, 0), sticky="nsew")
        result_panel.columnconfigure(0, weight=1)
        result_panel.rowconfigure(1, weight=1)

        result_header = ttk.Frame(result_panel, style="Panel.TFrame")
        result_header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        result_header.columnconfigure(0, weight=1)
        ttk.Label(result_header, text="Hasil Pengolahan SVD", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        self.result_fullscreen_button = ttk.Button(
            result_header,
            text="Lihat",
            command=lambda: self._open_fullscreen("compressed"),
        )
        self.result_fullscreen_button.grid(row=0, column=1)

        self.result_canvas = tk.Canvas(result_panel, bg="#ffffff", highlightthickness=0)
        result_scrollbar = ttk.Scrollbar(result_panel, orient="vertical", command=self.result_canvas.yview)
        self.result_frame = ttk.Frame(self.result_canvas, style="Panel.TFrame")
        self.result_frame.bind(
            "<Configure>",
            lambda _event: self.result_canvas.configure(scrollregion=self.result_canvas.bbox("all")),
        )
        self.result_canvas_window = self.result_canvas.create_window((0, 0), window=self.result_frame, anchor="nw")
        self.result_canvas.configure(yscrollcommand=result_scrollbar.set)
        self.result_canvas.bind(
            "<Configure>",
            lambda event: self.result_canvas.itemconfigure(self.result_canvas_window, width=event.width),
        )
        self.result_canvas.grid(row=1, column=0, sticky="nsew")
        result_scrollbar.grid(row=1, column=1, sticky="ns")
        result_panel.rowconfigure(1, weight=1)
        self._clear_result_previews()

        stats = ttk.Frame(self.root, padding=(16, 10, 16, 14))
        stats.grid(row=3, column=0, sticky="ew")
        stats.columnconfigure(0, weight=1)
        ttk.Label(stats, text="Hasil Perhitungan", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(stats, textvariable=self.status_var, foreground="#4b5563").grid(row=1, column=0, sticky="w", pady=(2, 8))

        self.progress = ttk.Progressbar(stats, mode="indeterminate", length=160)
        self.progress.grid(row=0, column=1, rowspan=2, sticky="e")

        metrics = ttk.Frame(stats)
        metrics.grid(row=2, column=0, columnspan=2, sticky="ew")
        for col in range(4):
            metrics.columnconfigure(col, weight=1)

        for index, (label, variable) in enumerate(self.metric_vars.items()):
            item = ttk.Frame(metrics, style="Panel.TFrame", padding=(8, 6))
            item.grid(row=index // 4, column=index % 4, padx=3, pady=3, sticky="ew")
            ttk.Label(item, text=label, style="Muted.TLabel").pack(anchor="w")
            ttk.Label(item, textvariable=variable, style="Panel.TLabel", font=("Segoe UI", 10, "bold")).pack(anchor="w")

    def choose_image(self) -> None:
        filename = filedialog.askopenfilename(
            title="Pilih file citra satelit",
            filetypes=[
                ("Citra Satelit", "*.jpg *.jpeg *.png *.tif *.tiff"),
                ("TIFF", "*.tif *.tiff"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
            ],
        )
        if not filename:
            return

        self.image_path = Path(filename)
        self.path_var.set(str(self.image_path))
        self.status_var.set(f"Citra siap diproses. Format didukung: {SUPPORTED_FORMAT_TEXT}.")
        self.last_compressed_image = None
        self.last_compressed_k = None
        self.last_original_path = None
        self.last_mse = None
        self.compressed_images_by_k = {}
        self.save_button.configure(state="disabled")
        self.result_fullscreen_button.grid()
        self._show_original_preview()
        self._show_original_stats()

    def compress_image(self) -> None:
        if self.image_path is None:
            messagebox.showerror("Error", "Pilih citra satelit terlebih dahulu.")
            return
        if self.is_compressing:
            return

        self.is_compressing = True
        self.compress_button.configure(state="disabled")
        self.save_button.configure(state="disabled")
        self.progress.start(12)
        self.status_var.set("Memproses SVD, mohon tunggu...")
        self._reset_metrics()
        self._clear_result_previews("Sedang memproses...")

        threading.Thread(
            target=self._compress_worker,
            args=(
                self.image_path,
                self.k_var.get(),
                self.unit_var.get(),
                self.color_mode_var.get(),
            ),
            daemon=True,
        ).start()

    def _compress_worker(self, image_path: Path, raw_value: str, unit: str, color_mode: str) -> None:
        try:
            start_time = time.perf_counter()
            validate_image_path(image_path)
            matrix = load_image_matrix(image_path, color_mode)
            k_values = parse_compression_values(raw_value, min(matrix.shape[:2]), unit)
            compressed_images = {k: compress_matrix_with_rank_k(matrix, k) for k in k_values}
            mse_by_k = {
                k: float(np.mean((matrix - image_matrix) ** 2))
                for k, image_matrix in compressed_images.items()
            }
            selected_k = k_values[-1]
            process_time = time.perf_counter() - start_time
            self.worker_queue.put((
                "success",
                image_path,
                selected_k,
                compressed_images[selected_k],
                compressed_images,
                mse_by_k,
                matrix.shape,
                k_values,
                process_time,
                mse_by_k[selected_k],
            ))
        except (FileNotFoundError, ValueError, np.linalg.LinAlgError) as error:
            self.worker_queue.put(("error", str(error)))

    def _poll_worker_queue(self) -> None:
        try:
            while True:
                message = self.worker_queue.get_nowait()
                if message[0] == "success":
                    (
                        _,
                        image_path,
                        selected_k,
                        selected_image,
                        compressed_images,
                        mse_by_k,
                        matrix_shape,
                        k_values,
                        process_time,
                        mse,
                    ) = message
                    self._finish_compression(
                        image_path,
                        selected_k,
                        selected_image,
                        compressed_images,
                        mse_by_k,
                        matrix_shape,
                        k_values,
                        process_time,
                        mse,
                    )
                elif message[0] == "error":
                    self._show_compression_error(message[1])
        except queue.Empty:
            pass

        self.root.after(100, self._poll_worker_queue)

    def _finish_compression(
        self,
        image_path: Path,
        selected_k: int,
        selected_image: np.ndarray,
        compressed_images: dict[int, np.ndarray],
        mse_by_k: dict[int, float],
        matrix_shape: tuple[int, ...],
        k_values: list[int],
        process_time: float,
        mse: float,
    ) -> None:
        self.last_original_path = image_path
        self.last_compressed_k = selected_k
        self.last_compressed_image = selected_image
        self.last_matrix_shape = matrix_shape
        self.last_k_values = k_values
        self.last_process_time = process_time
        self.last_mse = mse
        self.compressed_images_by_k = compressed_images

        self._show_result_previews(compressed_images, mse_by_k)
        self._show_stats(image_path, matrix_shape, k_values, process_time, mse)

        if len(compressed_images) > 1:
            self.save_button.configure(state="disabled")
            self.result_fullscreen_button.grid_remove()
            self.status_var.set(f"Selesai. Menampilkan {len(compressed_images)} hasil.")
        else:
            self.save_button.configure(state="normal")
            self.result_fullscreen_button.grid()
            self.status_var.set(f"Selesai. Preview k = {selected_k}. Klik Simpan untuk menyimpan.")

        self._set_idle()

    def _show_compression_error(self, error_message: str) -> None:
        messagebox.showerror("Error", error_message)
        self.status_var.set("Kompresi gagal.")
        self._clear_result_previews()
        self._set_idle()

    def _set_idle(self) -> None:
        self.is_compressing = False
        self.progress.stop()
        self.compress_button.configure(state="normal")

    def save_photo(self) -> None:
        if self.last_compressed_image is None or self.last_compressed_k is None or self.last_original_path is None:
            messagebox.showerror("Error", "Belum ada hasil pengolahan SVD.")
            return

        save_path = filedialog.asksaveasfilename(
            title="Simpan citra hasil pengolahan SVD",
            defaultextension=".jpg",
            initialfile=f"citra_satelit_svd_k{self.last_compressed_k}.jpg",
            filetypes=[
                ("JPEG", "*.jpg"),
                ("JPEG", "*.jpeg"),
                ("PNG", "*.png"),
                ("TIFF", "*.tif"),
                ("TIFF", "*.tiff"),
            ],
        )
        if not save_path:
            return

        output_path = Path(save_path)
        self._save_image_matrix(self.last_compressed_image, output_path)
        self._show_stats(
            self.last_original_path,
            self.last_matrix_shape,
            self.last_k_values,
            self.last_process_time,
            self.last_mse,
            output_path,
        )
        self.status_var.set(f"Citra hasil SVD berhasil disimpan: {output_path.name}")

    def _save_compressed_k(self, k: int) -> None:
        image_matrix = self.compressed_images_by_k.get(k)
        if image_matrix is None:
            messagebox.showerror("Error", f"Hasil SVD k={k} tidak ditemukan.")
            return

        save_path = filedialog.asksaveasfilename(
            title=f"Simpan citra hasil SVD k={k}",
            defaultextension=".jpg",
            initialfile=f"citra_satelit_svd_k{k}.jpg",
            filetypes=[
                ("JPEG", "*.jpg"),
                ("JPEG", "*.jpeg"),
                ("PNG", "*.png"),
                ("TIFF", "*.tif"),
                ("TIFF", "*.tiff"),
            ],
        )
        if not save_path:
            return

        output_path = Path(save_path)
        self._save_image_matrix(image_matrix, output_path)
        self.status_var.set(f"Citra hasil SVD k={k} berhasil disimpan: {output_path.name}")

    def _save_image_matrix(self, image_matrix: np.ndarray, output_path: Path) -> None:
        output_image = Image.fromarray(image_matrix.astype(np.uint8))
        if output_path.suffix.lower() in {".jpg", ".jpeg"}:
            if output_image.mode != "RGB":
                output_image = output_image.convert("RGB")
            output_image.save(output_path, quality=85, optimize=True)
        else:
            output_image.save(output_path)

    def _show_original_preview(self) -> None:
        if self.image_path is None:
            return

        mode = "RGB" if self.color_mode_var.get() == "RGB" else "L"
        image = Image.open(self.image_path).convert(mode)
        self.original_pil = image
        self.original_preview.set_image(image)
        self.compressed_pil = None
        self.compressed_images_by_k = {}
        self._clear_result_previews()

    def _clear_result_previews(self, text: str = "Hasil akan muncul di sini") -> None:
        for widget in self.result_widgets:
            widget.destroy()
        self.result_widgets.clear()

        label = ttk.Label(self.result_frame, text=text, style="Muted.TLabel", anchor="center")
        label.grid(row=0, column=0, padx=20, pady=80, sticky="ew")
        self.result_frame.columnconfigure(0, weight=1)
        self.result_widgets.append(label)

    def _show_result_previews(self, compressed_images: dict[int, np.ndarray], mse_by_k: dict[int, float]) -> None:
        for widget in self.result_widgets:
            widget.destroy()
        self.result_widgets.clear()

        show_actions = len(compressed_images) > 1
        for index, (k, image_matrix) in enumerate(compressed_images.items()):
            image = Image.fromarray(image_matrix.astype(np.uint8))
            if k == self.last_compressed_k:
                self.compressed_pil = image

            _, _, ratio = storage_stats(self.last_matrix_shape, k)
            item = ttk.Frame(self.result_frame, style="Panel.TFrame", padding=8)
            item.grid(row=index // 2, column=index % 2, padx=5, pady=5, sticky="nsew")
            self.result_frame.columnconfigure(index % 2, weight=1)

            header = ttk.Frame(item, style="Panel.TFrame")
            header.pack(fill="x", pady=(0, 4))
            ttk.Label(header, text=f"SVD k={k}", style="Section.TLabel").pack(side="left")

            if show_actions:
                ttk.Button(
                    header,
                    text="Lihat",
                    command=lambda img=image.copy(), kk=k: self._open_image_fullscreen(img, f"SVD k={kk}"),
                ).pack(side="right", padx=(4, 0))
                ttk.Button(header, text="Simpan", command=lambda kk=k: self._save_compressed_k(kk)).pack(side="right")

            ttk.Label(
                item,
                text=self._result_summary_text(image_matrix, mse_by_k[k], ratio, show_actions),
                style="Muted.TLabel",
            ).pack(anchor="w")

            preview = ImagePreview(item, placeholder_text="")
            preview.pack(fill="both", expand=True, pady=(5, 0))
            preview.configure(height=190)
            preview.set_image(image)
            self.result_widgets.append(item)

    def _result_summary_text(
        self,
        image_matrix: np.ndarray,
        mse: float,
        ratio: float,
        show_file_size: bool,
    ) -> str:
        if show_file_size:
            return (
                f"Rasio {ratio:.1f}% | MSE {mse:.2f} | "
                f"Ukuran JPG {self._estimate_jpg_size(image_matrix)}"
            )

        rows, cols = self.last_matrix_shape[:2]
        return f"Rasio {ratio:.1f}% | MSE {mse:.2f} | {cols} x {rows}"

    def _estimate_jpg_size(self, image_matrix: np.ndarray) -> str:
        buffer = BytesIO()
        image = Image.fromarray(image_matrix.astype(np.uint8))
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffer, format="JPEG", quality=85, optimize=True)
        return format_file_size(buffer.tell())

    def _open_fullscreen(self, which: str) -> None:
        image = self.original_pil if which == "original" else self.compressed_pil
        title = "Citra Satelit Asli" if which == "original" else "Hasil Pengolahan SVD"
        if image is None:
            return
        self._open_image_fullscreen(image, title)

    def _open_image_fullscreen(self, image: Image.Image, title: str) -> None:
        top = tk.Toplevel(self.root)
        top.title(title)
        top.configure(bg="#f3f4f6")

        header = ttk.Frame(top, padding=(10, 8))
        header.pack(fill="x")
        ttk.Label(header, text=title, font=("Segoe UI", 11, "bold")).pack(side="left")
        ttk.Label(header, text="Scroll zoom, drag geser, Esc tutup", foreground="#4b5563").pack(side="left", padx=20)
        ttk.Button(header, text="Tutup", command=top.destroy).pack(side="right")

        viewer = ImagePreview(top, placeholder_text="")
        viewer.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        viewer.set_image(image)

        top.bind("<Escape>", lambda _event: top.destroy())
        top.after(100, lambda: (top.attributes("-fullscreen", True), top.lift(), top.focus_force()))

    def _show_original_stats(self) -> None:
        if self.image_path is None:
            return

        with Image.open(self.image_path) as image:
            cols, rows = image.size

        original_size = self.image_path.stat().st_size
        self.metric_vars["Resolusi"].set(f"{cols} x {rows}")
        self.metric_vars["Ukuran Asli"].set(format_file_size(original_size))
        self.metric_vars["Ukuran Kompres"].set("-")
        self.metric_vars["MSE"].set("-")
        self.metric_vars["Nilai k"].set("-")
        self.metric_vars["Rasio Data"].set("-")
        self.metric_vars["Waktu Proses"].set("-")

    def _show_stats(
        self,
        original_path: Path,
        matrix_shape: tuple[int, ...],
        k_values: list[int],
        process_time: float,
        mse: float | None,
        saved_path: Path | None = None,
    ) -> None:
        rows, cols = matrix_shape[:2]
        selected_k = k_values[-1]
        _, compressed_data, stored_ratio = storage_stats(matrix_shape, selected_k)
        original_size = original_path.stat().st_size

        self.metric_vars["Resolusi"].set(f"{cols} x {rows}")
        self.metric_vars["Ukuran Asli"].set(format_file_size(original_size))
        self.metric_vars["MSE"].set(f"{mse:.2f}" if mse is not None else "-")
        self.metric_vars["Nilai k"].set(", ".join(str(k) for k in k_values))
        self.metric_vars["Rasio Data"].set(f"{stored_ratio:.1f}% ({compressed_data:,} elemen)")
        self.metric_vars["Waktu Proses"].set(f"{process_time:.2f} detik")

        if saved_path and saved_path.exists():
            compressed_size = saved_path.stat().st_size
            file_ratio = (compressed_size / original_size) * 100 if original_size else 0
            self.metric_vars["Ukuran Kompres"].set(f"{format_file_size(compressed_size)} ({file_ratio:.1f}%)")
        else:
            self.metric_vars["Ukuran Kompres"].set("-")

    def _reset_metrics(self) -> None:
        for variable in self.metric_vars.values():
            variable.set("-")

    def _update_input_hint(self, _event=None) -> None:
        if self.unit_var.get() == "Persen":
            self.k_var.set("25")
            self.status_var.set("Mode persen aktif. Contoh input: 25 atau 10,25,50.")
        else:
            self.k_var.set("50")
            self.status_var.set("Mode nilai k aktif. Contoh input: 50 atau 10,30,50.")

    def _on_color_mode_changed(self, _event=None) -> None:
        self.last_compressed_image = None
        self.last_compressed_k = None
        self.last_mse = None
        self.compressed_images_by_k = {}
        self.save_button.configure(state="disabled")
        self.result_fullscreen_button.grid()

        if self.image_path is not None:
            self._show_original_preview()
            self._show_original_stats()
        else:
            self._reset_metrics()

        self.status_var.set(f"Kanal citra aktif: {self.color_mode_var.get()}.")


def run_gui() -> None:
    root = tk.Tk()
    SVDCompressorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
