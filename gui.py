"""Simple GUI for SVD-based satellite image processing using customtkinter."""
from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

from main import (
    compress_matrix_with_rank_k,
    file_size_stats,
    format_file_size,
    load_image_matrix,
    parse_compression_values,
    storage_stats,
    SUPPORTED_FORMAT_TEXT,
    validate_image_path,
)

COLORS = {
    "bg":           ("#f3f4f6", "#f3f4f6"),
    "card":         ("#ffffff", "#ffffff"),
    "card_hover":   ("#f9fafb", "#f9fafb"),
    "text":         ("#111827", "#111827"),
    "muted":        ("#4b5563", "#4b5563"),
    "border":       ("#d1d5db", "#d1d5db"),
    "accent":       ("#2563eb", "#2563eb"),
    "accent_hover": ("#1d4ed8", "#1d4ed8"),
    "accent_fg":    ("#ffffff", "#ffffff"),
    "entry_bg":     ("#ffffff", "#ffffff"),
    "entry_border": ("#9ca3af", "#9ca3af"),
    "preview_bg":   ("#e5e7eb", "#e5e7eb"),
    "green":        ("#15803d", "#15803d"),
}
C = COLORS


def _resolve_color(color):
    """Pick the right color from a (light, dark) tuple based on current theme."""
    if isinstance(color, tuple):
        return color[0] if ctk.get_appearance_mode() == "Light" else color[1]
    return color


class ZoomablePreview(ctk.CTkFrame):
    """Canvas-based image preview with scroll-to-zoom and drag-to-pan."""

    def __init__(self, master, placeholder_text: str = "", **kwargs):
        super().__init__(master, corner_radius=4, **kwargs)
        self._pil_image: Image.Image | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._scale = 1.0
        self._offset_x = 0.0
        self._offset_y = 0.0
        self._drag = {"x": 0, "y": 0}
        self._placeholder = placeholder_text
        self._fitted = True

        bg = _resolve_color(C["preview_bg"])
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True, padx=2, pady=2)

        self.canvas.bind("<MouseWheel>", self._on_scroll)
        self.canvas.bind("<Button-4>", self._on_scroll)
        self.canvas.bind("<Button-5>", self._on_scroll)
        self.canvas.bind("<Enter>", lambda _event: self.canvas.focus_set())
        self.canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<Double-Button-1>", self._on_dbl_click)
        self.canvas.bind("<Configure>", self._on_resize)

    # -- public API --
    def set_image(self, pil_image: Image.Image) -> None:
        self._pil_image = pil_image.copy()
        self._fitted = True
        self.after(10, lambda: (self._fit(), self._redraw()))

    def clear(self) -> None:
        self._pil_image = self._photo = None
        self._scale, self._offset_x, self._offset_y = 1.0, 0.0, 0.0
        self._fitted = True
        self.canvas.delete("all")
        self._show_placeholder()

    # -- internal --
    def _fit(self) -> None:
        if not self._pil_image:
            return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return
        iw, ih = self._pil_image.size
        self._scale = min(cw / iw, ch / ih) * 0.95
        self._offset_x = (cw - iw * self._scale) / 2
        self._offset_y = (ch - ih * self._scale) / 2

    def _redraw(self) -> None:
        self.canvas.delete("all")
        if not self._pil_image:
            self._show_placeholder()
            return
        iw, ih = self._pil_image.size
        dw, dh = max(1, int(iw * self._scale)), max(1, int(ih * self._scale))
        resized = self._pil_image.resize((dw, dh), Image.BILINEAR)
        self._photo = ImageTk.PhotoImage(resized)
        self.canvas.create_image(self._offset_x, self._offset_y,
                                 image=self._photo, anchor="nw")

    def _on_scroll(self, event) -> str:
        if not self._pil_image:
            return "break"
        delta = getattr(event, "delta", 0)
        is_zoom_in = delta > 0 or getattr(event, "num", None) == 4
        factor = 1.15 if is_zoom_in else 1 / 1.15
        ns = max(0.05, min(20.0, self._scale * factor))
        r = ns / self._scale
        self._offset_x = event.x - (event.x - self._offset_x) * r
        self._offset_y = event.y - (event.y - self._offset_y) * r
        self._scale = ns
        self._fitted = False
        self._redraw()
        return "break"

    def _on_drag_start(self, event) -> None:
        self._drag = {"x": event.x, "y": event.y}

    def _on_drag(self, event) -> None:
        if not self._pil_image:
            return
        self._offset_x += event.x - self._drag["x"]
        self._offset_y += event.y - self._drag["y"]
        self._drag = {"x": event.x, "y": event.y}
        self._fitted = False
        self._redraw()

    def _on_dbl_click(self, _event) -> None:
        if self._pil_image:
            self._fitted = True
            self._fit()
            self._redraw()

    def _on_resize(self, _event) -> None:
        self.canvas.configure(bg=_resolve_color(C["preview_bg"]))
        if not self._pil_image:
            self._show_placeholder()
            return
        if self._fitted:
            self._fit()
        self._redraw()

    def _show_placeholder(self) -> None:
        self.canvas.delete("all")
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw > 1 and ch > 1:
            self.canvas.create_text(cw / 2, ch / 2, text=self._placeholder,
                                    fill=_resolve_color(C["muted"]),
                                    font=("Segoe UI", 13))


class SVDCompressorGUI:
    def __init__(self, root: ctk.CTk) -> None:
        self.root = root
        self.root.title("Kompresi Citra Satelit SVD")
        self.root.geometry("1080x760")
        self.root.minsize(900, 650)
        self.root.configure(fg_color=C["bg"])

        self.image_path: Path | None = None
        self._original_pil: Image.Image | None = None
        self._compressed_pil: Image.Image | None = None
        self._compressed_preview_widgets: list[ctk.CTkFrame] = []
        self._compressed_images_by_k: dict[int, np.ndarray] = {}
        self.last_compressed_image: np.ndarray | None = None
        self.last_compressed_k: int | None = None
        self.last_original_path: Path | None = None
        self.last_result_text = ""
        self.is_compressing = False
        self.worker_queue: queue.Queue[tuple] = queue.Queue()
        self._last_matrix_shape: tuple = ()
        self._last_k_values: list[int] = []
        self._last_process_time: float = 0.0
        self._last_color_mode: str = "Grayscale"
        self._last_mse: float | None = None
        self._metric_widgets: list[dict] = []
        self._progress_running = False

        self.path_var = tk.StringVar(value="Belum ada citra satelit dipilih")
        self.k_var = tk.StringVar(value="50")
        self.unit_var = tk.StringVar(value="Nilai k")
        self.color_mode_var = tk.StringVar(value="Grayscale")
        self.status_var = tk.StringVar(value="Pilih citra, masukkan nilai k, lalu klik Proses.")

        self._build_layout()
        self._poll_worker_queue()

    def _build_layout(self) -> None:
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(2, weight=1)

        # -- header
        header = ctk.CTkFrame(self.root, fg_color="transparent")
        header.grid(row=0, column=0, padx=18, pady=(16, 6), sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header, text="Kompresi Citra Satelit dengan SVD",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color=C["text"],
        ).pack(anchor="w")

        ctk.CTkLabel(
            header,
            text="Program sederhana untuk membandingkan citra asli dan hasil kompresi.",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=C["muted"],
        ).pack(anchor="w", pady=(2, 0))

        # -- controls card
        controls = ctk.CTkFrame(
            self.root, corner_radius=4,
            fg_color=C["card"], border_color=C["border"], border_width=1,
        )
        controls.grid(row=1, column=0, padx=18, pady=(4, 10), sticky="ew")

        inner = ctk.CTkFrame(controls, fg_color="transparent")
        inner.pack(fill="x", padx=14, pady=12)
        inner.grid_columnconfigure(1, weight=1)

        # image picker row
        ctk.CTkButton(
            inner, text="Pilih Citra", width=135, height=32,
            corner_radius=4, fg_color=C["accent"], hover_color=C["accent_hover"],
            text_color=C["accent_fg"],
            font=ctk.CTkFont(family="Segoe UI", size=12),
            command=self.choose_image,
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            inner, textvariable=self.path_var,
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=C["muted"], anchor="w",
        ).grid(row=0, column=1, columnspan=5, padx=(10, 0), sticky="ew")

        # labels row
        for col, txt in [(0, "Nilai SVD"), (1, "Satuan"), (2, "Kanal Citra")]:
            ctk.CTkLabel(
                inner, text=txt,
                font=ctk.CTkFont(family="Segoe UI", size=11),
                text_color=C["muted"],
            ).grid(row=1, column=col, sticky="w", pady=(16, 0),
                   padx=(0 if col == 0 else 14, 0))

        # inputs row
        ctk.CTkEntry(
            inner, textvariable=self.k_var, width=150, height=32,
            corner_radius=4, border_color=C["entry_border"],
            fg_color=C["entry_bg"], text_color=C["text"],
            font=ctk.CTkFont(family="Segoe UI", size=13),
            placeholder_text="cth: 50 atau 10,30,50",
        ).grid(row=2, column=0, pady=(4, 0), sticky="ew")

        ctk.CTkOptionMenu(
            inner, variable=self.unit_var,
            values=["Nilai k", "Persen"], width=120, height=32,
            corner_radius=4, font=ctk.CTkFont(size=12),
            fg_color=C["entry_bg"], button_color=C["accent"],
            button_hover_color=C["accent_hover"], text_color=C["text"],
            dropdown_fg_color=C["card"], dropdown_hover_color=C["accent"],
            dropdown_text_color=C["text"],
            command=self._update_input_hint,
        ).grid(row=2, column=1, padx=(14, 0), pady=(4, 0), sticky="w")

        ctk.CTkOptionMenu(
            inner, variable=self.color_mode_var,
            values=["Grayscale", "RGB"], width=120, height=32,
            corner_radius=4, font=ctk.CTkFont(size=12),
            fg_color=C["entry_bg"], button_color=C["accent"],
            button_hover_color=C["accent_hover"], text_color=C["text"],
            dropdown_fg_color=C["card"], dropdown_hover_color=C["accent"],
            dropdown_text_color=C["text"],
            command=self._on_color_mode_changed,
        ).grid(row=2, column=2, padx=(14, 0), pady=(4, 0), sticky="w")

        self.compress_button = ctk.CTkButton(
            inner, text="Proses", width=105, height=32,
            corner_radius=4, fg_color=C["accent"], hover_color=C["accent_hover"],
            text_color=C["accent_fg"],
            font=ctk.CTkFont(family="Segoe UI", size=12),
            command=self.compress_image,
        )
        self.compress_button.grid(row=2, column=3, padx=(20, 0), pady=(4, 0))

        self.save_button = ctk.CTkButton(
            inner, text="Simpan", width=105, height=32,
            corner_radius=4, state="disabled",
            fg_color=C["border"], hover_color=C["card_hover"],
            text_color=C["text"],
            font=ctk.CTkFont(family="Segoe UI", size=12),
            command=self.save_photo,
        )
        self.save_button.grid(row=2, column=4, padx=(10, 0), pady=(4, 0))

        # -- preview area
        preview_area = ctk.CTkFrame(self.root, fg_color="transparent")
        preview_area.grid(row=2, column=0, padx=18, pady=(0, 8), sticky="nsew")
        preview_area.grid_columnconfigure(0, weight=1)
        preview_area.grid_columnconfigure(1, weight=1)
        preview_area.grid_rowconfigure(0, weight=1)

        for col, title, canvas_attr, which in [
            (0, "Citra Satelit Asli", "original_canvas", "original"),
            (1, "Hasil Pengolahan SVD", "compressed_canvas", "compressed"),
        ]:
            card = ctk.CTkFrame(
                preview_area, corner_radius=4,
                fg_color=C["card"], border_color=C["border"], border_width=1,
            )
            card.grid(row=0, column=col,
                      padx=(0, 8) if col == 0 else (8, 0), sticky="nsew")
            card.grid_columnconfigure(0, weight=1)
            card.grid_rowconfigure(1, weight=1)

            hdr = ctk.CTkFrame(card, fg_color="transparent")
            hdr.grid(row=0, column=0, padx=16, pady=(14, 0), sticky="ew")
            hdr.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(
                hdr, text=title,
                font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
                text_color=C["text"],
            ).grid(row=0, column=0, sticky="w")

            fullscreen_button = ctk.CTkButton(
                hdr, text="Lihat", width=52, height=26,
                corner_radius=4, fg_color=C["border"], hover_color=C["card_hover"],
                text_color=C["text"],
                font=ctk.CTkFont(size=11),
                command=lambda w=which: self._open_fullscreen(w),
            )
            fullscreen_button.grid(row=0, column=1, sticky="e")
            if col == 1:
                self.compressed_fullscreen_button = fullscreen_button

            if col == 0:
                canvas = ZoomablePreview(card, placeholder_text="Belum ada citra",
                                         fg_color=C["preview_bg"])
                canvas.grid(row=1, column=0, padx=14, pady=(10, 14), sticky="nsew")
                setattr(self, canvas_attr, canvas)
            else:
                self.compressed_results_frame = ctk.CTkScrollableFrame(
                    card, fg_color="transparent"
                )
                self.compressed_results_frame.grid(
                    row=1, column=0, padx=14, pady=(10, 14), sticky="nsew"
                )
                for grid_col in range(2):
                    self.compressed_results_frame.grid_columnconfigure(grid_col, weight=1)
                self._clear_compressed_previews()

        # -- stats section
        stats_outer = ctk.CTkFrame(self.root, fg_color="transparent")
        stats_outer.grid(row=3, column=0, padx=18, pady=(0, 16), sticky="ew")
        stats_outer.grid_columnconfigure(0, weight=1)

        stats_header = ctk.CTkFrame(stats_outer, fg_color="transparent")
        stats_header.pack(fill="x", pady=(0, 10))
        stats_header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            stats_header, text="Hasil Perhitungan",
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            text_color=C["text"],
        ).grid(row=0, column=0, sticky="w")

        self.status_label = ctk.CTkLabel(
            stats_header, textvariable=self.status_var,
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=C["muted"],
        )
        self.status_label.grid(row=1, column=0, sticky="w", pady=(2, 0))

        self.progress = ctk.CTkProgressBar(
            stats_header, width=140, height=6, corner_radius=2,
            progress_color=C["accent"], fg_color=C["border"],
        )
        self.progress.grid(row=0, column=1, rowspan=2, sticky="e", padx=(10, 0))
        self.progress.set(0)

        self.metrics_frame = ctk.CTkFrame(stats_outer, fg_color="transparent")
        self.metrics_frame.pack(fill="x")
        for i in range(4):
            self.metrics_frame.grid_columnconfigure(i, weight=1)

        self._build_empty_metrics()

    # -- metric tile system
    def _build_empty_metrics(self) -> None:
        self._clear_metrics()
        for i, (lbl, val) in enumerate([
            ("Resolusi", "- x -"), ("Ukuran Asli", "-"),
            ("Ukuran Kompres", "-"), ("MSE", "-"),
            ("Nilai k", "-"), ("Rasio Kompresi", "-"),
            ("Waktu Proses", "-"),
        ]):
            self._create_metric_tile(i, lbl, val)

    def _clear_metrics(self) -> None:
        for w in self._metric_widgets:
            w["frame"].destroy()
        self._metric_widgets.clear()

    def _create_metric_tile(self, col: int, label: str, value: str,
                            sub: str = "", ratio: float = -1) -> None:
        tile = ctk.CTkFrame(
            self.metrics_frame, corner_radius=4,
            fg_color=C["card"], border_color=C["border"], border_width=1,
        )
        tile.grid(row=col // 4, column=col % 4, padx=4, pady=3, sticky="nsew")

        lbl = ctk.CTkLabel(
            tile, text=label,
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=C["muted"],
        )
        lbl.pack(anchor="w", padx=10, pady=(8, 0))

        val = ctk.CTkLabel(
            tile, text=value,
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=C["text"],
        )
        val.pack(anchor="w", padx=10, pady=(0, 2))

        if sub:
            ctk.CTkLabel(
                tile, text=sub,
                font=ctk.CTkFont(family="Segoe UI", size=10),
                text_color=C["muted"],
            ).pack(anchor="w", padx=10, pady=(0, 8))
        else:
            ctk.CTkFrame(tile, height=8, fg_color="transparent").pack(pady=(0, 4))

        self._metric_widgets.append({"frame": tile, "bar": None})

    def _show_stats(self, original_path: Path, matrix_shape: tuple[int, ...],
                    k_values: list[int], process_time: float,
                    mse: float | None, saved_path: Path | None = None) -> None:
        self._clear_metrics()
        rows, cols = matrix_shape[:2]
        original_size = original_path.stat().st_size
        selected_k = k_values[-1]
        _, compressed_data, stored_ratio = storage_stats(matrix_shape, selected_k)

        self._create_metric_tile(0, "Resolusi", f"{cols} x {rows}", sub="piksel")
        self._create_metric_tile(1, "Ukuran Asli", format_file_size(original_size))

        if saved_path and saved_path.exists():
            csize = saved_path.stat().st_size
            fratio = (csize / original_size) * 100 if original_size else 0
            self._create_metric_tile(2, "Ukuran Kompres", format_file_size(csize),
                                     sub=f"{fratio:.1f}% dari asli", ratio=fratio)
        else:
            self._create_metric_tile(2, "Ukuran Kompres", "-", sub="simpan dulu")

        mse_display = f"{mse:.2f}" if mse is not None else "-"
        self._create_metric_tile(3, "MSE", mse_display, sub="galat rekonstruksi")

        k_display = ", ".join(str(k) for k in k_values)
        self._create_metric_tile(4, "Nilai k", k_display, sub=f"maks {min(rows, cols)}")
        self._create_metric_tile(5, "Rasio Data", f"{stored_ratio:.1f}%",
                                 sub=f"{compressed_data:,} elemen", ratio=stored_ratio)
        self._create_metric_tile(6, "Waktu Proses", f"{process_time:.2f}s", sub="detik")

    def _show_original_stats(self) -> None:
        if self.image_path is None:
            return

        with Image.open(self.image_path) as image:
            cols, rows = image.size

        color_mode = self.color_mode_var.get()
        channels = 3 if color_mode == "RGB" else 1
        original_size = self.image_path.stat().st_size

        self._clear_metrics()
        self._create_metric_tile(0, "Resolusi", f"{cols} x {rows}", sub="piksel")
        self._create_metric_tile(1, "Ukuran Asli", format_file_size(original_size))
        self._create_metric_tile(2, "Ukuran Kompres", "-", sub="simpan dulu")
        self._create_metric_tile(3, "MSE", "-", sub="setelah proses SVD")
        self._create_metric_tile(4, "Nilai k", "-", sub=f"maks {min(rows, cols)}")
        self._create_metric_tile(5, "Rasio Data", "-")
        self._create_metric_tile(6, "Waktu Proses", "-")

    # -- image selection
    def choose_image(self) -> None:
        filename = filedialog.askopenfilename(
            title="Pilih file citra satelit",
            filetypes=[("Citra Satelit", "*.jpg *.jpeg *.png *.tif *.tiff"),
                       ("TIFF", "*.tif *.tiff"),
                       ("JPEG", "*.jpg *.jpeg"), ("PNG", "*.png")],
        )
        if not filename:
            return
        self.image_path = Path(filename)
        self.path_var.set(str(self.image_path))
        self.status_var.set(f"Citra satelit siap diproses. Format didukung: {SUPPORTED_FORMAT_TEXT}.")
        self.last_compressed_image = None
        self.last_compressed_k = None
        self.last_original_path = None
        self.last_result_text = ""
        self._last_mse = None
        self._compressed_images_by_k = {}
        self.save_button.configure(state="disabled", text="Simpan")
        if hasattr(self, "compressed_fullscreen_button"):
            self.compressed_fullscreen_button.grid()
        self._show_original_preview()
        self._show_original_stats()

    # -- compression
    def compress_image(self) -> None:
        if self.image_path is None:
            messagebox.showerror("Error", "Pilih citra satelit terlebih dahulu.")
            return
        if self.is_compressing:
            return
        image_path = self.image_path
        raw_value = self.k_var.get()
        unit = self.unit_var.get()
        color_mode = self.color_mode_var.get()
        self.is_compressing = True
        self.compress_button.configure(state="disabled")
        self.save_button.configure(state="disabled", text="Simpan")
        self._start_progress()
        self.status_var.set("Memproses SVD citra satelit di background, mohon tunggu...")
        self._build_empty_metrics()
        threading.Thread(
            target=self._compress_worker,
            args=(image_path, raw_value, unit, color_mode), daemon=True,
        ).start()

    def _compress_worker(self, image_path: Path, raw_value: str,
                         unit: str, color_mode: str) -> None:
        try:
            t0 = time.perf_counter()
            validate_image_path(image_path)
            matrix = load_image_matrix(image_path, color_mode)
            k_values = parse_compression_values(raw_value, min(matrix.shape[:2]), unit)
            compressed_images = {k: compress_matrix_with_rank_k(matrix, k) for k in k_values}
            mse_by_k = {
                k: float(np.mean((matrix - image_matrix) ** 2))
                for k, image_matrix in compressed_images.items()
            }
            selected_k = k_values[-1]
            selected_image = compressed_images[selected_k]
            mse = mse_by_k[selected_k]
            ptime = time.perf_counter() - t0
            self.worker_queue.put((
                "success", image_path, selected_k,
                selected_image, compressed_images, mse_by_k,
                matrix.shape, k_values, ptime, color_mode, mse,
            ))
        except (FileNotFoundError, ValueError, np.linalg.LinAlgError) as error:
            self.worker_queue.put(("error", str(error)))

    def _poll_worker_queue(self) -> None:
        try:
            while True:
                msg = self.worker_queue.get_nowait()
                if msg[0] == "success":
                    (_, img_path, sel_k, sel_img, all_images, mse_by_k,
                     m_shape, k_vals, p_time, c_mode, mse) = msg
                    self._last_matrix_shape = m_shape
                    self._last_k_values = k_vals
                    self._last_process_time = p_time
                    self._last_color_mode = c_mode
                    self._last_mse = mse
                    self._finish_compression(img_path, sel_k, sel_img,
                                             all_images, mse_by_k)
                elif msg[0] == "error":
                    self._show_compression_error(msg[1])
        except queue.Empty:
            pass
        self.root.after(100, self._poll_worker_queue)

    def _finish_compression(self, image_path: Path, selected_k: int,
                            selected_image: np.ndarray,
                            compressed_images: dict[int, np.ndarray],
                            mse_by_k: dict[int, float]) -> None:
        self.last_original_path = image_path
        self.last_compressed_k = selected_k
        self.last_compressed_image = selected_image
        self._compressed_images_by_k = compressed_images
        self._show_compressed_previews(compressed_images, mse_by_k)
        self._show_stats(image_path, self._last_matrix_shape,
                         self._last_k_values, self._last_process_time,
                         self._last_mse)
        if len(compressed_images) > 1:
            self.save_button.configure(state="disabled", text="Simpan via kartu")
            if hasattr(self, "compressed_fullscreen_button"):
                self.compressed_fullscreen_button.grid_remove()
        else:
            self.save_button.configure(state="normal", text="Simpan")
            if hasattr(self, "compressed_fullscreen_button"):
                self.compressed_fullscreen_button.grid()
        if len(compressed_images) > 1:
            self.status_var.set(
                f"Selesai. Menampilkan {len(compressed_images)} preview. Gunakan tombol Simpan pada masing-masing kartu.")
        else:
            self.status_var.set(
                f"Selesai. Preview k = {selected_k}. Klik Simpan untuk menyimpan.")
        self._set_compression_idle()

    def _show_compression_error(self, error_message: str) -> None:
        messagebox.showerror("Error", error_message)
        self.status_var.set("Kompresi gagal.")
        self._set_compression_idle()

    def _set_compression_idle(self) -> None:
        self.is_compressing = False
        self._stop_progress()
        self.compress_button.configure(state="normal")

    # -- progress
    def _start_progress(self) -> None:
        self._progress_running = True
        self._animate_progress()

    def _animate_progress(self) -> None:
        if not self._progress_running:
            return
        cur = self.progress.get()
        nxt = cur + 0.02
        if nxt > 1.0:
            nxt = 0.0
        self.progress.set(nxt)
        self.root.after(50, self._animate_progress)

    def _stop_progress(self) -> None:
        self._progress_running = False
        self.progress.set(0)

    # -- save
    def save_photo(self) -> None:
        if (self.last_compressed_image is None
                or self.last_compressed_k is None
                or self.last_original_path is None):
            messagebox.showerror("Error", "Belum ada hasil pengolahan SVD.")
            return
        save_path = filedialog.asksaveasfilename(
            title="Simpan citra hasil pengolahan SVD",
            defaultextension=".tif",
            initialfile=f"citra_satelit_svd_k{self.last_compressed_k}.tif",
            filetypes=[("TIFF", "*.tif"), ("TIFF", "*.tiff"),
                       ("PNG", "*.png"), ("JPEG", "*.jpg"), ("JPEG", "*.jpeg")],
        )
        if not save_path:
            return
        output_path = Path(save_path)
        output_image = Image.fromarray(self.last_compressed_image.astype(np.uint8))
        if output_path.suffix.lower() in {".jpg", ".jpeg"}:
            if output_image.mode != "RGB":
                output_image = output_image.convert("RGB")
            output_image.save(output_path, quality=85, optimize=True)
        else:
            output_image.save(output_path)
        self._show_stats(self.last_original_path, self._last_matrix_shape,
                         self._last_k_values, self._last_process_time,
                         self._last_mse, saved_path=output_path)
        self.status_var.set(f"Citra hasil SVD berhasil disimpan: {output_path.name}")

    def _save_compressed_k(self, k: int) -> None:
        image_matrix = self._compressed_images_by_k.get(k)
        if image_matrix is None:
            messagebox.showerror("Error", f"Hasil SVD k={k} tidak ditemukan.")
            return

        save_path = filedialog.asksaveasfilename(
            title=f"Simpan citra hasil SVD k={k}",
            defaultextension=".tif",
            initialfile=f"citra_satelit_svd_k{k}.tif",
            filetypes=[("TIFF", "*.tif"), ("TIFF", "*.tiff"),
                       ("PNG", "*.png"), ("JPEG", "*.jpg"), ("JPEG", "*.jpeg")],
        )
        if not save_path:
            return

        output_path = Path(save_path)
        output_image = Image.fromarray(image_matrix.astype(np.uint8))
        if output_path.suffix.lower() in {".jpg", ".jpeg"}:
            if output_image.mode != "RGB":
                output_image = output_image.convert("RGB")
            output_image.save(output_path, quality=85, optimize=True)
        else:
            output_image.save(output_path)

        self.status_var.set(f"Citra hasil SVD k={k} berhasil disimpan: {output_path.name}")

    # -- preview
    def _show_original_preview(self) -> None:
        if self.image_path is None:
            return
        mode = "RGB" if self.color_mode_var.get() == "RGB" else "L"
        image = Image.open(self.image_path).convert(mode)
        self._original_pil = image
        self.original_canvas.set_image(image)
        self._compressed_pil = None
        self._compressed_images_by_k = {}
        self._clear_compressed_previews()

    def _show_compressed_preview(self, image_matrix: np.ndarray) -> None:
        image = Image.fromarray(image_matrix.astype(np.uint8))
        self._compressed_pil = image
        self._clear_compressed_previews()
        self._add_compressed_preview_card(0, 50, image, None, show_actions=False)

    def _clear_compressed_previews(self) -> None:
        for widget in self._compressed_preview_widgets:
            widget.destroy()
        self._compressed_preview_widgets.clear()

        placeholder = ctk.CTkLabel(
            self.compressed_results_frame,
            text="Hasil akan muncul di sini",
            font=ctk.CTkFont(size=13),
            text_color=C["muted"],
        )
        placeholder.grid(row=0, column=0, columnspan=2, sticky="nsew", pady=80)
        self._compressed_preview_widgets.append(placeholder)

    def _show_compressed_previews(self, compressed_images: dict[int, np.ndarray],
                                  mse_by_k: dict[int, float]) -> None:
        for widget in self._compressed_preview_widgets:
            widget.destroy()
        self._compressed_preview_widgets.clear()

        rows, cols = self._last_matrix_shape[:2]
        show_actions = len(compressed_images) > 1
        for index, (k, image_matrix) in enumerate(compressed_images.items()):
            image = Image.fromarray(image_matrix.astype(np.uint8))
            if k == self.last_compressed_k:
                self._compressed_pil = image
            _, _, ratio = storage_stats(self._last_matrix_shape, k)
            subtitle = f"Rasio {ratio:.1f}% | MSE {mse_by_k[k]:.2f} | {cols} x {rows}"
            self._add_compressed_preview_card(index, k, image, subtitle, show_actions)

    def _add_compressed_preview_card(self, index: int, k: int, image: Image.Image,
                                     subtitle: str | None,
                                     show_actions: bool = False) -> None:
        card = ctk.CTkFrame(
            self.compressed_results_frame, corner_radius=4,
            fg_color=C["card"], border_color=C["border"], border_width=1,
        )
        grid_row = index // 2
        grid_col = index % 2
        card.grid(row=grid_row, column=grid_col, padx=6, pady=6, sticky="nsew")
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(2, weight=1)

        header = ctk.CTkFrame(card, fg_color="transparent")
        header.grid(row=0, column=0, padx=10, pady=(8, 0), sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header, text=f"SVD k={k}",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=C["text"],
        ).grid(row=0, column=0, sticky="w")

        if show_actions:
            ctk.CTkButton(
                header, text="Full", width=42, height=26,
                corner_radius=4, fg_color=C["border"], hover_color=C["card_hover"],
                text_color=C["text"],
                font=ctk.CTkFont(family="Segoe UI", size=11),
                command=lambda img=image.copy(), kk=k: self._open_image_fullscreen(
                    img, f"SVD k={kk}"
                ),
            ).grid(row=0, column=1, padx=(6, 0), sticky="e")

            ctk.CTkButton(
                header, text="Simpan", width=70, height=26,
                corner_radius=4, fg_color=C["accent"], hover_color=C["accent_hover"],
                text_color=C["accent_fg"],
                font=ctk.CTkFont(family="Segoe UI", size=11),
                command=lambda kk=k: self._save_compressed_k(kk),
            ).grid(row=0, column=2, padx=(6, 0), sticky="e")

        if subtitle:
            ctk.CTkLabel(
                card, text=subtitle,
                font=ctk.CTkFont(family="Segoe UI", size=10),
                text_color=C["muted"],
            ).grid(row=1, column=0, padx=10, pady=(2, 4), sticky="w")

        preview = ZoomablePreview(card, placeholder_text="", fg_color=C["preview_bg"])
        preview.grid(row=2, column=0, padx=8, pady=(4, 8), sticky="nsew")
        preview.configure(height=190)
        preview.set_image(image)
        self._compressed_preview_widgets.append(card)

    # -- fullscreen
    def _open_fullscreen(self, which: str) -> None:
        pil_img = self._original_pil if which == "original" else self._compressed_pil
        title = "Citra Satelit Asli" if which == "original" else "Hasil Pengolahan SVD"
        if pil_img is None:
            return
        self._open_image_fullscreen(pil_img, title)

    def _open_image_fullscreen(self, pil_img: Image.Image, title: str) -> None:
        top = ctk.CTkToplevel(self.root)
        top.title(title)
        top.configure(fg_color="#f3f4f6")

        bar = ctk.CTkFrame(top, fg_color="#ffffff", height=46)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        ctk.CTkLabel(
            bar, text=f"  {title}",
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            text_color="#111827",
        ).pack(side="left", padx=10)

        ctk.CTkLabel(
            bar, text="Scroll zoom, drag geser, Esc tutup",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color="#4b5563",
        ).pack(side="left", padx=20)

        ctk.CTkButton(
            bar, text="Tutup", width=80, height=30,
            corner_radius=4, fg_color="#d1d5db", hover_color="#e5e7eb",
            text_color="#111827",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            command=top.destroy,
        ).pack(side="right", padx=16, pady=8)

        viewer = ZoomablePreview(top, fg_color="#e5e7eb")
        viewer.pack(fill="both", expand=True, padx=8, pady=(4, 8))
        viewer.set_image(pil_img)

        top.bind("<Escape>", lambda e: top.destroy())
        # Delay fullscreen + lift to work around CTkToplevel z-order bug
        top.after(100, lambda: (
            top.attributes("-fullscreen", True),
            top.lift(),
            top.focus_force(),
        ))

    def _update_input_hint(self, value: str | None = None) -> None:
        if self.unit_var.get() == "Persen":
            self.k_var.set("25")
            self.status_var.set("Mode persen aktif. Contoh input: 25 atau 10,25,50.")
        else:
            self.k_var.set("50")
            self.status_var.set("Mode nilai k aktif. Contoh input: 50 atau 10,30,50.")

    def _on_color_mode_changed(self, value: str | None = None) -> None:
        self.last_compressed_image = None
        self.last_compressed_k = None
        self._last_mse = None
        self._compressed_images_by_k = {}
        self.save_button.configure(state="disabled", text="Simpan")
        if hasattr(self, "compressed_fullscreen_button"):
            self.compressed_fullscreen_button.grid()
        if self.image_path is not None:
            self._show_original_preview()
            self._show_original_stats()
        else:
            self._build_empty_metrics()
        self.status_var.set(f"Kanal citra aktif: {self.color_mode_var.get()}.")


def run_gui() -> None:
    ctk.set_appearance_mode("Light")
    root = ctk.CTk()
    SVDCompressorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
