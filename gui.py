"""Stage-by-stage viewer for the shape detection pipeline.

Shows the condition of the image at every step, from the original photo through
to the annotated result, with live controls for every pipeline parameter. Each
detected shape is listed with its type, centre point and the measured geometry
that produced the verdict, so nothing about the classification is hidden.

Run:  python gui.py [image]
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image

import pipeline as P
from classify import classify

THUMB_W = 250
STAGE_COLUMNS = 3

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


def to_pil(img: np.ndarray) -> Image.Image:
    """numpy (grayscale or BGR) -> PIL RGB."""
    if img.ndim == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def fit(size: tuple[int, int], width: int) -> tuple[int, int]:
    w, h = size
    return (width, max(1, round(h * width / w)))


class App(ctk.CTk):
    def __init__(self, image_path: str):
        super().__init__()
        self.title("Shape Detection - classical CV pipeline")
        self.geometry("1500x950")
        self.minsize(1100, 700)

        self.params = P.Params()
        self.image_path = image_path
        self.bgr = P.load_image(image_path)
        self.stages: list[tuple[str, np.ndarray]] = []
        self.shapes: list[P.Shape] = []
        self._images: list[ctk.CTkImage] = []  # keep refs alive
        self._pending: str | None = None

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_controls()
        self._build_stages()
        self._build_results()
        self.refresh()

    # ---------------------------------------------------------------- controls
    def _build_controls(self) -> None:
        panel = ctk.CTkScrollableFrame(self, width=290, label_text="Pipeline controls")
        panel.grid(row=0, column=0, sticky="nsw", padx=(8, 4), pady=8)
        self.controls = panel

        ctk.CTkLabel(panel, text="Segmentation mode",
                     font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(4, 2))
        self.mode_btn = ctk.CTkSegmentedButton(
            panel, values=["chroma", "gray"], command=self._set_mode)
        self.mode_btn.set(self.params.mode)
        self.mode_btn.pack(fill="x", pady=(0, 2))
        ctk.CTkLabel(panel, text="chroma: colour distance from the wood (works on\n"
                                 "this photo). gray: pure black-and-white route.",
                     justify="left", text_color="gray60",
                     font=ctk.CTkFont(size=10)).pack(anchor="w", pady=(0, 8))

        ctk.CTkLabel(panel, text="Threshold",
                     font=ctk.CTkFont(weight="bold")).pack(anchor="w")
        self.thresh_menu = ctk.CTkOptionMenu(
            panel, values=["otsu", "adaptive", "manual"], command=self._set_threshold)
        self.thresh_menu.set(self.params.threshold_mode)
        self.thresh_menu.pack(fill="x", pady=(0, 6))

        self._slider(panel, "Manual level", 0, 255, self.params.manual_thresh,
                     "manual_thresh", 255, "{:.0f}")
        self._slider(panel, "Adaptive block", 3, 201, self.params.adaptive_block,
                     "adaptive_block", 99, "{:.0f}")
        self._slider(panel, "Adaptive C", -20, 40, self.params.adaptive_c,
                     "adaptive_c", 60, "{:.0f}")

        ctk.CTkLabel(panel, text="Board isolation",
                     font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(10, 0))
        self._slider(panel, "White cut-off", 150, 254, self.params.white_cut,
                     "white_cut", 104, "{:.0f}")
        self._slider(panel, "Board erode", 1, 81, self.params.board_erode,
                     "board_erode", 40, "{:.0f}")

        ctk.CTkLabel(panel, text="Grayscale route",
                     font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(10, 0))
        self._slider(panel, "Flatten kernel (% width)", 2, 40,
                     self.params.flatten_frac * 100, "flatten_frac", 38,
                     "{:.0f}%", scale=0.01)
        self._slider(panel, "CLAHE clip", 0.5, 8.0, self.params.clahe_clip,
                     "clahe_clip", 30, "{:.1f}")
        self._slider(panel, "Bilateral d", 1, 25, self.params.bilateral_d,
                     "bilateral_d", 24, "{:.0f}")
        self._slider(panel, "Bilateral sigma", 5, 200, self.params.bilateral_sigma,
                     "bilateral_sigma", 39, "{:.0f}")

        ctk.CTkLabel(panel, text="Morphology",
                     font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(10, 0))
        self._slider(panel, "Open kernel", 1, 31, self.params.open_ksize,
                     "open_ksize", 15, "{:.0f}")
        self._slider(panel, "Close kernel", 1, 41, self.params.close_ksize,
                     "close_ksize", 20, "{:.0f}")

        ctk.CTkLabel(panel, text="Contour filter",
                     font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(10, 0))
        self._slider(panel, "Min area (% frame)", 0.05, 8.0,
                     self.params.min_area_frac * 100, "min_area_frac", 159,
                     "{:.2f}%", scale=0.01)
        self._slider(panel, "Max area (% frame)", 5, 95,
                     self.params.max_area_frac * 100, "max_area_frac", 90,
                     "{:.0f}%", scale=0.01)

        for text, cmd in (("Load image...", self.load_image),
                          ("Save annotated result", self.save_result),
                          ("Export results to CSV", self.export_csv),
                          ("Reset to defaults", self.reset)):
            ctk.CTkButton(panel, text=text, command=cmd).pack(fill="x", pady=(8, 0))

    def _slider(self, parent, label, frm, to, init, attr, steps, fmt, scale=1.0):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=(4, 0))
        caption = ctk.CTkLabel(row, text=f"{label}: {fmt.format(init)}",
                               font=ctk.CTkFont(size=11))
        caption.pack(anchor="w")

        def on_change(value: float) -> None:
            setattr(self.params, attr,
                    value * scale if scale != 1.0 else
                    (int(round(value)) if isinstance(getattr(self.params, attr), int)
                     else value))
            caption.configure(text=f"{label}: {fmt.format(value)}")
            self.schedule_refresh()

        slider = ctk.CTkSlider(row, from_=frm, to=to, number_of_steps=steps,
                               command=on_change)
        slider.set(init)
        slider.pack(fill="x")

    def _set_mode(self, value: str) -> None:
        self.params.mode = value
        self.schedule_refresh()

    def _set_threshold(self, value: str) -> None:
        self.params.threshold_mode = value
        self.schedule_refresh()

    # ------------------------------------------------------------------ layout
    def _build_stages(self) -> None:
        self.stage_frame = ctk.CTkScrollableFrame(
            self, label_text="Image condition at each stage  (click to enlarge)")
        self.stage_frame.grid(row=0, column=1, sticky="nsew", padx=4, pady=8)
        for c in range(STAGE_COLUMNS):
            self.stage_frame.grid_columnconfigure(c, weight=1)

    def _build_results(self) -> None:
        self.result_frame = ctk.CTkScrollableFrame(
            self, width=400, label_text="Detected shapes")
        self.result_frame.grid(row=0, column=2, sticky="nse", padx=(4, 8), pady=8)

    # ----------------------------------------------------------------- refresh
    def schedule_refresh(self) -> None:
        """Debounce: sliders fire rapidly, so coalesce into one recompute."""
        if self._pending is not None:
            self.after_cancel(self._pending)
        self._pending = self.after(120, self.refresh)

    def refresh(self) -> None:
        self._pending = None
        try:
            self.stages, self.shapes = P.run(self.bgr, self.params, classifier=classify)
        except Exception as exc:  # keep the UI alive on a bad parameter combination
            self.stage_frame.configure(label_text=f"Pipeline error: {exc}")
            return
        self._images.clear()
        self._render_stages()
        self._render_results()

    def _render_stages(self) -> None:
        for w in self.stage_frame.winfo_children():
            w.destroy()
        self.stage_frame.configure(
            label_text=f"Image condition at each stage  -  {len(self.shapes)} shapes "
                       f"detected  (click to enlarge)")
        for i, (name, img) in enumerate(self.stages):
            pil = to_pil(img)
            thumb = ctk.CTkImage(light_image=pil, dark_image=pil,
                                 size=fit(pil.size, THUMB_W))
            self._images.append(thumb)

            cell = ctk.CTkFrame(self.stage_frame)
            cell.grid(row=i // STAGE_COLUMNS, column=i % STAGE_COLUMNS,
                      padx=6, pady=6, sticky="n")
            ctk.CTkLabel(cell, text=name,
                         font=ctk.CTkFont(size=12, weight="bold")).pack(pady=(6, 2))
            label = ctk.CTkLabel(cell, image=thumb, text="")
            label.pack(padx=6, pady=(0, 6))
            label.bind("<Button-1>", lambda _e, n=name, im=img: self.enlarge(n, im))
            ctk.CTkLabel(cell, text=f"{img.shape[1]}x{img.shape[0]}"
                                    f"{'  grayscale' if img.ndim == 2 else '  colour'}",
                         font=ctk.CTkFont(size=10),
                         text_color="gray60").pack(pady=(0, 6))

    def _render_results(self) -> None:
        for w in self.result_frame.winfo_children():
            w.destroy()

        if not self.shapes:
            ctk.CTkLabel(self.result_frame,
                         text="No shapes detected.\nTry adjusting the threshold or\n"
                              "the contour area filter.",
                         justify="left", text_color="gray60").pack(pady=20)
            return

        for i, s in enumerate(self.shapes, 1):
            f, d = s.features, s.detail
            card = ctk.CTkFrame(self.result_frame)
            card.pack(fill="x", padx=6, pady=6)

            header = "unknown" if s.label == "unknown" else s.label.upper()
            colour = "gray60" if s.label == "unknown" else "#4da6ff"
            ctk.CTkLabel(card, text=f"{i}.  {header}",
                         font=ctk.CTkFont(size=16, weight="bold"),
                         text_color=colour).pack(anchor="w", padx=10, pady=(8, 0))
            ctk.CTkLabel(card, text=f"centre point:  ({s.centre[0]}, {s.centre[1]})",
                         font=ctk.CTkFont(size=13)).pack(anchor="w", padx=10)
            ctk.CTkLabel(card,
                         text=f"confidence {s.confidence:.0%}   "
                              f"evidence {d.get('checks_passed', '-')}",
                         font=ctk.CTkFont(size=11),
                         text_color="gray60").pack(anchor="w", padx=10)

            # the outline that was measured, rendered on its own
            crop = self._outline_crop(s)
            pil = to_pil(crop)
            img = ctk.CTkImage(light_image=pil, dark_image=pil, size=(96, 96))
            self._images.append(img)
            ctk.CTkLabel(card, image=img, text="").pack(anchor="w", padx=10, pady=4)

            rows = [
                ("vertices", f"{f['vertices']}"),
                ("solidity", f"{f['solidity']:.3f}"),
                ("circularity", f"{f['circularity']:.3f}"),
                ("aspect ratio", f"{f['aspect']:.2f}"),
                ("extent", f"{f['extent']:.3f}"),
                ("deep notches", f"{f['defects']}"),
                ("chord / perimeter", f"{f['straight_frac']:.3f}"),
                ("chord dominance", f"{f['chord_dominance']:.2f}"),
                ("area (px)", f"{f['area']:.0f}"),
            ]
            table = ctk.CTkFrame(card, fg_color="transparent")
            table.pack(fill="x", padx=10, pady=(2, 4))
            for r, (k, v) in enumerate(rows):
                ctk.CTkLabel(table, text=k, font=ctk.CTkFont(size=11),
                             text_color="gray60").grid(row=r, column=0, sticky="w")
                ctk.CTkLabel(table, text=v,
                             font=ctk.CTkFont(size=11)).grid(row=r, column=1,
                                                             sticky="e", padx=(14, 0))
            table.grid_columnconfigure(1, weight=1)

            ctk.CTkLabel(card, text=f"rule: {d.get('rule_reason', '')}",
                         font=ctk.CTkFont(size=10), text_color="gray55",
                         wraplength=340, justify="left").pack(anchor="w", padx=10)

            tpl = d.get("template_distances", {})
            if tpl:
                best = min(tpl.items(), key=lambda kv: kv[1])
                ctk.CTkLabel(card,
                             text=f"Hu-moment nearest: {best[0]} ({best[1]:.3f}) "
                                  f"- diagnostic only",
                             font=ctk.CTkFont(size=10), text_color="gray45",
                             wraplength=340, justify="left").pack(anchor="w", padx=10,
                                                                  pady=(0, 8))

    def _outline_crop(self, s: P.Shape, size: int = 200) -> np.ndarray:
        """Render one contour as a black outline on white, matching the
        reference-render style used by data/."""
        canvas = np.full((size, size), 255, np.uint8)
        c = s.contour.reshape(-1, 2).astype(np.float32)
        lo, hi = c.min(axis=0), c.max(axis=0)
        span = max((hi - lo).max(), 1.0)
        scale = (size * 0.8) / span
        pts = ((c - lo) * scale + (size - (hi - lo) * scale) / 2).astype(np.int32)
        cv2.polylines(canvas, [pts], True, 0, 2, cv2.LINE_AA)
        return canvas

    # ------------------------------------------------------------------ actions
    def enlarge(self, name: str, img: np.ndarray) -> None:
        win = ctk.CTkToplevel(self)
        win.title(name)
        pil = to_pil(img)
        width = min(1100, max(pil.size[0], 700))
        big = ctk.CTkImage(light_image=pil, dark_image=pil, size=fit(pil.size, width))
        self._images.append(big)
        ctk.CTkLabel(win, image=big, text="").pack(padx=10, pady=10)
        win.lift()
        win.focus()

    def load_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"),
                       ("All files", "*.*")])
        if not path:
            return
        try:
            self.bgr = P.load_image(path)
        except Exception as exc:
            messagebox.showerror("Could not open image", str(exc))
            return
        self.image_path = path
        self.title(f"Shape Detection - {Path(path).name}")
        self.refresh()

    def save_result(self) -> None:
        if not self.stages:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png", initialfile="result.png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if path:
            cv2.imwrite(path, self.stages[-1][1])
            messagebox.showinfo("Saved", f"Annotated result written to:\n{path}")

    def export_csv(self) -> None:
        if not self.shapes:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", initialfile="shapes.csv",
            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        keys = ("vertices", "solidity", "circularity", "aspect", "extent",
                "defects", "straight_frac", "chord_dominance", "area")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["index", "type", "centre_x", "centre_y", "confidence", *keys])
            for i, s in enumerate(self.shapes, 1):
                w.writerow([i, s.label, s.centre[0], s.centre[1],
                            f"{s.confidence:.3f}",
                            *[f"{s.features.get(k, '')}" for k in keys]])
        messagebox.showinfo("Exported", f"{len(self.shapes)} shapes written to:\n{path}")

    def reset(self) -> None:
        self.params = P.Params()
        for w in self.controls.winfo_children():
            w.destroy()
        self._build_controls()
        self.refresh()


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    selftest = "--selftest" in sys.argv
    path = args[0] if args else "test.webp"
    if not Path(path).exists():
        print(f"Image not found: {path}")
        raise SystemExit(1)

    app = App(path)
    if selftest:
        # Build, render one full pass, then quit - proves the UI constructs and
        # the pipeline renders without a human present.
        app.after(1200, lambda: (print(
            f"selftest OK: {len(app.stages)} stages, {len(app.shapes)} shapes -> "
            + ", ".join(f"{s.label}{s.centre}" for s in app.shapes)), app.destroy()))
    app.mainloop()


if __name__ == "__main__":
    main()
