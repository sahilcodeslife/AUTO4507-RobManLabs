"""Classical computer-vision pipeline for locating shapes and their centre points.

The pipeline is deliberately split into small stages that each return their
intermediate image, so the GUI can display the condition of the image at every
step and re-run from any stage when a parameter changes.

Two segmentation modes are available:

``chroma`` (default)
    Measures each pixel's distance from the *board's own* median colour in the
    CIELAB a*/b* chromaticity plane.  Chromaticity is largely independent of how
    brightly the scene is lit, and the wood reference is re-derived from every
    image, so the pipeline self-calibrates instead of relying on fixed values.

``gray``
    The pure black-and-white route: illumination flattening, CLAHE, bilateral
    filtering and Otsu.  Kept because it is the textbook approach, but on the
    supplied photo the shapes and the wood overlap in luminance (shapes measure
    108-200, wood 164-191), so no global grey threshold separates them.  See
    README notes; ``chroma`` is the mode that works on this image.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------


@dataclass
class Params:
    """Every tunable in the pipeline. The GUI binds sliders straight to these."""

    mode: str = "chroma"  # "chroma" | "gray"

    # board isolation
    board_mode: str = "saturation"  # "saturation" | "brightness"
    white_cut: int = 235  # brightness mode only: brighter than this is background
    board_erode: int = 25  # shrink the board mask to drop its edge and shadow

    # illumination flattening (gray mode)
    flatten_frac: float = 0.125  # blur kernel as a fraction of image width
    clahe_clip: float = 2.0

    # denoising
    bilateral_d: int = 9
    bilateral_sigma: int = 75

    # thresholding
    threshold_mode: str = "otsu"  # "otsu" | "adaptive" | "manual"
    manual_thresh: int = 90
    adaptive_block: int = 81
    adaptive_c: int = 8

    # morphology - the open kernel removes speckle, the close kernel seals
    # interiors. Keep close modest: an aggressive kernel fills the star's
    # concavities and destroys the very feature that identifies it.
    open_ksize: int = 7
    close_ksize: int = 11

    # contour filtering, as a fraction of total frame area
    min_area_frac: float = 0.005
    max_area_frac: float = 0.40
    border_margin: int = 2  # contours touching this close to the edge are dropped

    def odd(self, v: int) -> int:
        v = max(1, int(v))
        return v if v % 2 == 1 else v + 1


@dataclass
class Shape:
    """One detected shape: its outline, centre point and measured geometry."""

    contour: np.ndarray
    centre: tuple[int, int]
    area: float
    area_frac: float
    bbox: tuple[int, int, int, int]
    label: str = "unknown"
    confidence: float = 0.0
    features: dict = field(default_factory=dict)
    detail: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# individual stages
# --------------------------------------------------------------------------


def load_image(path: str | Path) -> np.ndarray:
    """Read any OpenCV-supported format (webp, png, jpg, ...) as BGR."""
    path = Path(path)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def convert_to_png(src: str | Path, dst: str | Path | None = None) -> Path:
    """Write a PNG copy next to the source (covers the format-conversion step)."""
    src = Path(src)
    dst = Path(dst) if dst else src.with_suffix(".png")
    cv2.imwrite(str(dst), load_image(src))
    return dst


def to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def flatten_illumination(gray: np.ndarray, p: Params) -> np.ndarray:
    """Divide out the illumination field to cancel brightness gradients.

    A heavy blur estimates how the scene is lit; dividing the original by that
    estimate leaves only local contrast, so a global threshold behaves the same
    under uneven or shifted lighting.
    """
    k = p.odd(int(gray.shape[1] * p.flatten_frac))
    background = cv2.GaussianBlur(gray, (k, k), 0)
    return cv2.divide(gray, background, scale=255)


def equalize(gray: np.ndarray, p: Params) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=p.clahe_clip, tileGridSize=(8, 8)).apply(gray)


def denoise(gray: np.ndarray, p: Params) -> np.ndarray:
    """Bilateral filter: smooths wood grain but keeps shape boundaries sharp."""
    return cv2.bilateralFilter(gray, p.bilateral_d, p.bilateral_sigma, p.bilateral_sigma)


def board_mask(bgr: np.ndarray, gray: np.ndarray, p: Params) -> np.ndarray:
    """Isolate the board so thresholding ignores the plain background.

    This matters more than it looks: the strongest contrast in the photo is
    background-versus-board, so an unrestricted Otsu locks onto that split and
    lumps the shapes in with the wood. Restricting the histogram to board pixels
    lets the threshold find the wood-versus-shape split instead.

    Saturation is the default cue rather than brightness, because S = (max-min)/max
    is invariant when every channel is scaled by the same factor - which is what
    dimming, brightening, a lighting gradient and a vignette all do. A fixed
    brightness cut-off breaks under all four: darken the photo and the white
    background falls below the cut-off, so the "board" swallows the whole frame.
    """
    h, w = gray.shape
    if p.board_mode == "brightness":
        fg = (gray < p.white_cut).astype(np.uint8) * 255
    else:
        sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
        # Otsu splits the unsaturated background from the coloured board.
        _, fg = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    fg = cv2.morphologyEx(
        fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    )
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((h, w), np.uint8)
    if not contours:
        mask[:] = 255  # no distinct board: treat the whole frame as valid
        return mask

    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < 0.10 * h * w:
        mask[:] = 255  # board fills frame or is undetectable
        return mask

    # Filling the outer contour reclaims the shapes, which sit inside the board
    # as holes in the saturation mask.
    cv2.drawContours(mask, [biggest], -1, 255, -1)
    k = p.odd(p.board_erode)
    return cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def _fit_surface(channel: np.ndarray, sample_mask: np.ndarray,
                 degree: int = 2, max_samples: int = 20000) -> np.ndarray:
    """Least-squares fit of a smooth 2D polynomial to `channel` over a mask.

    Used to model how the wood's colour drifts across the frame, so the
    reference tracks a lighting gradient instead of being a single number.
    """
    h, w = channel.shape
    ys, xs = np.nonzero(sample_mask)
    if len(xs) < 50:
        return np.full((h, w), float(np.median(channel)), np.float32)
    if len(xs) > max_samples:  # subsample for speed; the fit is over-determined
        pick = np.linspace(0, len(xs) - 1, max_samples).astype(np.int64)
        ys, xs = ys[pick], xs[pick]

    def terms(x, y):
        out = [np.ones_like(x)]
        for d in range(1, degree + 1):
            out += [x ** (d - k) * y ** k for k in range(d + 1)]
        return np.stack(out, axis=-1)

    xn, yn = (xs / max(w - 1, 1)) * 2 - 1, (ys / max(h - 1, 1)) * 2 - 1
    coeffs, *_ = np.linalg.lstsq(terms(xn, yn).astype(np.float64),
                                 channel[ys, xs].astype(np.float64), rcond=None)
    gx, gy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    return (terms(gx, gy) @ coeffs).astype(np.float32)


def chroma_distance(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-pixel colour distance from the board's own chromaticity.

    Works in CIELAB a*/b*, where colour is largely separated from brightness.
    Normalised rg-chromaticity was tried as a more strictly illumination-invariant
    alternative and is a poor fit for this palette: it divides out intensity, but
    brown wood and a yellow star share almost the same chromaticity and differ
    mainly in *lightness*, so the star disappears entirely. Lightness carries the
    signal here and cannot be discarded.

    Robustness to lighting instead comes from two estimation passes:

    1. A global median locates candidate shape pixels.
    2. Excluding those, a smooth surface is fitted to the wood's luminance, and
       the image is divided by it. This removes gradients and vignettes at the
       source, restoring an evenly-lit picture before any colour is measured.
    3. Smooth surfaces are then fitted to the wood's a*/b* on the balanced image,
       so any residual drift is tracked without the shapes dragging it around.
    """
    lab0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    inside = mask > 0
    if not inside.any():
        inside = np.ones(mask.shape, bool)

    # pass 1 - global reference, just to locate candidate shape pixels
    a0 = lab0[:, :, 1].astype(np.float32)
    b0 = lab0[:, :, 2].astype(np.float32)
    rough = np.sqrt((a0 - np.median(a0[inside])) ** 2 + (b0 - np.median(b0[inside])) ** 2)
    candidates = ((rough > np.percentile(rough[inside], 75)) & inside).astype(np.uint8) * 255
    candidates = cv2.dilate(
        candidates, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))

    wood = inside & (candidates == 0)
    if wood.sum() < 0.05 * inside.sum():
        wood = inside

    # pass 2 - flatten the illumination using the wood's luminance only
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    illum = np.clip(_fit_surface(gray, wood), 1.0, None)
    gain = float(np.median(gray[wood])) / illum
    balanced = np.clip(bgr.astype(np.float32) * gain[:, :, None], 0, 255).astype(np.uint8)

    # pass 3 - chroma distance on the evenly-lit image
    lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.float32)
    b = lab[:, :, 2].astype(np.float32)
    dist = np.sqrt((a - _fit_surface(a, wood)) ** 2 + (b - _fit_surface(b, wood)) ** 2)

    # Percentile scaling rather than min-max: a few outlier pixels should not
    # compress the whole range and move the threshold.
    hi = float(np.percentile(dist[inside], 99.5))
    return np.clip(dist / max(hi, 1e-6) * 255.0, 0, 255).astype(np.uint8)


def binarize(src: np.ndarray, mask: np.ndarray, p: Params, invert: bool) -> np.ndarray:
    """Threshold `src` using only pixels inside `mask` to pick the level."""
    inside = mask > 0
    if not inside.any():
        inside = np.ones(mask.shape, bool)

    if p.threshold_mode == "manual":
        level = float(p.manual_thresh)
    elif p.threshold_mode == "adaptive":
        flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        out = cv2.adaptiveThreshold(
            src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, flag,
            p.odd(p.adaptive_block), p.adaptive_c,
        )
        return cv2.bitwise_and(out, mask)
    else:  # otsu, computed over board pixels only
        level, _ = cv2.threshold(src[inside], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    hit = (src <= level) if invert else (src >= level)
    return ((hit & inside).astype(np.uint8)) * 255


def morphology(binary: np.ndarray, p: Params) -> np.ndarray:
    out = binary
    if p.open_ksize >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.odd(p.open_ksize),) * 2)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
    if p.close_ksize >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.odd(p.close_ksize),) * 2)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    return out


def find_shapes(binary: np.ndarray, p: Params) -> list[Shape]:
    """External contours, filtered to plausible shapes, with centroids.

    RETR_EXTERNAL is deliberate: the triangle and square are hollow frames, and
    external retrieval returns each one's outer silhouette as a single region
    rather than two nested rings. That silhouette is both the right thing to
    classify and the right region for the centre point.
    """
    h, w = binary.shape
    frame = float(h * w)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shapes: list[Shape] = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (p.min_area_frac * frame < area < p.max_area_frac * frame):
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        m = p.border_margin
        if x <= m or y <= m or x + bw >= w - m or y + bh >= h - m:
            continue  # touching the frame edge: the board or a crop artefact
        mom = cv2.moments(c)
        if mom["m00"] == 0:
            continue
        centre = (int(round(mom["m10"] / mom["m00"])), int(round(mom["m01"] / mom["m00"])))
        shapes.append(
            Shape(contour=c, centre=centre, area=area, area_frac=area / frame,
                  bbox=(x, y, bw, bh))
        )
    return sorted(shapes, key=lambda s: s.area, reverse=True)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

_PALETTE = [(60, 200, 255), (255, 160, 60), (120, 255, 120), (255, 120, 220),
            (120, 200, 255), (200, 120, 255)]


def draw_contours(bgr: np.ndarray, shapes: list[Shape]) -> np.ndarray:
    out = bgr.copy()
    for i, s in enumerate(shapes):
        cv2.drawContours(out, [s.contour], -1, _PALETTE[i % len(_PALETTE)], 3)
    return out


def annotate(bgr: np.ndarray, shapes: list[Shape]) -> np.ndarray:
    """Final output: outline, centre dot, and the shape's type."""
    out = bgr.copy()
    for i, s in enumerate(shapes):
        colour = _PALETTE[i % len(_PALETTE)]
        cv2.drawContours(out, [s.contour], -1, colour, 3)
        cx, cy = s.centre
        cv2.drawMarker(out, (cx, cy), (0, 0, 0), cv2.MARKER_CROSS, 18, 4)
        cv2.circle(out, (cx, cy), 7, (0, 0, 0), -1)
        cv2.circle(out, (cx, cy), 5, (255, 0, 255), -1)

        text = f"{s.label} ({cx},{cy})"
        x, y, bw, bh = s.bbox
        ty = max(18, y - 8)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        tx = min(max(0, x), out.shape[1] - tw - 2)
        cv2.rectangle(out, (tx - 2, ty - th - 4), (tx + tw + 2, ty + 4), (0, 0, 0), -1)
        cv2.putText(out, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------


def run(bgr: np.ndarray, p: Params | None = None, classifier=None):
    """Run every stage. Returns (list of (stage_name, image), shapes)."""
    p = p or Params()
    stages: list[tuple[str, np.ndarray]] = [("1. Original", bgr)]

    gray = to_gray(bgr)
    stages.append(("2. Grayscale", gray))

    flat = flatten_illumination(gray, p)
    stages.append(("3. Illumination flattened", flat))

    mask = board_mask(bgr, gray, p)
    stages.append(("4. Board mask", mask))

    if p.mode == "chroma":
        feature_img = chroma_distance(bgr, mask)
        stages.append(("5. Chroma distance", feature_img))
        invert = False  # shapes are FAR from the wood colour, so bright here
    else:
        eq = equalize(flat, p)
        stages.append(("5a. CLAHE", eq))
        feature_img = denoise(eq, p)
        stages.append(("5. Denoised", feature_img))
        invert = True  # shapes are darker than the flattened wood

    binary = binarize(feature_img, mask, p, invert)
    stages.append(("6. Binary", binary))

    cleaned = morphology(binary, p)
    stages.append(("7. Morphology", cleaned))

    shapes = find_shapes(cleaned, p)
    if classifier is not None:
        for s in shapes:
            classifier(s)

    stages.append(("8. Contours", draw_contours(bgr, shapes)))
    stages.append(("9. Result", annotate(bgr, shapes)))
    return stages, shapes


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Detect shapes and their centre points.")
    ap.add_argument("image", nargs="?", default="test.webp")
    ap.add_argument("--mode", choices=("chroma", "gray"), default="chroma")
    ap.add_argument("--save", default="result.png")
    args = ap.parse_args()

    from classify import classify

    bgr = load_image(args.image)
    stages, shapes = run(bgr, Params(mode=args.mode), classifier=classify)

    print(f"{args.image}: {bgr.shape[1]}x{bgr.shape[0]}, {len(shapes)} shapes\n")
    header = f"{'type':<12}{'centre':>12}{'conf':>7}{'verts':>7}{'solid':>8}{'circ':>7}{'aspect':>8}"
    print(header)
    print("-" * len(header))
    for s in shapes:
        f = s.features
        print(f"{s.label:<12}{str(s.centre):>12}{s.confidence:>7.2f}"
              f"{f.get('vertices', 0):>7}{f.get('solidity', 0):>8.3f}"
              f"{f.get('circularity', 0):>7.3f}{f.get('aspect', 0):>8.2f}")

    cv2.imwrite(args.save, stages[-1][1])
    print(f"\nAnnotated image -> {args.save}")


if __name__ == "__main__":
    main()
