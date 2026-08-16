"""Geometric shape identification from a contour.

Identification is done by measuring the contour and applying explicit rules, so
every verdict traces back to a number the GUI can show. Scope is the four shapes
on the test board: triangle, square, star and semicircle; anything else is
reported as ``unknown``.

The thresholds below are calibrated against the real silicone shapes rather than
textbook ideals, because the toy shapes are rounded and chunky. Measured values:

    shape      verts  solidity  circularity  aspect  extent  chord  notches
    arch         6      0.980      0.742      1.63    0.775   0.309    0
    square       4      0.993      0.850      1.01    0.949   0.205    0
    star         5      0.886      0.660      1.05    0.656   0.101    5
    triangle     3      0.977      0.682      1.10    0.624   0.272    0

Note the star's solidity is 0.886, nowhere near the ~0.55 of a mathematically
sharp star, because its arms are fat and rounded.

Hu-moment template matching (``cv2.matchShapes``) is computed and reported for
transparency, but is deliberately *not* allowed to assign a label: on these
rounded shapes it is actively misleading (the star's nearest template is
``square`` at distance 0.001, the arch's is ``star``). Gross Hu moments are
dominated by overall symmetry, which these shapes share. The measured geometry
separates them cleanly, so that is what decides.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

CLASSES = ("triangle", "square", "star", "semicircle")

# --- calibrated thresholds -------------------------------------------------
SOLIDITY_CONVEX_MIN = 0.93   # star 0.886 sits below, everything else >= 0.977
DEFECT_DEPTH_FRAC = 0.04     # notch depth floor, as a fraction of diameter
STAR_MIN_NOTCHES = 4         # star measures 5; all others measure 0
SEMI_ASPECT_MIN = 1.35       # arch 1.63; all others <= 1.10
SEMI_CHORD_MIN = 0.15        # arch's flat side is 0.31 of its perimeter
SEMI_CIRC_MIN = 0.62
# One dominant chord is what makes a semicircle a semicircle. Measured: arch
# 2.51 and an ideal half-disc 3.84, against 1.00-1.13 for every regular polygon
# (trapezoid is the closest competitor at 1.79). Without this, a pentagon is
# confidently mislabelled as a semicircle.
SEMI_DOMINANCE_MIN = 2.0
# Fraction of the minimum-area rectangle the shape fills. Exact for ideal
# shapes (triangle 0.50, square 1.00) and stable under rotation, which the
# vertex count is not.
# Measured: board triangle 0.627, reference triangle 0.498, tetrahedron 0.654,
# against pentagon 0.680 - so the ceiling stays below the pentagon.
TRI_FILL_MIN, TRI_FILL_MAX = 0.38, 0.66
# Measured: board square 0.959, cylinder 0.911, against dodecahedron 0.833.
SQUARE_FILL_MIN = 0.88


# --------------------------------------------------------------------------
# ideal templates (diagnostic only)
# --------------------------------------------------------------------------


def _canvas_contour(points: np.ndarray, size: int = 400) -> np.ndarray:
    img = np.zeros((size, size), np.uint8)
    cv2.fillPoly(img, [points.astype(np.int32)], 255)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea)


def _regular_polygon(n: int, size: int = 400, radius: float = 170.0,
                     rotation: float = -math.pi / 2) -> np.ndarray:
    c = size / 2
    return np.array([
        [c + radius * math.cos(rotation + 2 * math.pi * i / n),
         c + radius * math.sin(rotation + 2 * math.pi * i / n)]
        for i in range(n)
    ])


def _star(points: int = 5, ratio: float = 0.62, size: int = 400,
          radius: float = 170.0) -> np.ndarray:
    c = size / 2
    verts = []
    for i in range(points * 2):
        r = radius if i % 2 == 0 else radius * ratio
        a = -math.pi / 2 + math.pi * i / points
        verts.append([c + r * math.cos(a), c + r * math.sin(a)])
    return np.array(verts)


def _semicircle(size: int = 400, radius: float = 180.0) -> np.ndarray:
    c = size / 2
    cy = size / 2 + radius / 2
    return np.array([[c + radius * math.cos(a), cy - radius * math.sin(a)]
                     for a in np.linspace(0, math.pi, 72)])


TEMPLATES = {
    "triangle": _canvas_contour(_regular_polygon(3)),
    "square": _canvas_contour(_regular_polygon(4)),
    "star": _canvas_contour(_star()),
    "semicircle": _canvas_contour(_semicircle()),
}


# --------------------------------------------------------------------------
# geometric features
# --------------------------------------------------------------------------


def _defect_rows(defects) -> np.ndarray:
    """Normalise convexityDefects output across OpenCV versions.

    OpenCV 4 returns (N,1,4); OpenCV 5 returns (N,4). Handling both keeps this
    working on Windows and macOS regardless of which wheel pip resolves.
    """
    if defects is None:
        return np.empty((0, 4))
    a = np.asarray(defects)
    if a.ndim == 3 and a.shape[1] == 1:
        a = a[:, 0, :]
    if a.ndim != 2 or a.shape[-1] != 4:
        return np.empty((0, 4))
    return a


def stable_vertex_count(contour: np.ndarray) -> tuple[int, dict[float, int]]:
    """Vertex count that holds steady across a sweep of approxPolyDP epsilons.

    A single fixed epsilon is unreliable because the silicone shapes have rounded
    corners: the square reads as 8 vertices at eps=0.01 but 4 from eps=0.02
    upward. Taking the most frequent count across the sweep is robust to that.
    """
    perim = cv2.arcLength(contour, True)
    sweep = {eps: len(cv2.approxPolyDP(contour, eps * perim, True))
             for eps in (0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05)}
    tally: dict[int, int] = {}
    for n in sweep.values():
        tally[n] = tally.get(n, 0) + 1
    return max(tally.items(), key=lambda kv: (kv[1], -kv[0]))[0], sweep


def deep_defect_count(contour: np.ndarray,
                      min_depth_frac: float = DEFECT_DEPTH_FRAC) -> int:
    """Count convexity defects deeper than a fraction of the shape's diameter.

    The star has five genuine notches; contour noise produces many shallow ones,
    so the depth floor is what makes this discriminative (measured: star 5,
    every other shape 0).
    """
    if len(contour) < 4:
        return 0
    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) < 3:
        return 0
    try:
        rows = _defect_rows(cv2.convexityDefects(contour, hull))
    except cv2.error:
        return 0
    _, radius = cv2.minEnclosingCircle(contour)
    floor = max(3.0, min_depth_frac * radius * 2)
    return int(sum(1 for r in rows if r[3] / 256.0 > floor))  # depth is 1/256 px


def edge_lengths(contour: np.ndarray) -> tuple[float, list[float]]:
    """Perimeter and the straight-segment lengths, longest first."""
    perim = cv2.arcLength(contour, True)
    if perim <= 0:
        return 0.0, []
    approx = cv2.approxPolyDP(contour, 0.01 * perim, True).reshape(-1, 2)
    if len(approx) < 2:
        return perim, []
    lengths = [float(np.hypot(*(approx[(i + 1) % len(approx)] - approx[i])))
               for i in range(len(approx))]
    return perim, sorted(lengths, reverse=True)


def straight_edge_fraction(contour: np.ndarray) -> float:
    """Longest straight run along the outline, as a fraction of the perimeter.

    Distinguishes a semicircle (one long flat chord) from a full round shape.
    """
    perim, lengths = edge_lengths(contour)
    return lengths[0] / perim if lengths and perim > 0 else 0.0


def chord_dominance(contour: np.ndarray) -> float:
    """Longest straight edge divided by the second longest.

    A semicircle has exactly one long chord and short arc segments, so this runs
    high; a regular polygon has near-equal sides, so it sits at ~1. This is the
    feature that separates a semicircle from a pentagon, which otherwise share a
    similar aspect ratio and circularity.
    """
    _, lengths = edge_lengths(contour)
    if len(lengths) < 2:
        return 1.0
    return lengths[0] / max(lengths[1], 1e-6)


def features(contour: np.ndarray) -> dict:
    area = cv2.contourArea(contour)
    perim = cv2.arcLength(contour, True)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    long_side, short_side = max(rw, rh), max(min(rw, rh), 1e-6)
    _, _, bw, bh = cv2.boundingRect(contour)
    verts, sweep = stable_vertex_count(contour)

    # How much of the *minimum-area* rectangle the shape fills. Unlike `extent`,
    # which uses the axis-aligned box and therefore changes as a shape rotates,
    # this is rotation-invariant and near-constant per shape family:
    # any triangle fills exactly 0.5, a square 1.0, a half-disc 0.785.
    rect_fill = area / max(rw * rh, 1e-6)

    return {
        "area": area,
        "perimeter": perim,
        "vertices": verts,
        "vertex_sweep": sweep,
        "solidity": area / hull_area if hull_area > 0 else 0.0,
        "circularity": 4 * math.pi * area / (perim * perim) if perim > 0 else 0.0,
        "aspect": long_side / short_side,
        "extent": area / (bw * bh) if bw * bh > 0 else 0.0,
        "rect_fill": rect_fill,
        "defects": deep_defect_count(contour),
        "straight_frac": straight_edge_fraction(contour),
        "chord_dominance": chord_dominance(contour),
    }


# --------------------------------------------------------------------------
# identification
# --------------------------------------------------------------------------


def corroborating_checks(f: dict) -> dict[str, dict[str, bool]]:
    """Per-class supporting evidence, used for confidence and for the GUI."""
    sol, circ, asp = f["solidity"], f["circularity"], f["aspect"]
    ext, verts = f["extent"], f["vertices"]
    return {
        "star": {
            "has deep notches": f["defects"] >= STAR_MIN_NOTCHES,
            "concave (solidity < 0.93)": sol < SOLIDITY_CONVEX_MIN,
            "low extent": ext < 0.75,
            "low circularity": circ < 0.72,
            "5 vertices": verts == 5,
        },
        "semicircle": {
            "elongated (aspect >= 1.35)": asp >= SEMI_ASPECT_MIN,
            "long straight chord": f["straight_frac"] >= SEMI_CHORD_MIN,
            "one dominant chord": f["chord_dominance"] >= SEMI_DOMINANCE_MIN,
            "rounded": circ >= SEMI_CIRC_MIN,
            "convex": sol >= SOLIDITY_CONVEX_MIN,
        },
        "triangle": {
            "3 vertices": verts == 3,
            "fills ~0.5 of min-area rect": TRI_FILL_MIN <= f["rect_fill"] <= TRI_FILL_MAX,
            "convex": sol >= SOLIDITY_CONVEX_MIN,
            "not elongated": asp < SEMI_ASPECT_MIN,
            "no notches": f["defects"] < STAR_MIN_NOTCHES,
        },
        "square": {
            "4 vertices": verts == 4,
            "fills ~1.0 of min-area rect": f["rect_fill"] >= SQUARE_FILL_MIN,
            "convex": sol >= SOLIDITY_CONVEX_MIN,
            "aspect near 1": asp < SEMI_ASPECT_MIN,
            "high circularity": circ > 0.78,
        },
    }


def cascade(f: dict) -> tuple[str, str]:
    """Rule-based identification. Returns (label, the reason it fired).

    Ordered by how decisive each measurement is in the calibration data.
    """
    sol, circ, asp, ext = f["solidity"], f["circularity"], f["aspect"], f["extent"]
    verts, defects = f["vertices"], f["defects"]

    # 1. Concavity: the star is the only non-convex shape in scope.
    if defects >= STAR_MIN_NOTCHES or sol < SOLIDITY_CONVEX_MIN:
        return "star", f"concave: {defects} deep notches, solidity {sol:.3f}"

    # 2. Elongated and rounded with a single dominant flat side: the semicircle.
    #    The dominance test is what keeps a pentagon out of this branch.
    if (asp >= SEMI_ASPECT_MIN and f["straight_frac"] >= SEMI_CHORD_MIN
            and circ >= SEMI_CIRC_MIN
            and f["chord_dominance"] >= SEMI_DOMINANCE_MIN):
        return "semicircle", (f"aspect {asp:.2f}, one chord {f['straight_frac']:.2f} of "
                              f"perimeter dominating by {f['chord_dominance']:.1f}x")

    # 3/4. Triangle vs square, led by how much of the minimum-area rectangle the
    #      shape fills. That ratio is rotation-invariant and near-constant per
    #      family (triangle 0.5, square 1.0), whereas the vertex count jitters on
    #      these rounded shapes - the triangle reads 4 vertices in even light but
    #      5 under a gradient, so leading with vertices loses it.
    fill = f["rect_fill"]
    if asp < SEMI_ASPECT_MIN:
        if verts == 3 or TRI_FILL_MIN <= fill <= TRI_FILL_MAX:
            return "triangle", (f"fills {fill:.2f} of its min-area rect "
                                f"(~0.5 = triangle), {verts} vertices")
        if verts == 4 or fill >= SQUARE_FILL_MIN:
            return "square", (f"fills {fill:.2f} of its min-area rect "
                              f"(~1.0 = square), {verts} vertices")

    return "unknown", (f"no rule matched (verts {verts}, solidity {sol:.2f}, "
                       f"circ {circ:.2f}, aspect {asp:.2f})")


def template_match(contour: np.ndarray) -> dict[str, float]:
    """Hu-moment distance to each ideal template. Diagnostic only - see module
    docstring for why this does not decide the label."""
    return {name: float(cv2.matchShapes(contour, tpl, cv2.CONTOURS_MATCH_I1, 0.0))
            for name, tpl in TEMPLATES.items()}


def classify(shape) -> None:
    """Classify a pipeline.Shape in place, filling label/confidence/features."""
    f = features(shape.contour)
    label, reason = cascade(f)
    checks = corroborating_checks(f)
    distances = template_match(shape.contour)
    tpl_label = min(distances.items(), key=lambda kv: kv[1])[0]

    if label == "unknown":
        confidence = 0.0
        passed, total = 0, 0
    else:
        c = checks[label]
        passed, total = sum(c.values()), len(c)
        # Confidence reflects how much independent evidence backs the verdict.
        confidence = 0.55 + 0.44 * (passed / total)

    shape.label = label
    shape.confidence = float(confidence)
    shape.features = f
    shape.detail = {
        "rule_label": label,
        "rule_reason": reason,
        "checks": checks.get(label, {}),
        "checks_passed": f"{passed}/{total}" if total else "-",
        "all_checks": checks,
        "template_label": tpl_label,
        "template_distances": distances,
        "template_agrees": tpl_label == label,
    }


def main() -> None:
    """Regression check against the clean reference renders in data/.

    In-scope classes must come back with their own name. Out-of-scope classes
    (pentagon, circle, rectangle, ...) must come back as ``unknown`` - silently
    mislabelling them as one of the four would be a real failure.

    SILHOUETTE_ALIASES covers the classes whose outer outline genuinely *is* an
    in-scope shape: a tetrahedron drawn in wireframe has a triangular silhouette,
    a cube's and cylinder's are square-like, and data/rhombus/1.png has four
    equal sides at aspect 1.00, making it a square rotated 45 degrees. Since the
    pipeline classifies silhouettes, these answers are correct rather than
    regressions - distinguishing them needs interior line structure, which is
    out of scope.
    """
    SILHOUETTE_ALIASES = {
        "tetrahedron": "triangle",
        "cube": "square",
        "cylinder": "square",
        "rhombus": "square",
    }

    import sys
    from pathlib import Path

    from pipeline import Shape

    targets = sys.argv[1:] or ["data/triangle/1.png", "data/square/1.png"]
    print(f"{'file':<26}{'expected':<12}{'got':<12}{'conf':>6}  reason")
    print("-" * 100)
    ok = True
    for t in targets:
        path = Path(t)
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"{t:<26}  !! unreadable")
            ok = False
            continue
        cls = path.parent.name
        expected = cls if cls in CLASSES else SILHOUETTE_ALIASES.get(cls, "unknown")
        _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"{t:<26}  !! no contour")
            ok = False
            continue
        c = max(contours, key=cv2.contourArea)
        m = cv2.moments(c)
        s = Shape(contour=c,
                  centre=(int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])),
                  area=cv2.contourArea(c), area_frac=0.0, bbox=cv2.boundingRect(c))
        classify(s)
        match = s.label == expected
        ok &= match
        print(f"{t:<26}{expected:<12}{s.label:<12}{s.confidence:>6.2f}  "
              f"{s.detail['rule_reason']}{'' if match else '   <-- MISMATCH'}")
    print("\nPASS" if ok else "\nFAIL")


if __name__ == "__main__":
    main()
