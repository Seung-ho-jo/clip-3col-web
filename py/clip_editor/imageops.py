"""Low-level image helpers: loading, binarization, projection runs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Rect:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    def as_list(self) -> list[int]:
        return [int(self.x0), int(self.y0), int(self.x1), int(self.y1)]

    @classmethod
    def of(cls, v) -> "Rect":
        x0, y0, x1, y1 = (int(round(t)) for t in v)
        return cls(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    def overlaps_y(self, other: "Rect") -> int:
        return max(0, min(self.y1, other.y1) - max(self.y0, other.y0))

    def overlaps_x(self, other: "Rect") -> int:
        return max(0, min(self.x1, other.x1) - max(self.x0, other.x0))


class Page:
    """Original page image plus its ink mask."""

    def __init__(self, path: str | None = None, *, rgb: np.ndarray | None = None, threshold: int | None = None):
        img = Image.open(path) if path is not None else Image.fromarray(rgb)
        img.load()
        self.rgb = np.asarray(img.convert("RGB")).copy()
        self.gray = np.asarray(img.convert("L")).astype(np.uint8)
        self.threshold = threshold if threshold is not None else otsu(self.gray)
        self.ink = self.gray < self.threshold
        bright = self.gray[self.gray >= self.threshold]
        self.paper = int(np.median(bright)) if bright.size else 255
        paper_rgb = self.rgb[self.gray >= self.threshold]
        self.paper_rgb = (
            tuple(int(v) for v in np.median(paper_rgb, axis=0)) if paper_rgb.size else (255, 255, 255)
        )

    def masked_copy(self, mask: np.ndarray) -> "Page":
        """Copy of the page with ``mask`` pixels painted paper (e.g. images removed from text)."""
        rgb = self.rgb.copy()
        rgb[mask] = self.paper_rgb
        p = Page(rgb=rgb, threshold=self.threshold)
        p.paper, p.paper_rgb = self.paper, self.paper_rgb
        return p

    @property
    def h(self) -> int:
        return self.gray.shape[0]

    @property
    def w(self) -> int:
        return self.gray.shape[1]

    def clip(self, r: Rect) -> Rect:
        return Rect(max(0, r.x0), max(0, r.y0), min(self.w, r.x1), min(self.h, r.y1))

    def crop_rgb(self, r: Rect) -> np.ndarray:
        """Crop that may extend beyond the page; outside area is paper colour."""
        out = np.empty((max(r.h, 0), max(r.w, 0), 3), dtype=np.uint8)
        out[:] = self.paper_rgb
        c = self.clip(r)
        if c.w > 0 and c.h > 0:
            out[c.y0 - r.y0 : c.y1 - r.y0, c.x0 - r.x0 : c.x1 - r.x0] = self.rgb[c.y0 : c.y1, c.x0 : c.x1]
        return out

    def crop_ink(self, r: Rect) -> np.ndarray:
        out = np.zeros((max(r.h, 0), max(r.w, 0)), dtype=bool)
        c = self.clip(r)
        if c.w > 0 and c.h > 0:
            out[c.y0 - r.y0 : c.y1 - r.y0, c.x0 - r.x0 : c.x1 - r.x0] = self.ink[c.y0 : c.y1, c.x0 : c.x1]
        return out


def otsu(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 128
    omega = np.cumsum(hist) / total
    mu = np.cumsum(hist * np.arange(256)) / total
    mu_t = mu[-1]
    denom = omega * (1 - omega)
    denom[denom == 0] = np.nan
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    t = int(np.nanargmax(sigma_b))
    # Newspaper scans: keep the threshold from drifting too dark/light.
    return int(min(max(t + 1, 90), 215))


def runs(mask_1d: np.ndarray, min_gap: int = 0) -> list[tuple[int, int]]:
    """Return [start, end) runs of True; gaps <= min_gap are bridged."""
    idx = np.flatnonzero(mask_1d)
    if idx.size == 0:
        return []
    out: list[list[int]] = [[int(idx[0]), int(idx[0]) + 1]]
    for i in idx[1:]:
        i = int(i)
        if i - out[-1][1] <= min_gap:
            out[-1][1] = i + 1
        else:
            out.append([i, i + 1])
    return [(a, b) for a, b in out]


def ink_bbox(ink: np.ndarray) -> tuple[int, int, int, int] | None:
    ys = np.flatnonzero(ink.any(axis=1))
    xs = np.flatnonzero(ink.any(axis=0))
    if ys.size == 0 or xs.size == 0:
        return None
    return int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1


def resize(arr: np.ndarray, w: int, h: int) -> np.ndarray:
    img = Image.fromarray(arr)
    return np.asarray(img.resize((max(1, w), max(1, h)), Image.LANCZOS))


def to_ink(arr_rgb: np.ndarray, threshold: int) -> np.ndarray:
    gray = np.asarray(Image.fromarray(arr_rgb).convert("L"))
    return gray < threshold


def tone_mask(gray: np.ndarray, ch: int) -> np.ndarray:
    """Areas of continuous tone (photos, tinted graphics) as a closed mask."""
    import cv2

    mid = ((gray > 50) & (gray < 195)).astype(np.float32)
    win = max(3, 2 * ch) | 1
    dens = cv2.boxFilter(mid, -1, (win, win))
    tone = (dens > 0.3).astype(np.uint8)
    tone = cv2.morphologyEx(tone, cv2.MORPH_CLOSE, np.ones((ch, ch), np.uint8))
    # boxFilter smears the edge outwards by ~win/2; pull it back to the real tone pixels.
    tone = cv2.erode(tone, np.ones((win // 2 | 1, win // 2 | 1), np.uint8))
    tone = cv2.dilate(tone, np.ones((3, 3), np.uint8))
    return tone.astype(bool) & cv2.dilate(mid.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)


def strip_rules(arr: np.ndarray, threshold: int) -> np.ndarray:
    """Trim a crop to its text: drop long horizontal rules and surrounding paper."""
    ink = to_ink(arr, threshold)
    rows = ink.mean(axis=1) > 0.6
    keep = ink.copy()
    keep[rows] = False
    bb = ink_bbox(keep)
    if bb is None:
        return arr
    x0, y0, x1, y1 = bb
    pad = 2
    out = arr[max(0, y0 - pad) : y1 + pad, max(0, x0 - pad) : x1 + pad].copy()
    sub_rows = rows[max(0, y0 - pad) : y1 + pad]
    if sub_rows.any():
        bright = arr[~ink]
        out[sub_rows] = np.median(bright, axis=0).astype(np.uint8) if bright.size else 255
    return out


def silhouette(tone: np.ndarray, ch: int) -> np.ndarray:
    """Solid silhouette of a cut-out photo: grow the tone mask slightly and fill its holes."""
    import cv2

    k = max(3, ch // 3) | 1
    m = cv2.dilate(tone.astype(np.uint8), np.ones((k, k), np.uint8))
    pad = np.pad(1 - m, 1, constant_values=1).astype(np.uint8)
    ff = pad.copy()
    cv2.floodFill(ff, np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8), (0, 0), 2)
    holes = (ff == 1)[1:-1, 1:-1]
    return m.astype(bool) | holes


def rule_mask(ink: np.ndarray, ch: int) -> np.ndarray:
    """Long thin horizontal/vertical lines (page frame, rules) — not glyph strokes."""
    import cv2

    u = ink.astype(np.uint8)
    L = max(15, 6 * ch)
    h = cv2.morphologyEx(u, cv2.MORPH_OPEN, np.ones((1, L), np.uint8))
    v = cv2.morphologyEx(u, cv2.MORPH_OPEN, np.ones((L, 1), np.uint8))
    m = (h | v).astype(bool)
    # 두꺼운 덩어리(사진·배경 면)는 선이 아니므로 제외: 선 두께 4px 이하만
    thick_h = cv2.morphologyEx(u, cv2.MORPH_OPEN, np.ones((5, L), np.uint8)).astype(bool)
    thick_v = cv2.morphologyEx(u, cv2.MORPH_OPEN, np.ones((L, 5), np.uint8)).astype(bool)
    m &= ~(thick_h | thick_v)
    return cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ink


def portrait_mask(ink: np.ndarray, tone: np.ndarray, r: "Rect", ch: int) -> np.ndarray:
    """Mask (bbox-local) of a cut-out photo: filled silhouette plus leftover blobs bigger than a glyph.

    Text wrapped around the contour lies outside the silhouette, so it survives.
    """
    import cv2

    import cv2 as _cv2

    t = tone[r.y0 : r.y1, r.x0 : r.x1].astype(np.uint8)
    n0, lab0, st0, _ = _cv2.connectedComponentsWithStats(t, connectivity=8)
    if n0 > 2:
        # 사진 본체(가장 큰 덩어리)만 남긴다 — 촘촘한 글자의 번짐이 계조로 잡힌 조각은 버림
        keep = 1 + int(np.argmax(st0[1:, 4]))
        t = (lab0 == keep).astype(np.uint8)
    out = silhouette(t.astype(bool), ch).copy()
    sub = ink[r.y0 : r.y1, r.x0 : r.x1]
    n, lab, st, _ = cv2.connectedComponentsWithStats(sub.astype(np.uint8), connectivity=8)
    near = cv2.dilate(out.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    base = out.copy()
    for k in range(1, n):
        x, y, w, h, area = st[k]
        comp = lab == k
        inside = int(base[comp].sum())
        touches = bool(near[comp].any())
        # 사진에 닿아 있는 글자보다 큰 덩어리, 또는 대부분 사진 안에 있는 덩어리는 사진의 일부
        if (touches and (w > 1.6 * ch or h > 1.6 * ch)) or inside * 2 >= area:
            out[comp] = True
    return out
