"""Title: cut the original title glyphs and re-set them as two centred lines."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .imageops import Page, Rect, resize, runs
from .textflow import detect_line_runs


@dataclass
class TitleAtom:
    id: str
    x0: int
    x1: int
    band: Rect
    gap_before: int
    punct: str = ""  # "low" (… , .) / "mid" (·) / ""
    line_start: bool = False
    ink_px: int = 0

    @property
    def w(self) -> int:
        return self.x1 - self.x0


@dataclass
class TitleResult:
    lines: list[np.ndarray]
    atom_ids: list[list[str]]
    break_after: int | None
    break_reason: str
    char_h: int
    scale: float
    area_ratio: float
    notes: list[str] = field(default_factory=list)


def title_atoms(page: Page, rect: Rect) -> tuple[list[TitleAtom], int]:
    rect = page.clip(rect)
    line_runs = detect_line_runs(page, rect)
    if not line_runs:
        raise ValueError("제목 영역에서 글자를 찾지 못했습니다.")
    char_h = int(np.median([b - a for a, b in line_runs]))
    pad = max(1, int(0.08 * char_h))
    atoms: list[TitleAtom] = []
    for li, (y0, y1) in enumerate(line_runs):
        band = Rect(rect.x0, y0 - pad, rect.x1, y1 + pad)
        sub = page.ink[max(0, band.y0) : band.y1, band.x0 : band.x1]
        pieces = runs(sub.any(axis=0), min_gap=0)
        def small_mark(a0, a1) -> bool:  # 쉼표·마침표·가운뎃점처럼 작은 조각
            r = np.flatnonzero(sub[:, a0:a1].any(axis=1))
            return bool(r.size) and (r[-1] - r[0] + 1) < 0.4 * char_h

        merged: list[list[int]] = []
        for a0, a1 in pieces:
            # 문장부호 조각은 앞 글자에 붙이지 않는다 (끊을 자리를 찾을 수 있게)
            if (
                merged and a0 - merged[-1][1] <= max(1, 0.1 * char_h) and a1 - merged[-1][0] <= 1.15 * char_h
                and not small_mark(a0, a1) and not small_mark(*merged[-1])
            ):
                merged[-1][1] = a1
            else:
                merged.append([a0, a1])
        prev = None
        for k, (a0, a1) in enumerate(merged):
            ink = sub[:, a0:a1]
            rows = np.flatnonzero(ink.any(axis=1))
            punct = ""
            if rows.size:
                top, bot = rows[0] - pad, rows[-1] - pad
                ih = bot - top + 1
                if ih < 0.4 * char_h:
                    centre = (top + bot) / 2
                    punct = "low" if centre > 0.6 * (y1 - y0) else ("mid" if centre > 0.3 * (y1 - y0) else "")
            atoms.append(
                TitleAtom(
                    id=f"T{li + 1}A{k + 1}",
                    x0=band.x0 + a0,
                    x1=band.x0 + a1,
                    band=band,
                    gap_before=(a0 - prev) if prev is not None else 0,
                    punct=punct,
                    line_start=(k == 0),
                    ink_px=int(ink.sum()),
                )
            )
            prev = a1
    return atoms, char_h


def _row_width(atoms: list[TitleAtom], space: int) -> int:
    w = 0
    for i, a in enumerate(atoms):
        if i:
            w += space if a.line_start else a.gap_before
        w += a.w
    return w


def choose_break(atoms: list[TitleAtom], space: int, manual: int | None) -> tuple[int | None, str]:
    if manual is not None:
        return int(manual), "수동 지정(title_break)"
    n = len(atoms)
    starts = [i for i, a in enumerate(atoms) if a.line_start and i > 0]
    if len(starts) == 1:
        return starts[0] - 1, "원본 줄바꿈 위치 유지"
    # 연속된 점(… 가 점 여러 개로 나뉜 경우)은 마지막 점 뒤에서만 끊는다
    cands = [i for i, a in enumerate(atoms[:-1]) if a.punct in ("low", "mid") and atoms[i + 1].punct not in ("low", "mid")]
    reason = "원본 문장부호(…·쉼표) 뒤"
    if not cands:
        cands = starts or [i for i in range(n - 1) if atoms[i + 1].gap_before > 0.25 * atoms[i].band.h]
        reason = "문장부호가 없어 원본 줄바꿈/띄어쓰기 위치"
    if not cands:
        return None, "끊을 위치를 찾지 못함 (한 줄로 배치)"
    best = min(cands, key=lambda i: abs(_row_width(atoms[: i + 1], space) - _row_width(atoms[i + 1 :], space)))
    return best, reason


def render_row(page: Page, atoms: list[TitleAtom], space: int) -> np.ndarray:
    h = max(a.band.h for a in atoms)
    w = _row_width(atoms, space)
    canvas = np.empty((h, w, 3), dtype=np.uint8)
    canvas[:] = page.paper_rgb
    x = 0
    for i, a in enumerate(atoms):
        if i:
            x += space if a.line_start else a.gap_before
        crop = page.crop_rgb(Rect(a.x0, a.band.y0, a.x1, a.band.y1))
        canvas[: crop.shape[0], x : x + a.w] = np.minimum(canvas[: crop.shape[0], x : x + a.w], crop)
        x += a.w
    return canvas


def build_title(page: Page, rect: Rect, body_char_h: int, max_width: int, area_ratio: float, manual_break: int | None) -> TitleResult:
    atoms, char_h = title_atoms(page, rect)
    gaps = [a.gap_before for a in atoms if not a.line_start]
    space = int(np.median([g for g in gaps if g > 0.2 * char_h])) if any(g > 0.2 * char_h for g in gaps) else int(0.3 * char_h)
    brk, reason = choose_break(atoms, space, manual_break)
    if brk is None:
        rows = [atoms]
    else:
        rows = [atoms[: brk + 1], atoms[brk + 1 :]]
        # 끊은 자리 다음 줄 첫 글자는 새 줄의 시작으로 취급(앞 공백 제거)
        rows[1] = [TitleAtom(**{**rows[1][0].__dict__, "line_start": True, "gap_before": 0})] + rows[1][1:]
    rendered = [render_row(page, r, space) for r in rows]
    target = math.sqrt(area_ratio) * body_char_h
    scale = target / char_h
    notes = []
    widest = max(r.shape[1] for r in rendered)
    if widest * scale > max_width:
        fit = max_width / widest
        notes.append(
            f"제목을 면적 {area_ratio:.1f}배로 키우면 3단 폭을 넘어 {(fit * char_h / body_char_h) ** 2:.1f}배로 줄였습니다."
        )
        scale = fit
    lines = [resize(r, int(round(r.shape[1] * scale)), int(round(r.shape[0] * scale))) for r in rendered]
    return TitleResult(
        lines=lines,
        atom_ids=[[a.id for a in r] for r in rows],
        break_after=brk,
        break_reason=reason,
        char_h=char_h,
        scale=scale,
        area_ratio=(scale * char_h / body_char_h) ** 2,
        notes=notes,
    )
