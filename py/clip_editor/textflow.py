"""Body text: line detection, glyph atoms, paragraphs and re-packing of short lines.

Nothing here draws a glyph: every pixel that ends up in the output is cut out of the
original page. Re-packing only moves whole glyph crops ("atoms") between lines.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .imageops import Page, Rect, runs

LOW_PUNCT_MAX_H = 0.42  # atom ink height / char height → 마침표·쉼표 같은 아래쪽 문장부호


@dataclass
class Atom:
    """Smallest unit moved during re-packing: one glyph (or a few merged glyph pieces)."""

    id: str
    x0: int  # absolute page coords of the ink columns
    x1: int
    band: Rect  # vertical band of its source line (atom crop = band rows × [x0, x1))
    ink_px: int
    low: bool = False  # sits at the baseline and is small (. , …)
    gap_before: int = 0  # original gap to the previous atom in the same line
    space: bool = False  # the gap before this atom is a word space
    punct: bool = False  # the whole atom is a small baseline mark (줄머리 금칙 대상)
    glued: bool = False  # ink touches the previous atom (split at the thinnest column): keep them together

    @property
    def w(self) -> int:
        return self.x1 - self.x0


@dataclass
class Line:
    id: str
    region: int  # index into the flow list
    ink: Rect  # ink bounding box (absolute)
    band: Rect = None  # crop band (absolute); x = region text range
    atoms: list[Atom] = field(default_factory=list)
    indent: bool = False
    short: bool = False
    wrap: bool = False
    ink_px: int = 0


@dataclass
class Region:
    index: int
    rect: Rect
    wrap: bool
    text_left: int = 0
    text_right: int = 0
    lines: list[Line] = field(default_factory=list)

    @property
    def text_w(self) -> int:
        return self.text_right - self.text_left


@dataclass
class Metrics:
    char_h: int
    pitch: int
    letter_gap: int
    space_gap: int
    space_threshold: float
    column_width: int


@dataclass
class Paragraph:
    index: int
    lines: list[Line]
    repack: bool = False
    joins: list[tuple[str, str, bool]] = field(default_factory=list)  # (line id, next id, space?)


def detect_line_runs(page: Page, rect: Rect, hint: float | None = None) -> list[tuple[int, int]]:
    ink = page.crop_ink(rect)
    if ink.size == 0:
        return []
    row = ink.sum(axis=1)
    thr = max(1, int(0.002 * rect.w))
    rr = runs(row >= thr, min_gap=1)
    if not rr:
        return []
    heights = sorted(b - a for a, b in rr)
    med = hint or heights[len(heights) // 2]
    # Fold slivers (accent marks, underline dots, stray specks) into the nearest line so no ink is lost.
    merged: list[list[int]] = []
    for a, b in rr:
        if merged and (b - a) < 0.45 * med and a - merged[-1][1] < 0.5 * med:
            merged[-1][1] = b
            continue
        if merged and (merged[-1][1] - merged[-1][0]) < 0.45 * med and a - merged[-1][1] < 0.5 * med:
            merged[-1][1] = b
            continue
        merged.append([a, b])
    return [(a + rect.y0, b + rect.y0) for a, b in merged]


def build_regions(page: Page, flow) -> list[Region]:
    regions: list[Region] = []
    for i, item in enumerate(flow):
        if item.type != "text":
            continue
        reg = Region(index=i, rect=page.clip(item.rect), wrap=bool(item.wrap))
        for k, (y0, y1) in enumerate(detect_line_runs(page, reg.rect)):
            sub = page.ink[y0:y1, reg.rect.x0 : reg.rect.x1]
            cols = np.flatnonzero(sub.any(axis=0))
            if cols.size == 0:
                continue
            ink = Rect(reg.rect.x0 + int(cols[0]), y0, reg.rect.x0 + int(cols[-1]) + 1, y1)
            reg.lines.append(Line(id=f"R{i}L{k + 1}", region=i, ink=ink, ink_px=int(sub.sum())))
        if reg.lines:
            lefts = sorted(l.ink.x0 for l in reg.lines)
            rights = sorted(l.ink.x1 for l in reg.lines)
            reg.text_left = lefts[0] if len(lefts) < 3 else min(lefts[: max(1, len(lefts) // 3)])
            reg.text_right = max(rights)
            regions.append(reg)
    return regions


def estimate_metrics(regions: list[Region], column_width: int | None) -> Metrics:
    heights = [l.ink.h for r in regions for l in r.lines]
    if not heights:
        raise ValueError("본문 영역에서 글줄을 찾지 못했습니다. layout JSON 의 text 영역을 확인하세요.")
    char_h = int(np.median(heights))
    diffs = []
    for r in regions:
        for a, b in zip(r.lines, r.lines[1:]):
            d = b.ink.y0 - a.ink.y0
            if 0.9 * char_h < d < 2.6 * char_h:
                diffs.append(d)
    pitch = int(round(np.median(diffs))) if diffs else int(round(char_h * 1.6))
    if column_width:
        cw = int(column_width)
    else:
        widths = sorted((r.text_w for r in regions if not r.wrap), reverse=True) or sorted(
            (r.text_w for r in regions), reverse=True
        )
        top = [w for w in widths if w >= 0.9 * widths[0]]
        cw = int(np.median(top))
    return Metrics(char_h=char_h, pitch=pitch, letter_gap=0, space_gap=0, space_threshold=0, column_width=cw)


def assign_bands(regions: list[Region], m: Metrics) -> None:
    top_pad = max(0, (m.pitch - m.char_h) // 2)
    lines = [l for r in regions for l in r.lines]
    for r in regions:
        for l in r.lines:
            y0 = l.ink.y0 - top_pad
            y1 = max(y0 + m.pitch, l.ink.y1 + 1)
            l.band = Rect(r.text_left, y0, max(r.text_right, r.text_left + 1), y1)
    # Bands of lines that sit on top of each other must not share rows (no duplicated ink).
    for a in lines:
        for b in lines:
            if a is b or a.band.overlaps_x(b.band) == 0:
                continue
            if a.ink.y0 < b.ink.y0 and a.band.y1 > b.band.y0:
                cut = max(a.ink.y1, min(b.ink.y0, (a.band.y1 + b.band.y0) // 2))
                a.band = Rect(a.band.x0, a.band.y0, a.band.x1, cut)
                b.band = Rect(b.band.x0, cut, b.band.x1, b.band.y1)


def _two_means(values: list[int]) -> float:
    v = np.array(sorted(values), dtype=float)
    if v.size < 4 or v[0] == v[-1]:
        return float(v[-1] + 1) if v.size else 1.0
    lo, hi = v[0], v[-1]
    t = (lo + hi) / 2
    for _ in range(50):
        a, b = v[v <= t], v[v > t]
        if a.size == 0 or b.size == 0:
            break
        nt = (a.mean() + b.mean()) / 2
        if abs(nt - t) < 0.01:
            break
        t = nt
    return float(t)


def split_wide(pieces: list[tuple[int, int]], colsum: np.ndarray, syl_w: float) -> list[tuple[int, int]]:
    """Split ink runs where neighbouring glyphs touch ("따르") at the thinnest column."""
    out: list[tuple[int, int]] = []
    for a0, a1 in pieces:
        w = a1 - a0
        n = int(round(w / syl_w)) if w > 1.4 * syl_w else 1
        if n <= 1:
            out.append((a0, a1))
            continue
        cuts = [a0]
        for k in range(1, n):
            centre = a0 + k * w / n
            lo, hi = int(centre - 0.25 * syl_w), int(centre + 0.25 * syl_w) + 1
            lo, hi = max(lo, cuts[-1] + 1), min(hi, a1 - 1)
            if hi <= lo:
                continue
            cuts.append(lo + int(np.argmin(colsum[lo:hi])))
        cuts.append(a1)
        out.extend((cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1))
    return out


def segment_atoms(page: Page, regions: list[Region], m: Metrics) -> None:
    # 한 글자(음절)가 여러 조각(ㅂ+ㅣ 등)으로 나뉘지 않도록, 합친 폭이 보통 글자 폭 이하이면 묶는다.
    widths = []
    for r in regions:
        for l in r.lines:
            b = l.band
            for a0, a1 in runs(page.ink[b.y0 : b.y1, b.x0 : b.x1].any(axis=0), min_gap=0):
                if 0.6 * m.char_h <= a1 - a0 <= 1.3 * m.char_h:
                    widths.append(a1 - a0)
    syl_w = float(np.median(widths)) if widths else float(m.char_h)
    merge_gap = max(1, int(round(0.35 * m.char_h)))
    max_w = 1.1 * syl_w
    for r in regions:
        for l in r.lines:
            b = l.band
            sub = page.ink[b.y0 : b.y1, b.x0 : b.x1]
            col = sub.any(axis=0)
            pieces = split_wide(runs(col, min_gap=0), sub.sum(axis=0), syl_w)
            atoms: list[list[int]] = []
            for a0, a1 in pieces:
                if atoms and a0 - atoms[-1][1] <= merge_gap and a1 - atoms[-1][0] <= max_w:
                    atoms[-1][1] = a1
                    atoms[-1][2] = a0
                else:
                    atoms.append([a0, a1, a0])  # [start, end, start of last piece]
            prev_end = None
            l.atoms = []
            for k, (a0, a1, last0) in enumerate(atoms):
                ink = sub[:, a0:a1]
                # 마지막 조각이 작고 아래쪽에 있으면 문장부호(. ,)로 끝나는 것으로 본다 ("다." 처럼 붙은 경우 포함).
                tail = sub[:, last0:a1]
                rows = np.flatnonzero(tail.any(axis=1))
                ih = int(rows[-1] - rows[0] + 1) if rows.size else 0
                low = ih < LOW_PUNCT_MAX_H * m.char_h and rows.size and (b.y0 + rows[0]) > l.ink.y0 + 0.45 * m.char_h
                l.atoms.append(
                    Atom(
                        id=f"{l.id}A{k + 1}",
                        x0=b.x0 + a0,
                        x1=b.x0 + a1,
                        band=b,
                        ink_px=int(ink.sum()),
                        low=bool(low),
                        punct=bool(low and last0 == a0),
                        glued=bool(prev_end is not None and a0 == prev_end),
                        gap_before=(a0 - prev_end) if prev_end is not None else (b.x0 + a0 - r.text_left),
                    )
                )
                prev_end = a1
            l.indent = (l.ink.x0 - r.text_left) > 0.5 * m.char_h
            l.short = (r.text_right - l.ink.x1) > 0.9 * m.char_h and not r.wrap
    # 낱자 간격 vs 띄어쓰기: 꽉 찬(사진 옆이 아닌) 줄에서 기준값을 잡는다.
    full = [l for r in regions if not r.wrap for l in r.lines] or [l for r in regions for l in r.lines]
    # 띄어쓰기는 글자 높이보다 좁다: 그보다 넓은 간격(탭·정렬 공백 등)은 기준 계산에서 뺀다.
    gaps = [a.gap_before for l in full for a in l.atoms[1:] if a.gap_before <= 0.8 * m.char_h]
    thr = _two_means(gaps) if gaps else m.char_h * 0.25
    thr = max(thr, 0.15 * m.char_h)
    small = [g for g in gaps if g <= thr]
    big = [g for g in gaps if g > thr]
    m.space_threshold = thr
    m.letter_gap = int(np.median(small)) if small else max(1, int(0.05 * m.char_h))
    m.space_gap = int(np.median(big)) if big else max(2, int(0.3 * m.char_h))
    # 양끝맞춤으로 늘어난 줄은 모든 간격이 같은 양만큼 늘어나므로, 그 줄의 늘어난 양만큼 띄어쓰기 기준을 올린다.
    # 한글 음절은 글자 폭(advance)이 같으므로, 한글끼리는 잉크 간격보다 '글자 중심 사이 거리'가 더 정확하다.
    m.space_gap = max(m.space_gap, m.letter_gap + 2)
    space_add = m.space_gap - m.letter_gap
    for r in regions:
        for l in r.lines:
            at = l.atoms
            if len(at) < 2:
                continue
            g = [a.gap_before for a in at[1:]]
            stretch = max(0.0, float(sorted(g)[len(g) // 4]) - m.letter_gap)
            hangul = [0.75 * syl_w <= a.w <= 1.12 * syl_w for a in at]
            dists = [
                (at[k].x0 + at[k].x1 - at[k - 1].x0 - at[k - 1].x1) / 2
                for k in range(1, len(at))
                if hangul[k] and hangul[k - 1]
            ]
            adv = sorted(dists)[len(dists) // 4] if len(dists) >= 3 else None
            for k in range(1, len(at)):
                a = at[k]
                if adv is not None and hangul[k] and hangul[k - 1] and not a.glued and not at[k - 1].glued:
                    d = (a.x0 + a.x1 - at[k - 1].x0 - at[k - 1].x1) / 2
                    a.space = d > adv + 0.5 * space_add
                else:
                    a.space = a.gap_before > m.space_threshold + stretch


def mark_wrap(regions: list[Region], m: Metrics, flow) -> None:
    for r in regions:
        explicit = flow[r.index].wrap
        if explicit is None:
            r.wrap = r.text_w < 0.9 * m.column_width
        else:
            r.wrap = bool(explicit)
        for l in r.lines:
            l.wrap = r.wrap


def paragraphs(flow, regions: list[Region]) -> list[Paragraph]:
    """Split the reading-order line stream into paragraphs.

    Korean newspaper paragraphs start with an indented line, so a paragraph ends right before
    an indented line or at the end of the text. A paragraph continues across regions, columns,
    images and blocks (사이드바·도표가 문장 중간에 끼어 있어도 문단은 이어진다).

    A short line that is not followed by an indented line is either
      - shortened by something next to it (사진 옆 줄 바로 앞뒤): marked ``wrap`` → re-packed, or
      - a paragraph end before a non-indented line (◇ 중간제목 등): the paragraph ends there.
    """
    by_index = {r.index: r for r in regions}
    stream: list[Line] = []
    for i, item in enumerate(flow):
        if item.type == "text" and i in by_index:
            stream.extend(by_index[i].lines)
    groups: list[list[Line]] = []
    for k, l in enumerate(stream):
        if not groups or l.indent:
            groups.append([])
        groups[-1].append(l)
        if l.short and k + 1 < len(stream) and not stream[k + 1].indent:
            near_wrap = (k > 0 and stream[k - 1].wrap) or stream[k + 1].wrap
            if near_wrap:
                l.wrap = True
            else:
                groups.append([])
    out = [Paragraph(index=n + 1, lines=g) for n, g in enumerate(gr for gr in groups if gr)]
    for p in out:
        p.repack = any(l.wrap for l in p.lines)
    return out


@dataclass
class PackedLine:
    """A re-packed line: atoms with their x position inside the column."""

    id: str
    atoms: list[tuple[Atom, int]]  # (atom, x offset in column)
    height: int
    source_lines: list[str]
    last: bool


def repack(p: Paragraph, m: Metrics, joins: dict | None = None, default: str = "punct") -> list[PackedLine]:
    """Re-break a paragraph at full column width.

    joins: {line id: bool} — 그 줄 끝과 다음 줄 첫 글자 사이를 띄울지(수동 지정).
    default: punct = 문장부호 뒤만 띄움, space = 항상 띄움, none = 항상 붙임.
    """
    joins = joins or {}
    p.joins = []  # 다시 짤 때마다 새로 기록
    cw = m.column_width
    # Build the atom stream with normalized gaps.
    stream: list[tuple[Atom, int, bool]] = []  # (atom, gap before, is word space)
    for li, l in enumerate(p.lines):
        for k, a in enumerate(l.atoms):
            if li == 0 and k == 0:
                gap, space = (a.gap_before if l.indent else 0), False
            elif k == 0:
                prev_line = p.lines[li - 1]
                prev = prev_line.atoms[-1] if prev_line.atoms else None
                if prev_line.id in joins:
                    space = bool(joins[prev_line.id])
                elif default == "space":
                    space = True
                elif default == "none":
                    space = False
                else:
                    space = bool(prev and prev.low)
                p.joins.append((p.lines[li - 1].id, l.id, space))
                gap = m.space_gap if space else m.letter_gap
            else:
                space = a.space
                gap = 0 if a.glued else (m.space_gap if space else m.letter_gap)
            stream.append((a, gap, space))
    height = max(l.band.h for l in p.lines)

    rows: list[list[tuple[Atom, int, bool]]] = [[]]
    width = 0
    for idx, (a, gap, space) in enumerate(stream):
        row = rows[-1]
        g = gap if (row or idx == 0) else 0  # 문단 첫 줄 들여쓰기만 유지
        if row and width + g + a.w > cw:
            # 줄머리 금칙: 마침표·쉼표로 줄을 시작하지 않는다 → 앞 글자 하나를 함께 내린다.
            # 붙어 있던 글자(glued)도 떼지 않고 함께 내린다.
            carry = []
            if a.punct or a.glued:
                carry.append(row.pop())
                while row and carry[0][0].glued and len(row) > 1:
                    carry.insert(0, row.pop())
                if not row:
                    row.extend(carry)
                    carry = []
            rows.append([])
            row = rows[-1]
            width = 0
            for n, (c, cg, cs) in enumerate(carry):
                cg = 0 if n == 0 else cg
                row.append((c, cg, cs))
                width += cg + c.w
            g = gap if row else 0
        row.append((a, g, space))
        width += g + a.w

    packed: list[PackedLine] = []
    for ri, row in enumerate(rows):
        if not row:
            continue
        last = ri == len(rows) - 1
        gaps = [g for _, g, _ in row]
        used = sum(a.w for a, _, _ in row) + sum(gaps)
        extra = 0 if last else max(0, cw - used)
        add = [0] * len(row)
        if extra and len(row) > 1:
            spaces = [k for k in range(1, len(row)) if row[k][2]]
            allk = [k for k in range(1, len(row)) if not row[k][0].glued] or list(range(1, len(row)))
            if spaces:
                cap = 2 * m.space_gap
                per = min(cap, extra // len(spaces))
                for k in spaces:
                    add[k] += per
                extra -= per * len(spaces)
            base, rem = divmod(extra, len(allk))
            for n, k in enumerate(allk):
                add[k] += base + (1 if n < rem else 0)
        x = 0
        placed = []
        for k, (a, g, _) in enumerate(row):
            x += g + add[k]
            placed.append((a, x))
            x += a.w
        src = []
        for a, _ in placed:
            lid = a.id.split("A")[0]
            if lid not in src:
                src.append(lid)
        packed.append(PackedLine(id=f"P{p.index}N{len(packed) + 1}", atoms=placed, height=height, source_lines=src, last=last))
    return packed


def render_packed(page: Page, pl: PackedLine, width: int) -> np.ndarray:
    canvas = np.empty((pl.height, width, 3), dtype=np.uint8)
    canvas[:] = page.paper_rgb
    for a, x in pl.atoms:
        crop = page.crop_rgb(Rect(a.x0, a.band.y0, a.x1, a.band.y1))
        h = min(crop.shape[0], pl.height)
        w = min(crop.shape[1], width - x)
        if w > 0:
            canvas[:h, x : x + w] = np.minimum(canvas[:h, x : x + w], crop[:h, :w])
    return canvas


def render_line(page: Page, l: Line, width: int) -> np.ndarray:
    crop = page.crop_rgb(l.band)
    if crop.shape[1] == width:
        return crop
    out = np.empty((crop.shape[0], max(width, crop.shape[1]), 3), dtype=np.uint8)
    out[:] = page.paper_rgb
    out[:, : crop.shape[1]] = crop
    return out
