"""Best-effort automatic layout detection → draft layout JSON + preview image.

The draft is meant to be checked (and corrected by hand if needed) before ``build``.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .imageops import Page, Rect, portrait_mask, rule_mask, runs, silhouette, tone_mask
from .layout import FlowItem, Layout
from .textflow import detect_line_runs


def _boxes(mask: np.ndarray) -> list[Rect]:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [Rect(int(x), int(y), int(x + w), int(y + h)) for x, y, w, h, _ in stats[1:n]]


def _union(a: Rect, b: Rect) -> Rect:
    return Rect(min(a.x0, b.x0), min(a.y0, b.y0), max(a.x1, b.x1), max(a.y1, b.y1))


def _near(a: Rect, b: Rect, d: int) -> bool:
    return not (a.x1 + d < b.x0 or b.x1 + d < a.x0 or a.y1 + d < b.y0 or b.y1 + d < a.y0)


def merge_boxes(boxes: list[Rect], d: int) -> list[Rect]:
    boxes = list(boxes)
    changed = True
    while changed:
        changed = False
        out: list[Rect] = []
        for b in boxes:
            for i, o in enumerate(out):
                if _near(o, b, d):
                    out[i] = _union(o, b)
                    changed = True
                    break
            else:
                out.append(b)
        boxes = out
    return boxes


def estimate_char_h(page: Page) -> int:
    k = max(3, page.w // 150)
    lines = cv2.morphologyEx(page.ink.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((1, k), np.uint8))
    hs = [b.h for b in _boxes(lines) if b.w >= 4 * b.h and 4 <= b.h <= page.h // 10]
    if not hs:
        return max(8, page.h // 80)
    hist = np.bincount(hs)
    return int(np.argmax(hist))


def _grow_to_edges(mask: np.ndarray, b: Rect, limit: int) -> Rect:
    """Push each side of a box outwards while the next row/column is still (mostly) filled."""
    x0, y0, x1, y1 = b.x0, b.y0, b.x1, b.y1
    H, W = mask.shape
    for _ in range(limit):
        moved = False
        if y0 > 0 and mask[y0 - 1, x0:x1].mean() > 0.8:
            y0 -= 1; moved = True
        if y1 < H and mask[y1, x0:x1].mean() > 0.8:
            y1 += 1; moved = True
        if x0 > 0 and mask[y0:y1, x0 - 1].mean() > 0.8:
            x0 -= 1; moved = True
        if x1 < W and mask[y0:y1, x1].mean() > 0.8:
            x1 += 1; moved = True
        if not moved:
            break
    return Rect(x0, y0, x1, y1)


def find_images(page: Page, ch: int) -> list[tuple[Rect, str]]:
    g = page.gray
    mid = (g > 50) & (g < 195)
    tone = tone_mask(g, ch)
    cands: list[Rect] = [b for b in _boxes(tone) if b.w > 3 * ch and b.h > 3 * ch]
    # 연한 바탕색을 깐 그래픽 상자(회색·연분홍 박스 등): 거의 모든 화소가 흰 종이보다 어두운 넓은 면
    nonwhite = g < 245
    tint = cv2.boxFilter(nonwhite.astype(np.float32), -1, (2 * ch | 1, 2 * ch | 1)) > 0.9
    tint_boxes = []
    for b in _boxes(tint):
        if b.w > 6 * ch and b.h > 4 * ch and float(tint[b.y0 : b.y1, b.x0 : b.x1].mean()) > 0.6:
            b = _grow_to_edges(nonwhite, b, 2 * ch)
            # 바탕색 상자는 한 가지 연한 색이 넓게 깔려 있다 (사진에는 이런 평평한 색이 드물다)
            hist = np.bincount(g[b.y0 : b.y1, b.x0 : b.x1].ravel(), minlength=256)
            peak = int(np.argmax(hist[200:245])) + 200
            flat = hist[peak - 2 : peak + 3].sum() / max(1, b.w * b.h)
            if flat > 0.25:
                tint_boxes.append(b)
            cands.append(b)
    # Big connected line work (chart frames, axes, boxes) without much tone.
    # (페이지 테두리·긴 괘선처럼 지면 대부분을 두르는 선은 제외)
    cands += [
        b for b in _boxes(page.ink)
        if b.w > 5 * ch and b.h > 3 * ch and b.w < 0.7 * page.w and b.h < 0.7 * page.h
    ]
    merged = merge_boxes(cands, ch // 2)
    gf = g.astype(np.float32)
    local_std = np.sqrt(np.maximum(cv2.blur(gf * gf, (5, 5)) - cv2.blur(gf, (5, 5)) ** 2, 0))
    out = []
    for m in merged:
        sl = (slice(m.y0, m.y1), slice(m.x0, m.x1))
        midmask = mid[sl]
        midfrac = float(midmask.mean())
        sd = local_std[sl][midmask]
        textured = float((sd > 2.5).mean()) if midmask.any() else 0.0
        strong = float((sd > 6).mean()) if midmask.any() else 0.0
        fill = float(silhouette(tone[sl], ch).mean())
        tinted = sum(t.overlaps_x(m) * t.overlaps_y(m) for t in tint_boxes) > 0.4 * m.w * m.h
        if tinted:
            kind = "graphic"  # 바탕색 상자 안의 도표·그래픽 (안에 사진이 있어도 그래픽)
        elif midfrac > 0.2 and textured > 0.5 and fill < 0.8:
            kind = "portrait"  # 직사각형을 채우지 않는 연속 계조 = 윤곽(누끼) 인물사진
        elif midfrac > 0.2 and strong > 0.75:
            kind = "photo"
        else:
            kind = "graphic"  # 단색 면·글자가 많은 도표/그래픽 박스
        out.append((m, kind))
    return out


def attach_caption(ink: np.ndarray, r: Rect, ch: int) -> Rect:
    """Extend an image box over caption lines right below it (up to 3 short lines inside its x range)."""
    y = r.y1
    for _ in range(3):
        probe = Rect(r.x0, y, r.x1, min(ink.shape[0], y + int(2.2 * ch)))
        sub = ink[probe.y0 : probe.y1, probe.x0 : probe.x1]
        rows = np.flatnonzero(sub.any(axis=1))
        if rows.size == 0 or rows[0] > 0.8 * ch:
            break
        end = rows[0]
        while end + 1 < sub.shape[0] and sub[end + 1].any():
            end += 1
        if end - rows[0] + 1 > 1.3 * ch:
            break
        y0, y1 = probe.y0 + rows[0], probe.y0 + end + 1
        # 사진 폭 바깥으로 이어지면 본문이므로 설명이 아니다.
        if ink[y0:y1, max(0, r.x0 - ch // 2) : r.x0].any() or ink[y0:y1, r.x1 : r.x1 + ch // 2].any():
            break
        y = y1 + 1
    return Rect(r.x0, r.y0, r.x1, y + 1) if y > r.y1 else r


def _row_presence(sub: np.ndarray) -> np.ndarray:
    """For each column, which text rows (runs) touch it — counted per text line, not per pixel row."""
    rr = runs(sub.any(axis=1), min_gap=1)
    if not rr:
        return np.zeros((0, sub.shape[1]), dtype=bool)
    return np.stack([sub[a:b].any(axis=0) for a, b in rr])


def xycut(ink: np.ndarray, r: Rect, ch: int, depth: int = 0) -> list[Rect]:
    """Recursive XY-cut: split at the widest white band (row or column) wider than ~0.9 char."""
    sub = ink[r.y0 : r.y1, r.x0 : r.x1]
    ys = np.flatnonzero(sub.any(axis=1))
    xs = np.flatnonzero(sub.any(axis=0))
    if ys.size == 0 or xs.size == 0:
        return []
    r = Rect(r.x0 + int(xs[0]), r.y0 + int(ys[0]), r.x0 + int(xs[-1]) + 1, r.y0 + int(ys[-1]) + 1)
    sub = ink[r.y0 : r.y1, r.x0 : r.x1]
    best = None
    # 세로 자르기는 여러 단에 걸친 띠(시리즈 바·사진 설명 등)가 지나가도 되도록 일부 줄의 잉크를 허용한다.
    pres = _row_presence(sub)
    tol = int(0.1 * pres.shape[0]) if pres.shape[0] >= 6 else 0  # 글줄 10% 까지는 가로지르는 띠 허용
    xprof = np.count_nonzero(pres, axis=0) > tol
    min_gap = 0.9 * ch
    # 완전히 빈 띠를 먼저 쓰고, 없을 때만 띠가 가로지르는 단 사이(허용 자르기)를 쓴다.
    for cands in (((0, sub.any(axis=1)), (1, sub.any(axis=0))), ((1, xprof),)):
        for axis, prof in cands:
            for a, b in runs(~prof):
                if a == 0 or b >= prof.size:
                    continue
                gap = b - a
                if gap >= min_gap and (best is None or gap / min_gap > best[0]):
                    best = (gap / min_gap, axis, a, b)
        if best is not None:
            break
    if best is None or depth > 60:
        return [r]
    _, axis, a, b = best
    mid = (a + b) // 2  # 가운데에서 자른다 (허용한 띠의 잉크를 버리지 않게)
    if axis == 0:
        parts = [Rect(r.x0, r.y0, r.x1, r.y0 + mid), Rect(r.x0, r.y0 + mid, r.x1, r.y1)]
    else:
        parts = [Rect(r.x0, r.y0, r.x0 + mid, r.y1), Rect(r.x0 + mid, r.y0, r.x1, r.y1)]
    return [leaf for p in parts for leaf in xycut(ink, p, ch, depth + 1)]


def split_by_line_height(ink: np.ndarray, b: Rect, ch: int) -> list[Rect]:
    """Split a block whose text rows have very different heights (e.g. masthead line + title)."""
    sub = ink[b.y0 : b.y1, b.x0 : b.x1]
    rr = runs(sub.any(axis=1), min_gap=1)
    if len(rr) < 2:
        return [b]
    groups: list[list[tuple[int, int]]] = [[rr[0]]]
    for a, c in rr[1:]:
        ph = np.median([y1 - y0 for y0, y1 in groups[-1]])
        h = c - a
        if max(h, ph) > 1.6 * min(h, ph) and max(h, ph) > 1.3 * ch:
            groups.append([(a, c)])
        else:
            groups[-1].append((a, c))
    if len(groups) == 1:
        return [b]
    out = []
    for g in groups:
        y0, y1 = b.y0 + g[0][0], b.y0 + g[-1][1]
        cols = np.flatnonzero(ink[y0:y1, b.x0 : b.x1].any(axis=0))
        out.append(Rect(b.x0 + int(cols[0]), y0, b.x0 + int(cols[-1]) + 1, y1))
    return out


def _rows(boxes: list[Rect]) -> list[list[Rect]]:
    rows: list[list[Rect]] = []
    for b in sorted(boxes, key=lambda r: r.y0):
        for row in rows:
            if any(b.overlaps_y(o) > 0.5 * min(b.h, o.h) for o in row):
                row.append(b)
                break
        else:
            rows.append([b])
    return rows


def _unite(boxes: list[Rect]) -> Rect:
    r = boxes[0]
    for b in boxes[1:]:
        r = _union(r, b)
    return r


def _weighted_median(pairs) -> float | None:
    pairs = sorted((v, w) for v, w in pairs if v)
    if not pairs:
        return None
    total = sum(w for _, w in pairs)
    acc = 0
    for v, w in pairs:
        acc += w
        if acc >= total / 2:
            return float(v)
    return float(pairs[-1][0])


def block_info(tpage: Page, b: Rect) -> dict:
    lr = detect_line_runs(tpage, b)
    hs = [y1 - y0 for y0, y1 in lr]
    tops = [y0 for y0, _ in lr]
    pitch = float(np.median(np.diff(tops))) if len(tops) >= 2 else 0.0
    fill = float(tpage.ink[b.y0 : b.y1, b.x0 : b.x1].mean()) if b.w and b.h else 0.0
    return {"n": len(lr), "lh": float(np.median(hs)) if hs else float(b.h), "pitch": pitch, "fill": fill}


def long_rules(page: Page, ch: int) -> list[int]:
    """y of horizontal rules spanning most of the page width (page frame / masthead rule)."""
    rm = rule_mask(page.ink, ch)
    ys = np.flatnonzero(rm.sum(axis=1) > 0.7 * page.w)
    return [int((a + b) // 2) for a, b in runs(np.isin(np.arange(page.h), ys), min_gap=2)]


def is_rectangular(gray: np.ndarray, r: Rect) -> bool:
    """A normal photo fills its box: its edges are straight lines. A cut-out does not."""
    sub = gray[r.y0 : r.y1, r.x0 : r.x1] < 235
    if sub.size == 0:
        return False
    tol = 3
    rows_l = sub[:, :tol].any(axis=1).mean()
    rows_r = sub[:, -tol:].any(axis=1).mean()
    cols_t = sub[:tol, :].any(axis=0).mean()
    cols_b = sub[-tol:, :].any(axis=0).mean()
    return float(np.mean([rows_l, rows_r, cols_t, cols_b])) > 0.75


def detect(path: str) -> tuple[Layout, Image.Image]:
    page = Page(path)
    ch = estimate_char_h(page)

    # 1) 지면 틀: 맨 위 긴 괘선 위 = 매체명·날짜·면, 맨 아래 긴 괘선 아래 = 크기 표기
    rules = long_rules(page, ch)
    top = max([y for y in rules if y < 0.12 * page.h], default=None)
    bottom = min([y for y in rules if y > 0.88 * page.h], default=None)
    zone = Rect(0, (top + 3) if top is not None else 0, page.w, (bottom - 2) if bottom is not None else page.h)

    def ink_box(r: Rect) -> Rect | None:
        sub = page.ink[r.y0 : r.y1, r.x0 : r.x1] & ~rule_mask(page.ink[r.y0 : r.y1, r.x0 : r.x1], ch)
        ys, xs = np.flatnonzero(sub.any(axis=1)), np.flatnonzero(sub.any(axis=0))
        if not ys.size:
            return None
        return Rect(r.x0 + int(xs[0]), r.y0 + int(ys[0]), r.x0 + int(xs[-1]) + 1, r.y0 + int(ys[-1]) + 1)

    header = ink_box(Rect(0, 0, page.w, top - 1)) if top is not None else None
    footer = ink_box(Rect(0, bottom + 2, page.w, page.h)) if bottom is not None else None

    # 2) 사진·도표
    images = []
    for r, kind in find_images(page, ch):
        if r.y1 <= zone.y0 or r.y0 >= zone.y1:
            continue
        r = Rect(r.x0, max(r.y0, zone.y0), r.x1, min(r.y1, zone.y1))
        if kind == "portrait" and is_rectangular(page.gray, r):
            kind = "photo"
        images.append((attach_caption(page.ink, r, ch) if kind != "portrait" else r, kind))

    ink = page.ink & ~rule_mask(page.ink, ch)
    ink[: zone.y0] = False
    ink[zone.y1 :] = False
    tone = tone_mask(page.gray, ch)
    for r, kind in images:
        if kind == "portrait":
            ink[r.y0 : r.y1, r.x0 : r.x1] &= ~portrait_mask(page.ink, tone, r, ch)
        else:
            ink[r.y0 : r.y1, r.x0 : r.x1] = False
    # 사진 가장자리에 남은 잉크 덩어리(글자 크기보다 큰 것)도 지운다.
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    for r, kind in images:
        sl = lab[r.y0 : r.y1, r.x0 : r.x1]
        for k in np.unique(sl):
            if k and (st[k, 2] > 1.6 * ch or st[k, 3] > 1.6 * ch):
                x, y, w, h = st[k, :4]
                if x >= r.x0 - 2 and y >= r.y0 - 2 and x + w <= r.x1 + 2 and y + h <= r.y1 + 2:
                    ink[lab == k] = False

    # 3) 블록 나누기 (XY-cut)
    leaves = [b for b in xycut(ink, zone, ch) if b.h >= 0.5 * ch and b.w >= ch]
    leaves = [part for b in leaves for part in split_by_line_height(ink, b, ch)]
    tpage = page.masked_copy(page.ink & ~ink)  # 그림을 지운 '글자만' 페이지
    info = {b: block_info(tpage, b) for b in leaves}
    multi = [b for b in leaves if info[b]["n"] >= 3]
    body_lh = _weighted_median([(info[b]["lh"], info[b]["n"]) for b in multi]) or ch
    body_pitch = _weighted_median([(info[b]["pitch"], info[b]["n"]) for b in multi if info[b]["pitch"]]) or 1.6 * ch
    body_fill = _weighted_median([(info[b]["fill"], info[b]["n"]) for b in multi]) or 0.15

    # 4) 제목: 가장 큰 글자 줄
    title = None
    big = [b for b in leaves if info[b]["lh"] > 1.8 * body_lh]
    if big:
        rows = _rows(big)
        # 글자가 크고 폭이 넓은 줄 = 제목 (도표 숫자처럼 큰 글자 조각과 구별)
        trow = max(rows, key=lambda row: np.median([info[b]["lh"] for b in row]) * sum(b.w for b in row))
        title = _unite(trow)
        for b in leaves:
            if b not in trow and b.overlaps_y(title) > 0.5 * b.h and info[b]["n"] <= 1:
                title = _union(title, b)
                trow.append(b)
        leaves = [b for b in leaves if b not in trow]

    # 머리 괘선이 없는 지면: 제목 위 첫 줄을 매체명 줄로 본다
    if header is None and title is not None:
        above = sorted((b for b in leaves if b.y1 <= title.y0), key=lambda r: r.y0)
        if above:
            hrow = [b for b in above if b.y0 < above[0].y1 + 0.5 * ch]
            header = _unite(hrow)
            leaves = [b for b in leaves if b not in hrow]

    flow: list[FlowItem] = []
    # 제목 위에 있는 것(시리즈 바, 그래픽 띠 등)은 제목 위에 그대로
    if title is not None:
        pre = [b for b in leaves if b.y1 <= title.y0]
        # 그림에 붙어 있는 조각(그림 밖으로 삐져나온 부분)은 그림에 합친다
        for b in list(pre):
            for k, (r, ik) in enumerate(images):
                if _near(r, b, 3) and r.x0 <= b.x0 and b.x1 <= r.x1:
                    images[k] = (_union(r, b), ik)
                    pre.remove(b)
                    leaves.remove(b)
                    break
        for row in _rows(pre):
            flow.append(FlowItem("block", _unite(row), label="series", note="제목 위 요소"))
        leaves = [b for b in leaves if b not in pre]
        for r, kind in list(images):
            if r.y1 <= title.y0:
                flow.append(FlowItem("image", r, kind="graphic" if kind != "portrait" else kind, note="제목 위 요소"))
                images.remove((r, kind))

    # 5) 본문 블록 / 그 밖의 블록
    def is_body(b) -> bool:
        i = info[b]
        if not (0.8 * body_lh <= i["lh"] <= 1.2 * body_lh) or i["fill"] > 1.5 * body_fill:
            return False
        if i["n"] >= 2:
            return abs(i["pitch"] - body_pitch) <= 0.15 * body_pitch
        return False

    body = [b for b in leaves if is_body(b)]
    other = [b for b in leaves if b not in body]
    if footer is None and body:
        bottom_y = max(b.y1 for b in body)
        tail = [b for b in other if b.y0 >= bottom_y and info[b]["n"] <= 1]
        if tail:
            footer = _unite(tail)
            other = [b for b in other if b not in tail]

    # 단: 여러 줄 본문 블록의 x 범위
    spans: list[list[int]] = []
    for b in sorted((b for b in body if info[b]["n"] >= 3), key=lambda t: t.x0):
        for sp in spans:
            if min(sp[1], b.x1) - max(sp[0], b.x0) > 0.5 * min(sp[1] - sp[0], b.w):
                sp[0], sp[1] = min(sp[0], b.x0), max(sp[1], b.x1)
                break
        else:
            spans.append([b.x0, b.x1])
    spans.sort()
    col_w = int(np.median([sp[1] - sp[0] for sp in spans])) if spans else page.w
    # 본문이 없는 단(도표·사이드바 단)도 단 격자로 채운다
    if len(spans) >= 2:
        step = int(np.median(np.diff([sp[0] for sp in spans])))
        step = min([d for d in np.diff([sp[0] for sp in spans]) if d > 0.5 * col_w] or [step])
        filled = [spans[0]]
        for sp in spans[1:]:
            while sp[0] - filled[-1][0] > 1.5 * step:
                x0 = filled[-1][0] + step
                filled.append([x0, x0 + col_w])
            filled.append(sp)
        while filled[-1][0] + step + 0.5 * col_w <= zone.x1 and any(
            b.x0 >= filled[-1][1] for b in other + [r for r, _ in images]
        ):
            x0 = filled[-1][0] + step
            filled.append([x0, min(page.w, x0 + col_w)])
        spans = filled

    # 단 폭보다 훨씬 좁은 짧은 '본문'(도표 제목·범례 등)은 본문이 아니다
    for b in list(body):
        if info[b]["n"] < 5 and b.w < 0.75 * col_w:
            body.remove(b)
            other.append(b)

    # 같은 줄에 나란히 놓인 한 줄짜리 조각(단 사이에서 잘린 시리즈 바 등)은 다시 붙인다
    changed = True
    while changed:
        changed = False
        for a_ in list(other):
            for b_ in list(other):
                if a_ is b_ or a_ not in other or b_ not in other:
                    continue
                if (
                    info[a_]["n"] <= 1 and info[b_]["n"] <= 1
                    and a_.overlaps_y(b_) > 0.6 * min(a_.h, b_.h)
                    and 0 <= b_.x0 - a_.x1 < 1.6 * ch
                ):
                    u = _union(a_, b_)
                    info[u] = block_info(tpage, u)
                    other.remove(a_)
                    other.remove(b_)
                    other.append(u)
                    changed = True

    def column_of(b: Rect) -> int:
        if not spans:
            return 0
        # 여러 단에 걸친 것은 왼쪽 첫 단에 붙인다
        x = b.x0 + min(b.w, col_w) / 2
        best = min(range(len(spans)), key=lambda i: 0 if spans[i][0] - ch <= x <= spans[i][1] + ch else min(abs(x - spans[i][0]), abs(x - spans[i][1])))
        return best

    cols: list[list[tuple[str, Rect]]] = [[] for _ in range(max(1, len(spans)))]
    for b in body:
        cols[column_of(b)].append(("body", b))
    for b in other:
        cols[column_of(b)].append(("other", b))

    for ci, col in enumerate(cols):
        col.sort(key=lambda t: (t[1].y0, t[1].x0))
        # 같은 줄에 나란히 있는 조각, 이어진 '그 밖의' 조각은 하나의 블록으로 묶는다
        merged: list[list] = []
        for kind, b in col:
            if merged:
                pk, pb = merged[-1]
                same_row = b.overlaps_y(pb) > 0.5 * min(b.h, pb.h)
                if kind == "other" and pk == "other" and (same_row or b.y0 - pb.y1 < 2.5 * ch):
                    merged[-1][1] = _union(pb, b)
                    continue
                if kind == "body" and pk == "body" and b.y0 - pb.y1 < 0.9 * ch:
                    merged[-1][1] = _union(pb, b)
                    continue
            merged.append([kind, b])
        # 단 안 박스(용어 설명 등): '그 밖의' 블록 바로 밑에 붙은 짧은 본문은 그 박스의 일부로 본다
        out: list[list] = []
        for kind, b in merged:
            if (
                kind == "body"
                and out
                and out[-1][0] == "other"
                and b.y0 - out[-1][1].y1 < 1.2 * body_pitch
                and info.get(b, {"n": 99})["n"] < 10
                and not any(k == "body" for k, _ in out)
            ):
                out[-1][1] = _union(out[-1][1], b)
                out[-1].append("box")
                continue
            out.append([kind, b])
        cols[ci] = out

    def label_of(b: Rect, flags) -> tuple[str, str]:
        lr = detect_line_runs(tpage, b)
        hs = [y1 - y0 for y0, y1 in lr] or [b.h]
        fill = float(tpage.ink[b.y0 : b.y1, b.x0 : b.x1].mean())
        if "box" in flags:
            return "box", "단 안의 박스(용어 설명 등) 추정 — 본문이면 type 을 text 로"
        if len(lr) <= 1 and fill > 0.45:
            return "series", ""
        if len(lr) <= 1 and b.w < 0.6 * col_w:
            return "byline", "한 줄짜리 독립 블록(기자명·안내 문구 추정)"
        if np.median(hs) > 1.15 * body_lh or fill > 1.5 * body_fill:
            return "subtitle", ""
        return "box", "그 밖의 블록(그대로 붙임)"

    placed: set[int] = set()
    body_start = len(flow)
    front: list[FlowItem] = []  # 단 맨 위(본문보다 위)에 있던 부제 → 본문 맨 앞으로
    for ci, col in enumerate(cols):
        cx0, cx1 = (spans[ci] if spans else (0, page.w))
        seen_body = False
        for item in col:
            kind, b = item[0], item[1]
            for k, (r, ik) in enumerate(images):
                if k not in placed and r.overlaps_x(Rect(cx0, 0, cx1, 1)) > 0 and r.y0 < b.y0:
                    flow.append(FlowItem("image", r, kind=ik))
                    placed.add(k)
            if kind == "body":
                seen_body = True
                lr = detect_line_runs(tpage, b)
                for seg, wrap in _split_wrap(ink, b, lr, images, ch):
                    flow.append(FlowItem("text", seg, wrap=wrap))
            else:
                label, note = label_of(b, item[2:])
                # 사이드바 단 맨 위의 부제, 또는 본문 중간에 끼어 있는 여러 줄짜리 부제 묶음 → 본문 맨 앞으로
                multi_line = len(detect_line_runs(tpage, b)) >= 2
                if label == "subtitle" and ((not seen_body and ci > 0) or (seen_body and multi_line)):
                    front.append(FlowItem("block", b, label=label, note="본문 앞으로 옮긴 부제"))
                    continue
                flow.append(FlowItem("block", b, label=label, note=note))
        for k, (r, ik) in enumerate(images):
            if k not in placed and r.overlaps_x(Rect(cx0, 0, cx1, 1)) > 0:
                flow.append(FlowItem("image", r, kind=ik))
                placed.add(k)
    for k, (r, ik) in enumerate(images):
        if k not in placed:
            flow.append(FlowItem("image", r, kind=ik))

    if front:
        at = body_start
        while at < len(flow) and flow[at].type == "block" and flow[at].label == "series":
            at += 1
        flow[at:at] = front

    # 인물사진(윤곽) 옆의 한 줄짜리 설명(예: '○○○ 대표 ▶')은 사진과 함께 뺀다.
    for r, ik in images:
        if ik != "portrait":
            continue
        for f in flow:
            if f.type == "block" and f.label == "byline" and f.rect.overlaps_y(r) > 0 and min(abs(f.rect.x1 - r.x0), abs(r.x1 - f.rect.x0)) < 8 * ch:
                f.type, f.kind, f.note = "image", "portrait", "인물사진 설명"

    # 블록 안에 통째로 들어 있는 그림은 블록으로 이미 붙으므로 따로 넣지 않는다(중복 방지)
    blocks_r = [f.rect for f in flow if f.type == "block"]
    flow = [
        f for f in flow
        if not (f.type == "image" and f.kind != "portrait" and any(
            o.x0 <= f.rect.x0 + 2 and o.y0 <= f.rect.y0 + 2 and f.rect.x1 <= o.x1 + 2 and f.rect.y1 <= o.y1 + 2
            for o in blocks_r
        ))
    ]

    grow = lambda r: page.clip(Rect(r.x0 - ch // 3, r.y0 - ch // 4, r.x1 + ch // 3, r.y1 + ch // 4)) if r else r  # noqa: E731
    layout = Layout(image=path, header=grow(header), footer=grow(footer), title=title, flow=flow)
    return layout, preview(page, layout)


def _split_wrap(ink: np.ndarray, b: Rect, lr, images, ch) -> list[tuple[Rect, bool]]:
    """Split a column block into runs of full-width lines and lines shortened by an image."""
    if not lr:
        return [(b, False)]
    flags = []
    for y0, y1 in lr:
        row = Rect(b.x0, y0, b.x1, y1)
        hit = any(r.overlaps_y(row) > 0 and r.overlaps_x(b) > ch for r, _ in images)
        flags.append(hit)
    out = []
    start = 0
    for i in range(1, len(lr) + 1):
        if i == len(lr) or flags[i] != flags[start]:
            seg_rows = lr[start:i]
            ys0 = seg_rows[0][0]
            ys1 = seg_rows[-1][1]
            if flags[start]:
                sub = ink[ys0:ys1, b.x0 : b.x1]
                cols = np.flatnonzero(sub.any(axis=0))
                x0 = b.x0 + int(cols[0]) if cols.size else b.x0
                x1 = b.x0 + int(cols[-1]) + 1 if cols.size else b.x1
                seg = Rect(x0, ys0, x1, ys1)
            else:
                seg = Rect(b.x0, ys0, b.x1, ys1)
            out.append((seg, flags[start]))
            start = i
    return out


COLORS = {
    "header": (128, 0, 128),
    "footer": (128, 0, 128),
    "title": (220, 0, 0),
    "text": (0, 90, 255),
    "wrap": (255, 140, 0),
    "block": (0, 150, 150),
    "image": (0, 160, 0),
}


def preview(page: Page, layout: Layout) -> Image.Image:
    img = Image.fromarray(page.rgb).convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")

    def box(r: Rect, color, label):
        d.rectangle(r.as_list(), outline=color + (255,), width=3)
        d.rectangle([r.x0, r.y0, r.x0 + 8 * len(label) + 6, r.y0 + 14], fill=color + (210,))
        d.text((r.x0 + 3, r.y0 + 1), label, fill=(255, 255, 255, 255))

    if layout.header:
        box(layout.header, COLORS["header"], "header")
    if layout.footer:
        box(layout.footer, COLORS["footer"], "footer")
    if layout.title:
        box(layout.title, COLORS["title"], "title")
    for i, f in enumerate(layout.flow):
        if f.type == "text":
            box(f.rect, COLORS["wrap" if f.wrap else "text"], f"{i}:text{' wrap' if f.wrap else ''}")
        elif f.type == "image":
            box(f.rect, COLORS["image"], f"{i}:{f.kind}")
        else:
            box(f.rect, COLORS["block"], f"{i}:{f.label}")
    return img
