"""Assemble the 3-column clip and verify that no glyph was lost or duplicated."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

from .imageops import Page, Rect, portrait_mask, resize, strip_rules, to_ink, tone_mask
from .layout import Layout
from .textflow import (
    Metrics,
    Paragraph,
    assign_bands,
    build_regions,
    estimate_metrics,
    mark_wrap,
    paragraphs,
    render_line,
    render_packed,
    repack,
    segment_atoms,
)
from .title import TitleResult, build_title


@dataclass
class Options:
    columns: int = 3
    title_area_ratio: float = 7.0  # 제목 글자 면적 / 본문 글자 면적
    max_height_ratio: float = 2.0  # 편집본 세로/가로 가 이 값을 넘으면 일반 사진을 뺀다
    gutter: float = 1.5  # 단 사이 간격 (본문 글자 높이 배수)
    margin: float = 2.0
    rule: bool = True
    wide_ratio: float = 2.0  # 원래 폭이 단 폭의 이 배수 이상인 그림은 3단 아래 전체 폭으로
    join_default: str = "punct"  # 다시 짠 문단의 줄 이음: punct(문장부호 뒤만 띄움) | space | none


@dataclass(eq=False)
class Item:
    kind: str  # line | packed | block | image
    img: np.ndarray
    ids: list[str]  # source unit ids carried by this item (line ids or atom ids)
    label: str = ""
    para: int | None = None
    pad: int = 0  # paper rows above/below inside img

    @property
    def h(self) -> int:
        return self.img.shape[0]


@dataclass
class Placement:
    item: Item
    x: int
    y: int
    column: int


@dataclass
class Result:
    image: Image.Image
    annotated: Image.Image
    report: dict = field(default_factory=dict)
    joins: Image.Image | None = None
    page: Page | None = None  # text page (images painted out) the glyphs were cut from
    paras: list | None = None


def _pad_block(page: Page, img: np.ndarray, width: int, pad: int, align: str = "left") -> np.ndarray:
    out = np.empty((img.shape[0] + 2 * pad, max(width, img.shape[1]), 3), dtype=np.uint8)
    out[:] = page.paper_rgb
    x = (out.shape[1] - img.shape[1]) // 2 if align == "center" else 0
    out[pad : pad + img.shape[0], x : x + img.shape[1]] = img
    return out


def regroup_wide(img: np.ndarray, page: Page, width: int, ch: int) -> np.ndarray:
    """A header wider than the edit (매체명 ... 날짜·면) is re-spaced, not shrunk:
    groups separated by wide blanks keep their size; the first goes left, the last right."""
    if img.shape[1] <= width:
        return img
    from .imageops import runs as _runs

    ink = to_ink(img, page.threshold)
    groups = _runs(ink.any(axis=0), min_gap=3 * ch)
    if len(groups) < 2:
        return img
    total = sum(b - a for a, b in groups)
    if total + (len(groups) - 1) * 2 * ch > width:
        return img
    out = np.empty((img.shape[0], width, 3), dtype=np.uint8)
    out[:] = page.paper_rgb
    pad = 2
    first, last = groups[0], groups[-1]
    out[:, : first[1] - first[0] + pad] = img[:, max(0, first[0] - pad) : first[1]][:, : first[1] - first[0] + pad]
    lw = last[1] - last[0]
    out[:, width - lw - pad :] = img[:, last[0] - pad : last[1]] if last[0] >= pad else img[:, last[0] : last[1] + pad]
    mids = groups[1:-1]
    if mids:
        mw = mids[-1][1] - mids[0][0]
        x = (width - mw) // 2
        out[:, x : x + mw] = img[:, mids[0][0] : mids[-1][1]]
    return out


def _fit_width(img: np.ndarray, width: int) -> tuple[np.ndarray, bool]:
    if img.shape[1] <= width:
        return img, False
    s = width / img.shape[1]
    return resize(img, width, int(round(img.shape[0] * s))), True


def float_to_boundaries(items: list[Item]) -> list[Item]:
    """Blocks and images that sit inside a paragraph (사이드바·도표 등) move to the end of that paragraph,
    so that no sentence is interrupted by them."""
    texty = lambda it: it.kind in ("line", "packed")  # noqa: E731
    out = list(items)
    k = 0
    while k < len(out):
        it = out[k]
        if not texty(it):
            prev = next((out[j] for j in range(k - 1, -1, -1) if texty(out[j])), None)
            nxt_i = next((j for j in range(k + 1, len(out)) if texty(out[j])), None)
            if prev is not None and nxt_i is not None and out[nxt_i].para == prev.para:
                end = nxt_i
                while end + 1 < len(out) and (not texty(out[end + 1]) or out[end + 1].para == prev.para):
                    if not texty(out[end + 1]):
                        break
                    end += 1
                out.insert(end + 1, out.pop(k))
                continue
        k += 1
    return out


def pack_columns(items: list[Item], n: int) -> list[list[Item]]:
    if not items:
        return [[] for _ in range(n)]

    def fill(limit: int) -> list[list[Item]] | None:
        cols: list[list[Item]] = [[]]
        h = 0
        for it in items:
            if cols[-1] and h + it.h > limit:
                cols.append([])
                h = 0
            cols[-1].append(it)
            h += it.h
        return cols if len(cols) <= n else None

    lo = max(it.h for it in items)
    hi = sum(it.h for it in items)
    while lo < hi:
        mid = (lo + hi) // 2
        if fill(mid):
            hi = mid
        else:
            lo = mid + 1
    cols = fill(lo)
    cols += [[] for _ in range(n - len(cols))]
    # 단 끝 문장이 다음 단 첫 줄로 곧바로 이어지고 문장 중간이 끊기지 않도록, 그림·블록을
    # 같은 단 안의 가장 가까운 문단 경계로 옮긴다 (단 맨 위·맨 아래는 문장이 이어지는 자리라 피함).
    texty = lambda it: it.kind in ("line", "packed")  # noqa: E731
    for ci, col in enumerate(cols):
        later = any(cols[k] for k in range(ci + 1, n))
        for it in [x for x in col if not texty(x)]:
            k = col.index(it)
            rest = col[:k] + col[k + 1 :]

            def para_before(j):
                return next((rest[t].para for t in range(j - 1, -1, -1) if texty(rest[t])), None)

            def para_after(j):
                return next((rest[t].para for t in range(j, len(rest)) if texty(rest[t])), None)

            def ok(j):
                if j == 0:
                    return ci == 0 or not any(texty(x) for x in rest)
                if j == len(rest):
                    return not later
                pb, pa = para_before(j), para_after(j)
                return pb is None or pa is None or pb != pa

            def nice(j):  # 한 줄짜리 문단(◇ 중간제목 등) 바로 뒤는 피한다
                pb = para_before(j)
                return pb is None or sum(1 for x in rest if texty(x) and x.para == pb) > 1

            if ok(k) and nice(k):
                continue
            cands = [j for j in range(len(rest) + 1) if ok(j) and nice(j)] or [
                j for j in range(len(rest) + 1) if ok(j)
            ]
            if cands:
                j = min(cands, key=lambda c: abs(c - k))
            elif k == 0:
                j = 1  # 경계가 없으면 한 줄만이라도 앞으로 (단 첫 줄이 앞 단에서 이어지게)
            elif k == len(col) - 1:
                j = len(rest) - 1
            else:
                continue
            col[:] = rest[:j] + [it] + rest[j:]
    return cols


@dataclass
class Analysis:
    src: Page  # original page (images are cut from here)
    page: Page  # page with images painted out (text is cut from here)
    regions: list
    metrics: Metrics
    paragraphs: list[Paragraph]
    cut_glyph_px: int
    covered_glyph_px: int = 0


def analyze(layout: Layout) -> Analysis:
    src = Page(layout.image)
    flow = layout.flow
    # 본문 글줄은 그림을 지운 페이지에서 찾는다 (사진 조각이 글줄로 섞이지 않게).
    from .detect import estimate_char_h

    img_mask = np.zeros((src.h, src.w), dtype=bool)
    tone = None
    ch0 = 0
    suspicious = 0
    for f in flow:
        if f.type != "image":
            continue
        r = src.clip(f.rect)
        if f.kind == "portrait":
            if tone is None:
                ch0 = estimate_char_h(src)
                tone = tone_mask(src.gray, ch0)
            pm = portrait_mask(src.ink, tone, r, ch0)
            img_mask[r.y0 : r.y1, r.x0 : r.x1] |= pm
            suspicious += covered_glyph_ink(src.ink, tone, r, pm, ch0)
        else:
            img_mask[r.y0 : r.y1, r.x0 : r.x1] = True
    text_mask = np.zeros_like(img_mask)
    for f in flow:
        if f.type == "text":
            r = src.clip(f.rect)
            text_mask[r.y0 : r.y1, r.x0 : r.x1] = True
    img_mask = absorb_image_ink(src.ink, img_mask, text_mask)
    from .imageops import rule_mask

    img_mask |= rule_mask(src.ink, estimate_char_h(src)) & text_mask  # 본문 영역에 걸친 괘선·테두리
    cut = cut_glyph_ink(src.ink, img_mask, text_mask)
    page = src.masked_copy(img_mask)

    regions = build_regions(page, flow)
    m = estimate_metrics(regions, layout.column_width)
    mark_wrap(regions, m, flow)
    assign_bands(regions, m)
    segment_atoms(page, regions, m)
    fix_indent_beside_images(regions, img_mask, m)
    paras = paragraphs(flow, regions)
    return Analysis(src, page, regions, m, paras, cut, suspicious)


def covered_glyph_ink(ink: np.ndarray, tone: np.ndarray, r: Rect, pm: np.ndarray, ch: int) -> int:
    """Glyph-sized ink blobs that the portrait mask covers but that are nowhere near its tone area
    (= probably body text swallowed by the mask)."""
    import cv2

    from .imageops import silhouette

    core = silhouette(tone[r.y0 : r.y1, r.x0 : r.x1], 1)  # hole-filled, (almost) not grown
    sub = ink[r.y0 : r.y1, r.x0 : r.x1] & pm
    n, lab, st, _ = cv2.connectedComponentsWithStats(sub.astype(np.uint8), connectivity=8)
    px = 0
    for k in range(1, n):
        x, y, w, h, area = st[k]
        if 0.4 * ch <= h <= 1.3 * ch and w <= 3 * ch and not core[lab == k].any():
            px += int(area)
    return px


def fix_indent_beside_images(regions, img_mask: np.ndarray, m: Metrics) -> None:
    """Lines starting right of a (cut-out) photo follow its contour: measure indent from the photo edge."""
    for r in regions:
        offs = {}
        for l in r.lines:
            x_lo = max(0, r.rect.x0 - 3 * m.char_h)
            rows = img_mask[l.ink.y0 : l.ink.y1, x_lo : l.ink.x0]
            cols = np.flatnonzero(rows.any(axis=0)) if rows.size else np.array([])
            if cols.size:
                offs[l.id] = l.ink.x0 - (x_lo + int(cols[-1]) + 1)
        if not offs:
            continue
        base = sorted(offs.values())[len(offs) // 4]
        prev = None
        for l in r.lines:
            if l.id in offs:
                # 윤곽선 간격은 들쭉날쭉하므로, 앞줄이 문장부호(. " 등)로 끝날 때만 문단 시작으로 본다.
                ended = bool(prev and prev.atoms and (prev.atoms[-1].low or prev.atoms[-1].w < 0.5 * m.char_h))
                l.indent = offs[l.id] - base > 0.5 * m.char_h and (prev is None or ended)
            prev = l


def build(layout: Layout, opt: Options | None = None) -> Result:
    opt = opt or Options()
    flow = layout.flow
    an = analyze(layout)
    src, page, regions, m, paras = an.src, an.page, an.regions, an.metrics, an.paragraphs
    masked_text_ink = an.cut_glyph_px
    covered_px = an.covered_glyph_px
    cw = m.column_width
    para_of = {l.id: p for p in paras for l in p.lines}
    by_index = {r.index: r for r in regions}
    report: dict = {
        "metrics": m.__dict__,
        "removed": [],
        "repacked": [],
        "notes": [],
    }

    bottom: list[Item] = []

    def make_items(include_photos: bool) -> tuple[list[Item], list[Item], list[Item]]:
        items: list[Item] = []
        top_blocks: list[Item] = []
        pre_title: list[Item] = []
        bottom.clear()
        emitted: set[int] = set()
        seen_text = False
        for i, f in enumerate(flow):
            if f.type in ("block", "image") and f.kind != "portrait" and layout.title and f.rect.y1 <= layout.title.y0:
                # 제목 위에 있던 요소(그래픽 띠 등)는 제목 위에 그대로 둔다
                pre_title.append(Item(f.type, src.crop_rgb(page.clip(f.rect)), [f"{'B' if f.type == 'block' else 'I'}{i}"], label=f.label or f.kind))
                continue
            if f.type == "text" and i in by_index:
                for l in by_index[i].lines:
                    p = para_of[l.id]
                    if p.repack:
                        if p.index in emitted:
                            continue
                        emitted.add(p.index)
                        for pl in repack(p, m, layout.joins, opt.join_default):
                            items.append(
                                Item("packed", render_packed(page, pl, cw), [a.id for a, _ in pl.atoms], para=p.index)
                            )
                    else:
                        items.append(Item("line", render_line(page, l, cw), [l.id], para=p.index))
                    seen_text = True
            elif f.type == "block":
                img = src.crop_rgb(page.clip(f.rect))
                label = f.label or "block"
                if img.shape[1] > cw and not seen_text:
                    top_blocks.append(Item("block", img, [f"B{i}"], label=label))
                    continue
                img, scaled = _fit_width(img, cw)
                if scaled:
                    report["notes"].append(f"{label}(flow[{i}])이 단 폭보다 넓어 단 폭에 맞춰 축소했습니다.")
                pad = m.pitch // 2
                items.append(Item("block", _pad_block(page, img, cw, pad), [f"B{i}"], label=label, pad=pad))
            elif f.type == "image":
                r = page.clip(f.rect)
                if f.kind == "portrait":
                    continue
                if f.kind == "photo" and not include_photos:
                    continue
                if r.w >= opt.wide_ratio * cw:
                    # 여러 단에 걸친 넓은 그림은 단 안에 줄여 넣지 않고 3단 아래에 전체 폭으로 둔다
                    bottom.append(Item("image", src.crop_rgb(r), [f"I{i}"], label=f.kind))
                    continue
                img, scaled = _fit_width(src.crop_rgb(r), cw)
                if scaled and include_photos and r.w > 1.6 * cw:
                    report["notes"].append(
                        f"{f.kind}(flow[{i}])을 단 폭에 맞춰 {cw / r.w:.0%}로 축소 — 사진 설명 글자도 함께 작아짐"
                    )
                pad = m.pitch // 2
                items.append(
                    Item("image", _pad_block(page, img, cw, pad, align="center"), [f"I{i}"], label=f.kind, pad=pad)
                )
        return float_to_boundaries(items), top_blocks, pre_title

    # Title
    char_h = m.char_h
    margin = int(opt.margin * char_h)
    gutter = int(opt.gutter * char_h)
    W = 2 * margin + opt.columns * cw + (opt.columns - 1) * gutter
    title: TitleResult | None = None
    if layout.title:
        title = build_title(page, layout.title, char_h, W - 2 * margin, opt.title_area_ratio, layout.title_break)
        report["notes"].extend(title.notes)

    header = strip_rules(src.crop_rgb(page.clip(layout.header)), page.threshold) if layout.header else None
    if header is not None:
        header = regroup_wide(header, page, W - 2 * margin, char_h)
        header, s = _fit_width(header, W - 2 * margin)
        if s:
            report["notes"].append("상단 매체명·날짜·면 표시가 3단 폭보다 넓어 축소했습니다.")
    footer = strip_rules(src.crop_rgb(page.clip(layout.footer)), page.threshold) if layout.footer else None
    if footer is not None:
        footer, _ = _fit_width(footer, W - 2 * margin)

    def assemble(include_photos: bool):
        items, top_blocks, pre_title = make_items(include_photos)
        cols = pack_columns(items, opt.columns)
        body_h = max((sum(it.h for it in c) for c in cols), default=0)
        y = margin
        layout_plan: list[tuple[str, np.ndarray, int, int]] = []
        if header is not None:
            layout_plan.append(("header", header, margin, y))
            y += header.shape[0] + char_h // 2
        for b in pre_title:
            img, s_ = _fit_width(b.img, W - 2 * margin)
            layout_plan.append(("pretitle:" + b.label, img, (W - img.shape[1]) // 2, y))
            y += img.shape[0] + char_h // 2
        rule_y = None
        if opt.rule and (header is not None or title is not None):
            rule_y = y
            y += 2
        if title is not None:
            line_h = max(t.shape[0] for t in title.lines)
            gap = int(0.12 * line_h)
            pad = int(0.45 * line_h)
            block_h = sum(t.shape[0] for t in title.lines) + gap * (len(title.lines) - 1)
            y += pad
            for t in title.lines:
                layout_plan.append(("title", t, (W - t.shape[1]) // 2, y))
                y += t.shape[0] + gap
            y += pad - gap
            title_box = (rule_y, y, block_h, pad)
        else:
            title_box = None
        for b in top_blocks:
            img, _ = _fit_width(b.img, W - 2 * margin)
            layout_plan.append(("topblock:" + b.label, img, margin, y))
            y += img.shape[0] + m.pitch // 2
        body_y = y
        y += body_h
        for b in bottom:
            img, s_ = _fit_width(b.img, W - 2 * margin)
            y += m.pitch
            layout_plan.append(("bottom:" + b.label, img, (W - img.shape[1]) // 2, y))
            y += img.shape[0]
        if footer is not None:
            y += m.pitch
            fx = margin
            if layout.footer and (layout.footer.x0 + layout.footer.x1) / 2 > page.w / 2:
                fx = W - margin - footer.shape[1]
            layout_plan.append(("footer", footer, fx, y))
            y += footer.shape[0]
        H = y + margin
        return items, top_blocks, cols, layout_plan, body_y, H, rule_y, title_box

    has_photo = any(f.type == "image" and f.kind == "photo" for f in flow)
    state = assemble(True)
    if has_photo and state[5] / W > opt.max_height_ratio:
        for i, f in enumerate(flow):
            if f.type == "image" and f.kind == "photo":
                report["removed"].append(
                    {"flow": i, "kind": "photo", "rect": f.rect.as_list(),
                     "reason": f"사진 포함 시 편집본 세로/가로 {state[5] / W:.2f} > {opt.max_height_ratio} (길어짐)"}
                )
        state = assemble(False)
    elif has_photo:
        report["notes"].append(f"사진 포함 편집본 세로/가로 {state[5] / W:.2f} ≤ {opt.max_height_ratio} → 사진 유지")
    for i, f in enumerate(flow):
        if f.type == "image" and f.kind == "portrait":
            report["removed"].append({"flow": i, "kind": "portrait", "rect": f.rect.as_list(), "reason": "본문이 윤곽을 따라 감싼 인물사진"})

    items, top_blocks, cols, plan, body_y, H, rule_y, title_box = state
    canvas = np.empty((H, W, 3), dtype=np.uint8)
    canvas[:] = page.paper_rgb
    for _, img, x, y in plan:
        canvas[y : y + img.shape[0], x : x + img.shape[1]] = img
    if rule_y is not None:
        canvas[rule_y : rule_y + 2, margin : W - margin] = 40
    placements: list[Placement] = []
    for ci, col in enumerate(cols):
        x = margin + ci * (cw + gutter)
        y = body_y
        for it in col:
            w = min(it.img.shape[1], W - x)
            canvas[y : y + it.h, x : x + w] = it.img[:, :w]
            placements.append(Placement(it, x, y, ci))
            y += it.h

    for b in bottom:
        scale = min(1.0, (W - 2 * margin) / b.img.shape[1])
        report["notes"].append(
            f"넓은 {b.label}({b.ids[0]})은 3단 본문 아래에 전체 폭으로 배치 ({scale:.0%} 크기)"
        )
    report["layout"] = {
        "width": W, "height": H, "column_width": cw, "gutter": gutter, "columns": opt.columns,
        "column_heights": [sum(it.h for it in c) for c in cols],
        "column_lines": [sum(1 for it in c if it.kind in ("line", "packed")) for c in cols],
        "height_ratio": round(H / W, 3),
    }
    if title is not None:
        rule, end, block_h, pad = title_box
        report["title"] = {
            "break_after_atom": title.break_after,
            "break_reason": title.break_reason,
            "lines": [ids for ids in title.atom_ids],
            "area_ratio": round(title.area_ratio, 2),
            "vertical_padding_top_bottom": [pad, pad],
        }
    for p in paras:
        if p.repack:
            out_lines = [pl for pl in placements if pl.item.kind == "packed" and pl.item.para == p.index]
            report["repacked"].append(
                {
                    "paragraph": p.index,
                    "source_lines": [l.id for l in p.lines],
                    "source_line_count": len(p.lines),
                    "wrapped_lines": [l.id for l in p.lines if l.wrap],
                    "output_line_count": len(out_lines),
                    "output_position": [
                        {"column": pl.column + 1, "y": pl.y} for pl in out_lines[:1]
                    ],
                    "joins_with_space": [f"{a}→{b}" for a, b, s in p.joins if s],
                    "joins_without_space": [f"{a}→{b}" for a, b, s in p.joins if not s],
                }
            )
    report["verification"] = verify(page, regions, paras, placements, canvas, title, layout)
    report["verification"]["ink_text_under_images_px"] = masked_text_ink
    report["verification"]["glyphs_under_portrait_px"] = covered_px
    if covered_px > 40:
        report["verification"]["ok"] = False
    if masked_text_ink > max(20, 0.002 * report["verification"]["ink_source_px"]):
        report["verification"]["ok"] = False
    image = Image.fromarray(canvas)
    return Result(
        image=image,
        annotated=annotate(image, placements, cols, m),
        report=report,
        joins=joins_sheet(page, paras, m),
        page=page,
        paras=paras,
    )


def joins_sheet(page: Page, paras, m: Metrics) -> Image.Image | None:
    """Contact sheet of every line join inside re-packed paragraphs, to check spacing by eye."""
    rows = []
    for p in paras:
        if not p.repack:
            continue
        by_id = {l.id: l for l in p.lines}
        for a, b, space in p.joins:
            la, lb = by_id[a], by_id[b]
            if not la.atoms or not lb.atoms:
                continue
            ta = la.atoms[-6:]
            tb = lb.atoms[:6]
            left = page.crop_rgb(Rect(ta[0].x0, la.band.y0, ta[-1].x1, la.band.y1))
            right = page.crop_rgb(Rect(tb[0].x0, lb.band.y0, tb[-1].x1, lb.band.y1))
            rows.append((f"{a} > {b} : {'SPACE' if space else 'JOIN'}", left, right))
    if not rows:
        return None
    label_w = 16 * 8 + 20
    sep = 3 * m.char_h
    h = max(max(l.shape[0], r.shape[0]) for _, l, r in rows)
    w = label_w + max(l.shape[1] for _, l, _ in rows) + sep + max(r.shape[1] for _, _, r in rows) + 20
    sheet = Image.new("RGB", (w, len(rows) * (h + 6) + 6), page.paper_rgb)
    d = ImageDraw.Draw(sheet)
    lw = max(l.shape[1] for _, l, _ in rows)
    for i, (label, l, r) in enumerate(rows):
        y = 6 + i * (h + 6)
        d.text((6, y + h // 3), label, fill=(200, 0, 0) if "SPACE" in label else (0, 0, 200))
        sheet.paste(Image.fromarray(l), (label_w + lw - l.shape[1], y))
        x = label_w + lw + sep // 2
        d.line([(x, y), (x, y + h)], fill=(200, 200, 200))
        sheet.paste(Image.fromarray(r), (label_w + lw + sep, y))
    return sheet


def _ink_parts(ink: np.ndarray, img_mask: np.ndarray, text_mask: np.ndarray):
    import cv2

    n, lab = cv2.connectedComponents((ink & text_mask).astype(np.uint8), connectivity=8)
    inside = np.bincount(lab[img_mask & text_mask].ravel(), minlength=n)
    outside = np.bincount(lab[~img_mask & text_mask].ravel(), minlength=n)
    inside[0] = outside[0] = 0
    return lab, inside, outside


def absorb_image_ink(ink: np.ndarray, img_mask: np.ndarray, text_mask: np.ndarray) -> np.ndarray:
    """Grow the image mask over ink blobs that lie mostly inside it (photo edges, not glyphs)."""
    lab, inside, outside = _ink_parts(ink, img_mask, text_mask)
    mostly = (inside > 0) & (inside >= outside)
    return img_mask | (mostly[lab] & (lab > 0))


def cut_glyph_ink(ink: np.ndarray, img_mask: np.ndarray, text_mask: np.ndarray) -> int:
    """Ink of glyphs that straddle an image mask edge inside text regions (= glyph cut by the image box)."""
    _, inside, outside = _ink_parts(ink, img_mask, text_mask)
    cut = (inside > 0) & (outside > 0)
    return int(inside[cut].sum())


def verify(page: Page, regions, paras, placements: list[Placement], canvas: np.ndarray, title, layout: Layout) -> dict:
    expected_lines = {l.id: l for r in regions for l in r.lines}
    repacked_atoms = {a.id for p in paras if p.repack for l in p.lines for a in l.atoms}
    count: dict[str, int] = {}
    for pl in placements:
        if pl.item.kind in ("line", "packed"):
            for i in pl.item.ids:
                count[i] = count.get(i, 0) + 1
    missing, dup = [], []
    for lid, l in expected_lines.items():
        if l.atoms and l.atoms[0].id in repacked_atoms:
            for a in l.atoms:
                c = count.get(a.id, 0)
                if c == 0:
                    missing.append(a.id)
                elif c > 1:
                    dup.append(a.id)
        else:
            c = count.get(lid, 0)
            if c == 0:
                missing.append(lid)
            elif c > 1:
                dup.append(lid)

    # Ink conservation: ink cut from the page vs ink found in the pasted text strips.
    src_ink = 0
    for l in expected_lines.values():
        b = l.band
        src_ink += int(page.ink[max(0, b.y0) : b.y1, b.x0 : b.x1].sum())
    out_ink = 0
    for pl in placements:
        if pl.item.kind in ("line", "packed"):
            strip = canvas[pl.y : pl.y + pl.item.h, pl.x : pl.x + pl.item.img.shape[1]]
            out_ink += int(to_ink(strip, page.threshold).sum())
    # Ink inside text regions that no line band picked up (would mean a cut-off glyph).
    uncovered = 0
    for r in regions:
        mask = page.ink[r.rect.y0 : r.rect.y1, r.rect.x0 : r.rect.x1].copy()
        for l in r.lines:
            b = l.band
            y0, y1 = max(b.y0, r.rect.y0) - r.rect.y0, min(b.y1, r.rect.y1) - r.rect.y0
            if y1 > y0:
                mask[y0:y1, :] = False
        uncovered += int(mask.sum())

    title_check = None
    if title is not None:
        used = [i for row in title.atom_ids for i in row]
        title_check = {
            "atoms": len(used),
            "unique": len(set(used)),
            "ok": len(used) == len(set(used)),
        }
    diff = out_ink - src_ink
    ok = not missing and not dup and abs(diff) <= max(20, 0.002 * src_ink) and uncovered <= max(20, 0.002 * src_ink)
    res = {
        "ok": bool(ok and (title_check is None or title_check["ok"])),
        "source_lines": len(expected_lines),
        "missing": missing,
        "duplicated": dup,
        "ink_source_px": src_ink,
        "ink_output_px": out_ink,
        "ink_diff_px": diff,
        "ink_uncovered_in_regions_px": uncovered,
        "title": title_check,
    }
    res["ocr"] = ocr_check(page, regions, placements, canvas)
    return res


def ocr_check(page: Page, regions, placements, canvas) -> dict | None:
    try:
        import pytesseract  # type: ignore
    except Exception:
        return {"available": False, "note": "pytesseract/tesseract(kor) 미설치 → OCR 대조 생략 (픽셀 장부 검증만 수행)"}
    try:
        def ocr(img: np.ndarray) -> str:
            return pytesseract.image_to_string(Image.fromarray(img), lang="kor+eng", config="--psm 6")

        src = "".join(ocr(page.crop_rgb(r.rect)) for r in regions)
        out_parts = []
        for ci in sorted({p.column for p in placements}):
            strips = [p for p in placements if p.column == ci and p.item.kind in ("line", "packed")]
            if not strips:
                continue
            y0 = min(p.y for p in strips)
            y1 = max(p.y + p.item.h for p in strips)
            x0 = strips[0].x
            out_parts.append(ocr(canvas[y0:y1, x0 : x0 + strips[0].item.img.shape[1]]))
        out = "".join(out_parts)
        norm = lambda s: [c for c in s if not c.isspace()]  # noqa: E731
        from collections import Counter

        a, b = Counter(norm(src)), Counter(norm(out))
        return {
            "available": True,
            "source_chars": sum(a.values()),
            "output_chars": sum(b.values()),
            "only_in_source": dict((a - b).most_common(30)),
            "only_in_output": dict((b - a).most_common(30)),
            "note": "OCR 오인식이 섞이므로 참고용. 확정 판정은 픽셀 장부 검증 기준.",
        }
    except Exception as e:  # tesseract binary/lang missing
        return {"available": False, "note": f"OCR 실패: {e}"}


def annotate(image: Image.Image, placements: list[Placement], cols, m: Metrics) -> Image.Image:
    img = image.convert("RGB").copy()
    d = ImageDraw.Draw(img, "RGBA")
    for pl in placements:
        it = pl.item
        box = [pl.x, pl.y, pl.x + it.img.shape[1] - 1, pl.y + it.h - 1]
        if it.kind == "packed":
            d.rectangle(box, fill=(255, 140, 0, 50))
        elif it.kind == "image":
            d.rectangle(box, outline=(0, 160, 0, 255), width=3)
        elif it.kind == "block":
            d.rectangle(box, outline=(0, 90, 255, 255), width=2)
    seen = set()
    for pl in placements:
        if pl.item.kind == "packed" and pl.item.para not in seen:
            seen.add(pl.item.para)
            d.text((pl.x + 2, pl.y), f"P{pl.item.para}", fill=(220, 60, 0, 255))
    for ci in range(len(cols)):
        col = [p for p in placements if p.column == ci]
        if col:
            d.text((col[0].x, col[0].y - 12), f"{ci + 1}", fill=(200, 0, 0, 255))
    return img
