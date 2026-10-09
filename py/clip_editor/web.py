"""Entry points for the in-browser (Pyodide) web page.

  import clip_editor.web as w
  out = w.process(image_bytes, "page.jpg")   # detect + build
  out = w.rebuild({"R6L8": True})             # build again with join overrides

Every image comes back as bytes so the page can show it and offer it as a file.
"""

from __future__ import annotations

import io
import json
import os
import tempfile

from PIL import Image

from .compose import Options, build
from .detect import detect
from .report import to_markdown

_state: dict = {}


def _jpg(img: Image.Image, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality, subsampling=0)
    return buf.getvalue()


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _joins(res, layout) -> list[dict]:
    """One entry per line join inside re-packed paragraphs, with the two crops to compare."""
    from .imageops import Rect

    out = []
    page = res.page
    for p in res.paras or []:
        if not p.repack:
            continue
        by_id = {l.id: l for l in p.lines}
        for a, b, space in p.joins:
            la, lb = by_id[a], by_id[b]
            if not la.atoms or not lb.atoms:
                continue
            ta, tb = la.atoms[-6:], lb.atoms[:6]
            left = page.crop_rgb(Rect(ta[0].x0, la.band.y0, ta[-1].x1, la.band.y1))
            right = page.crop_rgb(Rect(tb[0].x0, lb.band.y0, tb[-1].x1, lb.band.y1))
            out.append(
                {
                    "line": a,
                    "next": b,
                    "space": bool(space),
                    "manual": a in layout.joins,
                    "left": _png(Image.fromarray(left)),
                    "right": _png(Image.fromarray(right)),
                }
            )
    return out


def _run(layout, preview: Image.Image | None) -> dict:
    res = build(layout, Options())
    rep = res.report
    joins = _joins(res, layout)
    md = to_markdown(rep, _state["stem"] + "_3col.jpg")
    out = {
        "result": _jpg(res.image, 95),
        "annotated": _jpg(res.annotated, 85),
        "report_md": md,
        "report_json": json.dumps(rep, ensure_ascii=False, indent=2),
        "layout_json": json.dumps(layout.to_json(), ensure_ascii=False, indent=2),
        "ok": bool(rep["verification"]["ok"]),
        "size": [res.image.width, res.image.height],
        "joins": joins,
        "summary": {
            "removed": len(rep["removed"]),
            "repacked": len(rep["repacked"]),
            "lines": rep["verification"]["source_lines"],
            "missing": len(rep["verification"]["missing"]),
            "duplicated": len(rep["verification"]["duplicated"]),
        },
    }
    if preview is not None:
        out["detect"] = _jpg(preview, 85)
    return out


def process(image_bytes: bytes, filename: str = "page.jpg") -> dict:
    if hasattr(image_bytes, "to_py"):  # Pyodide: JS Uint8Array → memoryview
        image_bytes = image_bytes.to_py()
    stem = os.path.splitext(os.path.basename(filename))[0] or "page"
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "page.png")
    Image.open(io.BytesIO(bytes(image_bytes))).convert("RGB").save(path)
    layout, preview = detect(path)
    _state.update(layout=layout, stem=stem)
    return _run(layout, preview)


def rebuild(joins: dict) -> dict:
    if hasattr(joins, "to_py"):
        joins = joins.to_py()
    layout = _state["layout"]
    layout.joins = {str(k): bool(v) for k, v in (joins or {}).items()}
    return _run(layout, None)
