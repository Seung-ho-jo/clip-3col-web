"""Layout description (JSON) of one clipped article.

The JSON is produced by ``detect`` and may be corrected by hand before ``build``.

{
  "image": "page.jpg",
  "header": [x0, y0, x1, y1] | null,     # 매체명·날짜·면 표시 (그대로 둠)
  "footer": [x0, y0, x1, y1] | null,     # 크기 표기 (그대로 둠)
  "title":  [x0, y0, x1, y1] | null,     # 제목 (두 줄로 다시 짬)
  "title_break": null | int,             # 수동 끊김 위치(제목 글자 조각 번호, 이 조각 뒤에서 끊음)
  "column_width": null | int,            # 본문 단 폭(px). null 이면 자동
  "joins": {"R3L2": true},               # 다시 짠 문단의 줄 이음 수동 지정: 그 줄 끝 뒤를 띄움(true)/붙임(false)
  "flow": [                              # 읽는 순서대로
    {"type": "block", "label": "series",   "rect": [...]},   # 시리즈 바 (그대로)
    {"type": "block", "label": "subtitle", "rect": [...]},   # 부제 (그대로)
    {"type": "text",  "rect": [...], "wrap": false},          # 본문 영역
    {"type": "text",  "rect": [...], "wrap": true},           # 사진 때문에 짧아진 줄 영역
    {"type": "image", "kind": "graphic" | "photo" | "portrait", "rect": [...]},
    {"type": "block", "label": "byline",   "rect": [...]}    # 기자명 (그대로)
  ]
}

kind:
  graphic  = 도표·그래픽 (항상 넣음)
  photo    = 일반 사진 (편집본이 길어지면 뺌)
  portrait = 본문이 윤곽을 따라 감싼 인물사진 (항상 뺌)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .imageops import Rect

FLOW_TYPES = {"text", "block", "image"}
IMAGE_KINDS = {"graphic", "photo", "portrait"}


@dataclass
class FlowItem:
    type: str
    rect: Rect
    label: str = ""
    kind: str = ""
    wrap: bool | None = None
    note: str = ""

    def to_json(self) -> dict:
        d: dict = {"type": self.type, "rect": self.rect.as_list()}
        if self.label:
            d["label"] = self.label
        if self.type == "image":
            d["kind"] = self.kind
        if self.type == "text" and self.wrap is not None:
            d["wrap"] = self.wrap
        if self.note:
            d["note"] = self.note
        return d


@dataclass
class Layout:
    image: str
    header: Rect | None = None
    footer: Rect | None = None
    title: Rect | None = None
    title_break: int | None = None
    column_width: int | None = None
    flow: list[FlowItem] = field(default_factory=list)
    joins: dict[str, bool] = field(default_factory=dict)

    def to_json(self) -> dict:
        r = lambda v: v.as_list() if v else None  # noqa: E731
        return {
            "image": self.image,
            "header": r(self.header),
            "footer": r(self.footer),
            "title": r(self.title),
            "title_break": self.title_break,
            "column_width": self.column_width,
            "flow": [f.to_json() for f in self.flow],
            "joins": self.joins,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, d: dict) -> "Layout":
        opt = lambda v: Rect.of(v) if v else None  # noqa: E731
        flow = []
        for i, f in enumerate(d.get("flow", [])):
            t = f.get("type")
            if t not in FLOW_TYPES:
                raise ValueError(f"flow[{i}]: unknown type {t!r}")
            kind = f.get("kind", "photo" if t == "image" else "")
            if t == "image" and kind not in IMAGE_KINDS:
                raise ValueError(f"flow[{i}]: unknown image kind {kind!r}")
            flow.append(
                FlowItem(
                    type=t,
                    rect=Rect.of(f["rect"]),
                    label=f.get("label", ""),
                    kind=kind,
                    wrap=f.get("wrap"),
                    note=f.get("note", ""),
                )
            )
        return cls(
            image=d["image"],
            header=opt(d.get("header")),
            footer=opt(d.get("footer")),
            title=opt(d.get("title")),
            title_break=d.get("title_break"),
            column_width=d.get("column_width"),
            flow=flow,
            joins={str(k): bool(v) for k, v in (d.get("joins") or {}).items()},
        )

    @classmethod
    def load(cls, path: str | Path) -> "Layout":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))
