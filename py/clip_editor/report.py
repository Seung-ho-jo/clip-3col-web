"""Human-readable (Korean) report of what was removed, re-packed and verified."""

from __future__ import annotations


def to_markdown(rep: dict, out_name: str) -> str:
    L = []
    lay = rep["layout"]
    v = rep["verification"]
    L.append("# 지면 기사 3단 편집 보고")
    L.append("")
    L.append(f"- 결과 이미지: `{out_name}` ({lay['width']}×{lay['height']}px, 세로/가로 {lay['height_ratio']})")
    L.append(f"- 단 폭 {lay['column_width']}px (원본 본문 단 폭 그대로), 단 사이 {lay['gutter']}px")
    L.append(
        "- 단별 줄 수: "
        + " / ".join(f"{i + 1}단 {n}줄" for i, n in enumerate(lay["column_lines"]))
        + "  (1단 → 2단 → 3단 순으로 이어짐)"
    )
    m = rep["metrics"]
    L.append(f"- 본문 글자 높이 {m['char_h']}px, 줄 간격 {m['pitch']}px — 원본 그대로 사용(확대·축소 없음)")
    if "title" in rep:
        t = rep["title"]
        L.append(
            f"- 제목: {len(t['lines'])}줄, 끊은 위치 = {t['break_reason']}, 글자 면적 본문 대비 {t['area_ratio']}배, "
            f"위쪽 선·본문 사이 여백 위 {t['vertical_padding_top_bottom'][0]}px / 아래 {t['vertical_padding_top_bottom'][1]}px"
        )
    L.append("")
    L.append("## 뺀 것")
    if rep["removed"]:
        for r in rep["removed"]:
            L.append(f"- flow[{r['flow']}] {r['kind']} {r['rect']} — {r['reason']}")
    else:
        L.append("- 없음")
    L.append("")
    L.append("## 다시 짠 문단 (사진 때문에 짧아진 줄)")
    if rep["repacked"]:
        for p in rep["repacked"]:
            pos = p["output_position"][0] if p["output_position"] else {}
            L.append(
                f"- 문단 P{p['paragraph']}: 원본 {p['source_line_count']}줄(그중 짧은 줄 {len(p['wrapped_lines'])}개: "
                f"{', '.join(p['wrapped_lines'])}) → 꽉 찬 줄 {p['output_line_count']}줄, "
                f"편집본 {pos.get('column', '?')}단 y={pos.get('y', '?')}px 부터"
            )
            if p["joins_without_space"] or p["joins_with_space"]:
                L.append(
                    f"  - 줄 이음(확인 권장): 붙여 이음 {len(p['joins_without_space'])}곳, 띄어 이음 {len(p['joins_with_space'])}곳 "
                    f"— 원본 줄끝이 단어 중간인지 띄어쓰기인지 이미지로는 확정할 수 없어, 문장부호 뒤만 띄움"
                )
        L.append("- `*_annotated.jpg` 에서 주황색 칸이 다시 짠 줄입니다.")
    else:
        L.append("- 없음 (모든 줄을 원본 줄 그대로 오려 붙임)")
    L.append("")
    L.append("## 기타")
    for n in rep["notes"] or ["- 없음"]:
        L.append(n if n.startswith("- ") else f"- {n}")
    L.append("")
    L.append("## 글자 누락·중복 검증")
    L.append(f"- 판정: **{'통과' if v['ok'] else '확인 필요'}**")
    L.append(f"- 원본 본문 줄 {v['source_lines']}개 — 누락 {len(v['missing'])}, 중복 {len(v['duplicated'])}")
    if v["missing"]:
        L.append(f"  - 누락: {', '.join(v['missing'][:40])}")
    if v["duplicated"]:
        L.append(f"  - 중복: {', '.join(v['duplicated'][:40])}")
    L.append(
        f"- 잉크(글자 픽셀) 보존: 원본 {v['ink_source_px']:,}px → 편집본 {v['ink_output_px']:,}px (차이 {v['ink_diff_px']:+,}px)"
    )
    L.append(f"- 본문 영역 안에서 어떤 줄에도 안 잡힌 잉크: {v['ink_uncovered_in_regions_px']:,}px (0에 가까워야 정상)")
    if "ink_text_under_images_px" in v:
        L.append(f"- 사진·도표 경계에 걸려 잘린 글자 픽셀: {v['ink_text_under_images_px']:,}px (0이어야 정상 — 아니면 layout JSON 의 image rect 를 줄이세요)")
    if "glyphs_under_portrait_px" in v:
        L.append(f"- 인물사진 마스크에 덮인 글자 추정 픽셀: {v['glyphs_under_portrait_px']:,}px (0이어야 정상)")
    if v.get("title"):
        L.append(f"- 제목 글자 조각 {v['title']['atoms']}개 사용, 중복 {'없음' if v['title']['ok'] else '있음'}")
    o = v.get("ocr") or {}
    if o.get("available"):
        L.append(f"- OCR 대조(참고): 원본 {o['source_chars']}자 / 편집본 {o['output_chars']}자")
        if o["only_in_source"]:
            L.append(f"  - 원본에만: {o['only_in_source']}")
        if o["only_in_output"]:
            L.append(f"  - 편집본에만: {o['only_in_output']}")
    else:
        L.append(f"- OCR 대조: {o.get('note', '생략')}")
    L.append("")
    return "\n".join(L)
