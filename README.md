# 지면 3단 편집기 (웹)

지면 기사 이미지를 올리면 원본 글자를 오려 붙여 3단 기사로 다시 짜는 웹페이지입니다.
모든 처리는 브라우저 안에서 이루어지며, 올린 이미지는 어디에도 전송되지 않습니다.

- 편집 프로그램: `py/clip_editor/` (원본 저장소의 `clip_editor` 패키지)
- 브라우저용 Python 실행기: [Pyodide](https://pyodide.org) 0.27.7 (MPL-2.0), 함께 실린 패키지 numpy (BSD-3-Clause), OpenCV (Apache-2.0), Pillow (MIT-CMU)
- `example/` 의 지면은 시험용으로 만든 가짜 기사입니다.
