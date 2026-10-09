// Runs the clip_editor Python package inside the browser (Pyodide), off the page's main thread.
const PY_FILES = [
  "__init__.py", "imageops.py", "layout.py", "textflow.py", "title.py",
  "detect.py", "compose.py", "report.py", "web.py",
];
const base = new URL("./", self.location.href).href;
let py = null;
let ready = null;

const status = (text, step) => self.postMessage({ type: "status", text, step });

function toPlain(proxy) {
  const out = proxy.toJs({ dict_converter: Object.fromEntries, create_pyproxies: false });
  proxy.destroy();
  return out;
}

function collectBuffers(obj, list = []) {
  if (obj instanceof Uint8Array) list.push(obj.buffer);
  else if (Array.isArray(obj)) obj.forEach((v) => collectBuffers(v, list));
  else if (obj && typeof obj === "object") Object.values(obj).forEach((v) => collectBuffers(v, list));
  return list;
}

async function init() {
  status("Python 실행기를 내려받는 중 (처음 한 번, 약 30MB)", 1);
  importScripts(base + "pyodide/pyodide.js");
  py = await loadPyodide({ indexURL: base + "pyodide/" });
  status("이미지 처리 도구를 불러오는 중", 2);
  await py.loadPackage(["numpy", "opencv-python", "pillow"]);
  status("편집 프로그램을 준비하는 중", 3);
  py.FS.mkdirTree("/home/pyodide/clip_editor");
  for (const name of PY_FILES) {
    const res = await fetch(base + "py/clip_editor/" + name);
    if (!res.ok) throw new Error(name + " 파일을 불러오지 못했습니다 (" + res.status + ")");
    py.FS.writeFile("/home/pyodide/clip_editor/" + name, await res.text());
  }
  py.runPython("import sys; sys.path.insert(0, '/home/pyodide'); import clip_editor.web");
  self.postMessage({ type: "ready" });
}

self.onmessage = async (ev) => {
  const msg = ev.data;
  try {
    if (msg.type === "init") {
      ready = ready || init();
      await ready;
      return;
    }
    await ready;
    const web = py.pyimport("clip_editor.web");
    let out;
    if (msg.type === "process") {
      status("지면을 분석하고 3단으로 짜는 중 (보통 10~40초)", 4);
      out = toPlain(web.process(msg.bytes, msg.name));
    } else if (msg.type === "rebuild") {
      status("띄어쓰기를 반영해 다시 짜는 중", 4);
      const joins = py.toPy(msg.joins);
      out = toPlain(web.rebuild(joins));
      joins.destroy();
    }
    web.destroy();
    self.postMessage({ type: "result", data: out, kind: msg.type }, collectBuffers(out));
  } catch (err) {
    self.postMessage({ type: "error", message: String((err && err.message) || err) });
  }
};
