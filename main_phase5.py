"""DimensionX Phase 5: local WebGL viewer and pipeline trigger server."""

from __future__ import annotations

import json
import mimetypes
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from src.pipeline import GLB_PATH, INPUT_DIR, OUTPUT_DIR, run_full_pipeline

ROOT = Path(__file__).resolve().parent
VIEWER_DIR = ROOT / "viewer"
HOST = "127.0.0.1"
PORT = 8000

_state_lock = threading.Lock()
_pipeline_state = {
    "status": "Ready" if GLB_PATH.is_file() else "Waiting for model",
    "busy": False,
    "error": None,
    "result": None,
}


def _set_status(message: str) -> None:
    with _state_lock:
        _pipeline_state["status"] = message
        if message == "Ready":
            _pipeline_state["error"] = None


def _state_snapshot() -> dict:
    with _state_lock:
        return dict(_pipeline_state)


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "DimensionXPhase5/1.0"

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        print(f"[phase5] {self.address_string()} - {format % args}")

    def do_GET(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        if path in {"/", "/index.html"}:
            self._serve_file(VIEWER_DIR / "index.html", "text/html; charset=utf-8")
            return
        if path == "/api/status":
            self._json(_state_snapshot())
            return
        if path.startswith("/data/"):
            rel = path[len("/data/") :]
            target = (ROOT / "data" / rel).resolve()
            data_root = (ROOT / "data").resolve()
            if not str(target).startswith(str(data_root)) or not target.is_file():
                self.send_error(404, "File not found")
                return
            ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            if target.suffix.lower() == ".glb":
                ctype = "model/gltf-binary"
            self._serve_file(target, ctype)
            return
        if path.startswith("/viewer/"):
            rel = path[len("/viewer/") :]
            target = (VIEWER_DIR / rel).resolve()
            if not str(target).startswith(str(VIEWER_DIR.resolve())) or not target.is_file():
                self.send_error(404, "File not found")
                return
            ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            self._serve_file(target, ctype)
            return
        self.send_error(404, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        if path != "/api/process":
            self.send_error(404, "Not found")
            return

        with _state_lock:
            if _pipeline_state["busy"]:
                self._json({"error": "Pipeline is already running. Please wait."}, status=409)
                return
            _pipeline_state["busy"] = True
            _pipeline_state["error"] = None
            _pipeline_state["status"] = "Uploading image..."

        try:
            saved = self._save_upload()
            _set_status("Preprocessing image...")
            result = run_full_pipeline(saved, on_status=_set_status)
            with _state_lock:
                _pipeline_state["result"] = result
                _pipeline_state["status"] = "Ready"
                _pipeline_state["busy"] = False
            self._json(result)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            with _state_lock:
                _pipeline_state["busy"] = False
                _pipeline_state["error"] = str(exc)
                _pipeline_state["status"] = f"Error: {exc}"
            self._json({"error": str(exc)}, status=500)

    def _save_upload(self) -> Path:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            raise ValueError("Empty upload.")
        body = self.rfile.read(length)
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            raise ValueError("Expected multipart/form-data upload.")

        boundary = None
        for part in content_type.split(";"):
            part = part.strip()
            if part.startswith("boundary="):
                boundary = part.split("=", 1)[1].strip().strip('"')
                break
        if not boundary:
            raise ValueError("Missing multipart boundary.")

        marker = b"--" + boundary.encode("utf-8")
        chunks = body.split(marker)
        filename = "upload.png"
        payload = None
        for chunk in chunks:
            if b"Content-Disposition" not in chunk:
                continue
            header, _, data = chunk.partition(b"\r\n\r\n")
            if not data:
                continue
            if data.endswith(b"\r\n"):
                data = data[:-2]
            if data.endswith(b"--"):
                data = data[:-2]
            if data.endswith(b"\r\n"):
                data = data[:-2]
            header_text = header.decode("utf-8", errors="ignore")
            if 'name="file"' not in header_text and "filename=" not in header_text:
                continue
            for token in header_text.replace("\r", "").split(";"):
                token = token.strip()
                if token.startswith("filename="):
                    filename = token.split("=", 1)[1].strip().strip('"') or filename
            payload = data
            break

        if payload is None:
            raise ValueError("No file field found in upload.")

        safe_name = Path(filename).name
        suffix = Path(safe_name).suffix.lower() or ".png"
        if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
            suffix = ".png"
            safe_name = f"{Path(safe_name).stem}{suffix}"

        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        dest = INPUT_DIR / f"upload{suffix}"
        dest.write_bytes(payload)
        print(f"[phase5] saved upload -> {dest} ({len(payload)} bytes)")
        return dest

    def _serve_file(self, path: Path, content_type: str) -> None:
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, payload: dict, status: int = 200) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def main() -> None:
    if not VIEWER_DIR.is_dir():
        raise SystemExit(f"Viewer directory missing: {VIEWER_DIR}")

    server = ThreadingHTTPServer((HOST, PORT), ViewerHandler)
    url = f"http://{HOST}:{PORT}"
    print("=== Phase 5: Web Viewer & Interactive Virtual Try-On ===")
    print(f"Viewer UI:     {url}")
    print(f"GLB asset:     {url}/data/output/final_garment.glb")
    print(f"Upload API:    POST {url}/api/process")
    print(f"Status API:    GET  {url}/api/status")
    if GLB_PATH.is_file():
        size_mb = GLB_PATH.stat().st_size / (1024 * 1024)
        print(f"Existing GLB:  {GLB_PATH} ({size_mb:.3f} MB)")
    else:
        print("Existing GLB:  not found — upload an image to generate one.")
    print("Press Ctrl+C to stop the server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Phase 5 server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
