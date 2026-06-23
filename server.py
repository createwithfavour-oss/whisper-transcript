#!/usr/bin/env python3
"""
Whisper Transcript App
----------------------
A tiny, fully-local web app. Drag a video (or audio) in, it:
  1. extracts 16kHz mono audio with ffmpeg,
  2. transcribes with whisper.cpp (whisper-cli) using your local model,
  3. produces an .srt and a timestamped .md you can download.

No internet, no pip installs. Pure Python standard library.

Run:  python3 server.py
Then open the URL it prints (default http://127.0.0.1:8756).
"""

import http.server
import socketserver
import threading
import subprocess
import shutil
import json
import uuid
import os
import sys
import re
import glob
import ssl
import webbrowser
import urllib.request
from urllib.parse import urlparse

# Packaged (PyInstaller) Python has no system CA bundle, so HTTPS verification
# fails ("CERTIFICATE_VERIFY_FAILED"). Ship certifi's CA bundle and use it.
# Built lazily — creating the context can take several seconds, and there's no
# reason to pay that at startup when it's only needed for model downloads.
_SSL_CONTEXT = None


def ssl_context():
    global _SSL_CONTEXT
    if _SSL_CONTEXT is None:
        try:
            import certifi
            _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            _SSL_CONTEXT = ssl.create_default_context()
    return _SSL_CONTEXT

# ----------------------------------------------------------------------------
# Paths — work both as a plain script and when frozen by PyInstaller.
# ----------------------------------------------------------------------------
IS_WIN = os.name == "nt"
FROZEN = getattr(sys, "frozen", False)
if FROZEN:
    # EXE_DIR holds the executable (and the bundled bin/ next to it).
    # RES_DIR is PyInstaller's extracted resource dir (static/, bin/).
    EXE_DIR = os.path.dirname(os.path.abspath(sys.executable))
    RES_DIR = getattr(sys, "_MEIPASS", EXE_DIR)
    # Writable data goes to a per-user app-data dir (Program Files / Applications
    # are not reliably writable).
    if IS_WIN:
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    DATA_HOME = os.path.join(base, "WhisperTranscript")
else:
    EXE_DIR = os.path.dirname(os.path.abspath(__file__))
    RES_DIR = EXE_DIR
    DATA_HOME = EXE_DIR

MODELS_DIR = os.path.join(DATA_HOME, "models")
UPLOADS_DIR = os.path.join(DATA_HOME, "uploads")
OUTPUTS_DIR = os.path.join(DATA_HOME, "outputs")
STATIC_DIR = os.path.join(RES_DIR, "static")
CONFIG_PATH = os.path.join(DATA_HOME, "config.json")
PORT = int(os.environ.get("PORT", "8756"))
HOST = "127.0.0.1"

# Hardening limits.
MAX_UPLOAD_BYTES = 8 * 1024 * 1024 * 1024   # reject uploads larger than 8 GB
MAX_JSON_BYTES = 64 * 1024                   # control-plane JSON bodies are tiny
MAX_JOBS = 50                                # cap in-memory job history
MODEL_MIN_BYTES = 20 * 1024 * 1024           # a real model is bigger than this
# Only answer requests addressed to the loopback host (anti DNS-rebinding).
ALLOWED_HOSTS = {f"127.0.0.1:{PORT}", f"localhost:{PORT}", "127.0.0.1", "localhost"}

for d in (UPLOADS_DIR, OUTPUTS_DIR, MODELS_DIR):
    os.makedirs(d, exist_ok=True)


def _find_binary(name):
    """Locate an external tool, preferring binaries bundled with the app.

    Order: app bin/ -> bundled resources bin/ -> PATH -> common install dirs.
    """
    exe = name + (".exe" if IS_WIN else "")
    for c in (os.path.join(EXE_DIR, "bin", exe),
              os.path.join(RES_DIR, "bin", exe)):
        if os.path.exists(c):
            return c
    found = shutil.which(name) or shutil.which(exe)
    if found:
        return found
    fallbacks = {
        "ffmpeg": ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"],
        "whisper-cli": ["/opt/homebrew/bin/whisper-cli", "/usr/local/bin/whisper-cli"],
    }
    for c in fallbacks.get(name, []):
        if os.path.exists(c):
            return c
    return exe  # last resort: hope it's on PATH


WHISPER_CLI = _find_binary("whisper-cli")
FFMPEG = _find_binary("ffmpeg")


def have_binary(path):
    """True if a resolved binary path exists on disk or is on PATH."""
    return bool(path) and (os.path.exists(path) or shutil.which(path) is not None)


# In-memory job store: job_id -> dict
JOBS = {}
JOBS_LOCK = threading.Lock()

# ----------------------------------------------------------------------------
# Model registry, selection, and on-demand download
# ----------------------------------------------------------------------------
HF_BASE = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/"

# Curated set shown in the first-run picker. size_mb is approximate.
MODEL_CATALOG = [
    {"name": "tiny",           "size_mb": 75,   "label": "Tiny",
     "note": "Fastest, roughest. Runs on any laptop."},
    {"name": "base",           "size_mb": 142,  "label": "Base",
     "note": "Fast, fine for clear speech."},
    {"name": "small",          "size_mb": 466,  "label": "Small",
     "note": "Balanced speed and accuracy."},
    {"name": "large-v3-turbo", "size_mb": 1536, "label": "Large v3 Turbo",
     "note": "Best accuracy, still fast. Recommended."},
]

# Single active download at a time.
DL = {"active": False, "name": None, "pct": 0, "done": False, "error": None}
DL_LOCK = threading.Lock()


def model_filename(name):
    return f"ggml-{name}.bin"


def safe_filename(name, fallback="upload"):
    """Whitelist a user-supplied name down to a safe, separator-free basename."""
    name = os.path.basename(name or "")
    cleaned = re.sub(r"[^A-Za-z0-9 ._-]", "_", name).strip(" .")
    return cleaned or fallback


def _read_config():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_config(cfg):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f)
    except OSError:
        pass


def installed_models():
    """Basenames of fully-downloaded models (ignores .part files)."""
    out = []
    for b in sorted(glob.glob(os.path.join(MODELS_DIR, "ggml-*.bin"))):
        if os.path.getsize(b) > MODEL_MIN_BYTES:
            out.append(os.path.basename(b))
    return out


def find_model():
    """Path to the active model file, or None if none usable."""
    installed = installed_models()
    if not installed:
        return None
    want = _read_config().get("model")
    if want and want in installed:
        return os.path.join(MODELS_DIR, want)
    # Default: the largest installed model (usually the most capable).
    best = max(installed,
               key=lambda n: os.path.getsize(os.path.join(MODELS_DIR, n)))
    return os.path.join(MODELS_DIR, best)


def download_model(name):
    """Background worker: fetch ggml-<name>.bin from Hugging Face with progress."""
    url = HF_BASE + model_filename(name)
    dest = os.path.join(MODELS_DIR, model_filename(name))
    part = dest + ".part"
    had_models = bool(installed_models())  # true first-run download auto-activates
    with DL_LOCK:
        DL.update(active=True, name=name, pct=0, done=False, error=None)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "WhisperTranscript"})
        with urllib.request.urlopen(req, timeout=30, context=ssl_context()) as resp:
            total = int(resp.headers.get("Content-Length", "0"))
            got = 0
            with open(part, "wb") as f:
                while True:
                    chunk = resp.read(262144)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    if total:
                        with DL_LOCK:
                            DL["pct"] = int(got * 100 / total)
        os.replace(part, dest)
        # Only the very first model auto-activates; later downloads don't
        # hijack the user's current selection (they click "Use" to switch).
        if not had_models:
            cfg = _read_config()
            cfg["model"] = model_filename(name)
            _write_config(cfg)
        with DL_LOCK:
            DL.update(active=False, pct=100, done=True)
    except Exception as e:  # noqa
        try:
            if os.path.exists(part):
                os.remove(part)
        except OSError:
            pass
        with DL_LOCK:
            DL.update(active=False, error=str(e), done=False)


def dl_state():
    with DL_LOCK:
        return dict(DL)


# ----------------------------------------------------------------------------
# Transcription pipeline
# ----------------------------------------------------------------------------
def srt_time_to_clock(srt_ts):
    """'00:01:23,456' -> '00:01:23' (drop millis for the markdown)."""
    return srt_ts.split(",")[0]


def parse_srt(srt_text):
    """Parse SRT into a list of (start, end, text) tuples."""
    blocks = re.split(r"\n\s*\n", srt_text.strip())
    out = []
    for block in blocks:
        lines = [l for l in block.splitlines() if l.strip() != ""]
        if len(lines) < 2:
            continue
        # lines[0] = index, lines[1] = timestamps, rest = text
        ts_line = lines[1] if "-->" in lines[1] else (lines[0] if "-->" in lines[0] else None)
        if not ts_line:
            continue
        m = re.match(r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})", ts_line)
        if not m:
            continue
        start, end = m.group(1), m.group(2)
        text_lines = lines[2:] if "-->" in lines[1] else lines[1:]
        text = " ".join(text_lines).strip()
        out.append((start, end, text))
    return out


def srt_to_markdown(srt_text, title):
    """Build a readable, timestamped markdown transcript from SRT."""
    segs = parse_srt(srt_text)
    lines = [f"# {title}", "", "_Transcript generated locally with Whisper._", ""]
    # Timestamped section
    lines.append("## Timestamped transcript")
    lines.append("")
    for start, end, text in segs:
        if not text:
            continue
        lines.append(f"**[{srt_time_to_clock(start)} → {srt_time_to_clock(end)}]** {text}")
        lines.append("")
    # Plain reading version (no timestamps)
    lines.append("---")
    lines.append("")
    lines.append("## Full text")
    lines.append("")
    full = " ".join(t for _, _, t in segs if t)
    lines.append(full)
    lines.append("")
    return "\n".join(lines)


def stream_multipart_file(rfile, content_length, boundary, dest_path):
    """Stream the first file field of a multipart body straight to disk.

    Avoids loading large videos into memory. We only need the single 'file'
    field the frontend sends.
    """
    boundary_b = boundary.encode("latin-1")
    delim = b"--" + boundary_b
    remaining = [content_length]

    def read_chunk(n):
        n = min(n, remaining[0])
        if n <= 0:
            return b""
        data = rfile.read(n)
        remaining[0] -= len(data)
        return data

    # 1) Read until we have the first part's headers (up to blank line).
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = read_chunk(8192)
        if not chunk:
            break
        buf += chunk
    if b"\r\n\r\n" not in buf:
        raise ValueError("malformed upload (no part headers)")
    header_blob, rest = buf.split(b"\r\n\r\n", 1)
    m = re.search(rb'filename="([^"]*)"', header_blob)
    filename = m.group(1).decode("utf-8", "replace") if m else "video"

    # 2) Stream body to disk, stopping at the closing boundary.
    end_marker = b"\r\n" + delim
    keep = len(end_marker)
    with open(dest_path, "wb") as out:
        carry = rest
        while True:
            idx = carry.find(end_marker)
            if idx != -1:
                out.write(carry[:idx])
                break
            if len(carry) > keep:
                out.write(carry[:-keep])
                carry = carry[-keep:]
            chunk = read_chunk(65536)
            if not chunk:
                out.write(carry)
                break
            carry += chunk
    return os.path.basename(filename) or "video"


def set_job(job_id, **kwargs):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(kwargs)


def append_log(job_id, msg):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id]["log"].append(msg)


def run_job(job_id, video_path, original_name):
    base = os.path.splitext(original_name)[0]
    safe_base = re.sub(r"[^A-Za-z0-9 ._-]", "_", base).strip() or "transcript"
    out_prefix = os.path.join(OUTPUTS_DIR, f"{job_id}__{safe_base}")
    wav_path = out_prefix + ".wav"

    try:
        if not have_binary(FFMPEG):
            set_job(job_id, status="error",
                    error="ffmpeg was not found. It should ship with the app — try reinstalling.")
            return
        if not have_binary(WHISPER_CLI):
            set_job(job_id, status="error",
                    error="The Whisper engine (whisper-cli) was not found. It should ship with the app — try reinstalling.")
            return
        model = find_model()
        if not model:
            set_job(job_id, status="error",
                    error="No Whisper model installed yet. Pick a model on the start screen and let it download.")
            return

        set_job(job_id, status="extracting", progress=2, model=os.path.basename(model))
        append_log(job_id, "Extracting audio with ffmpeg…")

        # 1) Extract 16kHz mono PCM wav (audio extraction is fast even for long
        # files; a 2h ceiling just stops a wedged process from hanging forever).
        try:
            ff = subprocess.run(
                [FFMPEG, "-y", "-i", video_path, "-ar", "16000", "-ac", "1",
                 "-c:a", "pcm_s16le", wav_path],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                timeout=7200,
            )
        except subprocess.TimeoutExpired:
            set_job(job_id, status="error",
                    error="Audio extraction took too long and was stopped.")
            return
        if ff.returncode != 0 or not os.path.exists(wav_path):
            tail = "\n".join(ff.stdout.splitlines()[-15:])
            set_job(job_id, status="error",
                    error=f"ffmpeg could not read this file.\n\n{tail}")
            return

        append_log(job_id, "Audio ready. Transcribing with Whisper…")
        set_job(job_id, status="transcribing", progress=5)

        # 2) Whisper transcription (stream progress)
        cmd = [
            WHISPER_CLI,
            "-m", model,
            "-f", wav_path,
            "-osrt",
            "-otxt",
            "-of", out_prefix,
            "-pp",            # print progress
            "-l", "auto",     # auto-detect language
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        prog_re = re.compile(r"progress\s*=\s*(\d+)%")
        for line in proc.stdout:
            m = prog_re.search(line)
            if m:
                pct = int(m.group(1))
                # map whisper's 0-100 onto 5-99
                set_job(job_id, progress=5 + int(pct * 0.94))
        proc.wait()

        srt_path = out_prefix + ".srt"
        if proc.returncode != 0 or not os.path.exists(srt_path):
            set_job(job_id, status="error",
                    error="Whisper finished without producing a transcript. The audio may be silent or unsupported.")
            return

        # 3) Read SRT, build markdown
        with open(srt_path, "r", encoding="utf-8") as f:
            srt_text = f.read()
        md_text = srt_to_markdown(srt_text, safe_base)
        md_path = out_prefix + ".md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_text)

        append_log(job_id, "Done.")
        set_job(job_id,
                status="done",
                progress=100,
                srt_path=srt_path,
                md_path=md_path,
                srt_text=srt_text,
                md_text=md_text,
                download_base=f"{safe_base}")

    except Exception as e:  # noqa
        sys.stderr.write(f"[whisper-transcript] job {job_id} failed: {e}\n")
        set_job(job_id, status="error",
                error="Something went wrong while processing this file.")
    finally:
        # clean intermediate wav + uploaded video to save space
        for p in (wav_path, video_path):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass


# ----------------------------------------------------------------------------
# HTTP handler
# ----------------------------------------------------------------------------
class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # quiet

    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _guard(self):
        """Reject DNS-rebinding and cross-site requests against this local server.

        The app is meant to be driven only by its own page on 127.0.0.1, so we
        require a loopback Host header and refuse anything a browser marks as a
        cross-site request (CSRF) or that carries a non-local Origin.
        """
        if self.headers.get("Host", "") not in ALLOWED_HOSTS:
            self._send_json({"error": "forbidden"}, 403)
            return False
        if (self.headers.get("Sec-Fetch-Site") or "same-origin") not in (
                "same-origin", "same-site", "none"):
            self._send_json({"error": "cross-site request blocked"}, 403)
            return False
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).hostname not in ("127.0.0.1", "localhost"):
            self._send_json({"error": "bad origin"}, 403)
            return False
        return True

    # ---- GET ----
    def do_GET(self):
        if not self._guard():
            return
        path = urlparse(self.path).path

        if path == "/" or path == "/index.html":
            with open(os.path.join(STATIC_DIR, "index.html"), "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/favicon.png" or path == "/favicon.ico":
            icon = os.path.join(STATIC_DIR, "favicon.png")
            if os.path.exists(icon):
                with open(icon, "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "max-age=86400")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self._send_json({"error": "not found"}, 404)
            return

        if path == "/status":
            model = find_model()
            self._send_json({
                "model": os.path.basename(model) if model else None,
                "model_ready": bool(model),
                "has_models": bool(installed_models()),
                "whisper": have_binary(WHISPER_CLI),
                "ffmpeg": have_binary(FFMPEG),
                "download": dl_state(),
            })
            return

        if path == "/models":
            installed = set(installed_models())
            active = find_model()
            active_name = os.path.basename(active) if active else None
            catalog = [{
                "name": m["name"],
                "label": m["label"],
                "size_mb": m["size_mb"],
                "note": m["note"],
                "filename": model_filename(m["name"]),
                "installed": model_filename(m["name"]) in installed,
                "active": model_filename(m["name"]) == active_name,
            } for m in MODEL_CATALOG]
            self._send_json({"current": active_name, "catalog": catalog,
                             "download": dl_state()})
            return

        if path.startswith("/progress/"):
            job_id = path.split("/")[-1]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if not job:
                    self._send_json({"error": "unknown job"}, 404)
                    return
                payload = {
                    "status": job["status"],
                    "progress": job["progress"],
                    "error": job.get("error"),
                    "model": job.get("model"),
                    "log": job["log"][-1] if job["log"] else "",
                }
                if job["status"] == "done":
                    payload.update({
                        "srt_text": job["srt_text"],
                        "md_text": job["md_text"],
                        "download_base": job["download_base"],
                    })
            self._send_json(payload)
            return

        if path.startswith("/download/"):
            # /download/<job_id>/<srt|md>
            parts = path.split("/")
            if len(parts) == 4:
                job_id, which = parts[2], parts[3]
                with JOBS_LOCK:
                    job = JOBS.get(job_id)
                if job and job.get("status") == "done":
                    fpath = job.get(f"{which}_path")
                    if fpath and os.path.exists(fpath):
                        fname = f"{job['download_base']}.{which}"
                        with open(fpath, "rb") as f:
                            body = f.read()
                        self.send_response(200)
                        ctype = "application/x-subrip" if which == "srt" else "text/markdown"
                        self.send_header("Content-Type", ctype)
                        self.send_header(
                            "Content-Disposition",
                            f'attachment; filename="{fname}"')
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                        return
            self._send_json({"error": "not found"}, 404)
            return

        self._send_json({"error": "not found"}, 404)

    # ---- POST ----
    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_JSON_BYTES:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            return {}

    def do_POST(self):
        if not self._guard():
            return
        path = urlparse(self.path).path

        if path == "/download_model":
            body = self._read_json_body()
            name = body.get("name", "")
            valid = {m["name"] for m in MODEL_CATALOG}
            if name not in valid:
                self._send_json({"error": "unknown model"}, 400)
                return
            with DL_LOCK:
                if DL["active"]:
                    self._send_json({"error": "a download is already running"}, 409)
                    return
            threading.Thread(target=download_model, args=(name,),
                             daemon=True).start()
            self._send_json({"ok": True, "name": name})
            return

        if path == "/select_model":
            body = self._read_json_body()
            fname = body.get("filename", "")
            if fname not in installed_models():
                self._send_json({"error": "model not installed"}, 400)
                return
            cfg = _read_config()
            cfg["model"] = fname
            _write_config(cfg)
            self._send_json({"ok": True, "model": fname})
            return

        if path != "/upload":
            self._send_json({"error": "not found"}, 404)
            return

        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ctype:
            self._send_json({"error": "expected multipart upload"}, 400)
            return

        m = re.search(r"boundary=([^;]+)", ctype)
        if not m:
            self._send_json({"error": "missing multipart boundary"}, 400)
            return
        boundary = m.group(1).strip().strip('"')
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length > MAX_UPLOAD_BYTES:
            self._send_json({"error": "file is too large"}, 413)
            return

        job_id = uuid.uuid4().hex[:12]
        tmp_dest = os.path.join(UPLOADS_DIR, f"{job_id}__upload")
        try:
            original_name = stream_multipart_file(
                self.rfile, content_length, boundary, tmp_dest)
        except Exception:  # noqa
            self._send_json({"error": "upload failed"}, 400)
            return

        # Build the on-disk path from a sanitized name (never trust the filename).
        dest = os.path.join(UPLOADS_DIR, f"{job_id}__{safe_filename(original_name, 'video')}")
        try:
            os.replace(tmp_dest, dest)
        except OSError:
            dest = tmp_dest

        with JOBS_LOCK:
            JOBS[job_id] = {"status": "queued", "progress": 0, "log": [],
                            "name": original_name}
            # Evict oldest jobs so the in-memory store can't grow without bound.
            while len(JOBS) > MAX_JOBS:
                JOBS.pop(next(iter(JOBS)))

        t = threading.Thread(target=run_job, args=(job_id, dest, original_name),
                             daemon=True)
        t.start()

        self._send_json({"job_id": job_id, "name": original_name})


class ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def server_bind(self):
        # http.server's server_bind() calls socket.getfqdn() (reverse DNS),
        # which can hang for 30s+ on some networks. Bind directly instead.
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = host
        self.server_port = port


def main():
    model = find_model()
    print("=" * 60)
    print("  Whisper Transcript App")
    print("=" * 60)
    print(f"  whisper-cli : {WHISPER_CLI}  {'OK' if have_binary(WHISPER_CLI) else 'MISSING'}")
    print(f"  ffmpeg      : {FFMPEG}  {'OK' if have_binary(FFMPEG) else 'MISSING'}")
    print(f"  model       : {os.path.basename(model) if model else 'none (pick one in the app)'}")
    print(f"\n  Open:  http://{HOST}:{PORT}\n")
    print("  (Ctrl+C to quit)")
    print("=" * 60)
    server = ThreadingServer((HOST, PORT), Handler)
    if os.environ.get("WT_NO_BROWSER") != "1":
        try:
            webbrowser.open(f"http://{HOST}:{PORT}")
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nBye.")


if __name__ == "__main__":
    main()
