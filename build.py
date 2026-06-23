#!/usr/bin/env python3
"""
Build a self-contained Whisper Transcript app for the current OS.

Run on each CI runner (macOS / Windows):

    python build.py

Steps:
  1. Build whisper.cpp from source (static) -> bin/whisper-cli[.exe]
  2. Download a static ffmpeg            -> bin/ffmpeg[.exe]
  3. Bundle server.py + static/ + bin/ with PyInstaller
  4. Zip the result into dist/

Notes:
  - whisper.cpp is built static (BUILD_SHARED_LIBS=OFF) so there are no
    loose .dll/.dylib files to chase. On macOS it links the system Metal /
    Accelerate frameworks (present on every Mac).
  - The model is NOT bundled. The app downloads it on first launch.
  - This script downloads third-party binaries (ffmpeg) and source
    (whisper.cpp) at build time. If an upstream URL changes, update the
    constants below.
"""

import os
import sys
import shutil
import zipfile
import hashlib
import subprocess
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(ROOT, "bin")
WORK = os.path.join(ROOT, "_buildwork")
IS_WIN = os.name == "nt"
IS_MAC = sys.platform == "darwin"
EXE = ".exe" if IS_WIN else ""

# Pin whisper.cpp to a release tag so every build is reproducible and we never
# ship whatever happens to be on master that day.
WHISPER_REPO = "https://github.com/ggerganov/whisper.cpp"
WHISPER_TAG = "v1.9.1"

# Static ffmpeg downloads (single self-contained binary), pinned by SHA-256.
# These hashes pin the CURRENT upstream build. If upstream publishes a new one,
# the build fails loudly with a checksum mismatch (a tamper-evident signal) —
# download the new file, confirm it's legitimate, and update the hash here.
FFMPEG_MAC = "https://evermeet.cx/ffmpeg/getrelease/zip"
FFMPEG_MAC_SHA256 = "e91df72a1ee7c26606f90dd2dd4dcccc6a75140ff9ea6fdd50faae828b82ba69"
FFMPEG_WIN = ("https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
              "ffmpeg-master-latest-win64-lgpl.zip")
FFMPEG_WIN_SHA256 = "9e2c17188ffbcc35f03d4f3e27f3844dec3075d4da686c67917d3b027f5bee1a"


def run(cmd, **kw):
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True, **kw)


def fresh_dir(p):
    if os.path.isdir(p):
        shutil.rmtree(p, ignore_errors=True)
    os.makedirs(p, exist_ok=True)


def download(url, dest, sha256=None):
    print(f"downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "WhisperTranscript-build"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    if sha256:
        h = hashlib.sha256()
        with open(dest, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        got = h.hexdigest()
        if got != sha256:
            raise SystemExit(
                f"CHECKSUM MISMATCH for {url}\n"
                f"  expected {sha256}\n"
                f"  got      {got}\n"
                "Upstream likely published a new build. Verify it is legitimate, "
                "then update the pinned hash in build.py.")
        print(f"  sha256 OK ({got[:16]}…)")


def find_one(root, name):
    """Return first path under root whose basename == name."""
    for dp, _, files in os.walk(root):
        if name in files:
            return os.path.join(dp, name)
    return None


def build_whisper():
    print("=== building whisper.cpp (static) ===")
    src = os.path.join(WORK, "whisper.cpp")
    fresh_dir(WORK)
    run(["git", "clone", "--depth", "1", "--branch", WHISPER_TAG, WHISPER_REPO, src])
    bld = os.path.join(src, "build")
    cfg = [
        "cmake", "-S", src, "-B", bld,
        "-DBUILD_SHARED_LIBS=OFF",
        "-DWHISPER_BUILD_EXAMPLES=ON",
        "-DWHISPER_BUILD_TESTS=OFF",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    run(cfg)
    run(["cmake", "--build", bld, "--config", "Release", "-j", "4"])
    binary = find_one(bld, "whisper-cli" + EXE)
    if not binary:
        # very old checkouts called it "main"
        binary = find_one(bld, "main" + EXE)
    if not binary:
        raise SystemExit("could not find built whisper-cli binary")
    dest = os.path.join(BIN, "whisper-cli" + EXE)
    shutil.copy2(binary, dest)
    if not IS_WIN:
        os.chmod(dest, 0o755)
    print("whisper-cli ->", dest)


def fetch_ffmpeg():
    print("=== fetching static ffmpeg ===")
    url = FFMPEG_WIN if IS_WIN else FFMPEG_MAC
    sha = FFMPEG_WIN_SHA256 if IS_WIN else FFMPEG_MAC_SHA256
    arc = os.path.join(WORK, "ffmpeg_dl.zip")
    download(url, arc, sha256=sha)
    ex = os.path.join(WORK, "ffmpeg_extract")
    fresh_dir(ex)
    with zipfile.ZipFile(arc) as z:
        z.extractall(ex)
    binary = find_one(ex, "ffmpeg" + EXE)
    if not binary:
        raise SystemExit("could not find ffmpeg binary in archive")
    dest = os.path.join(BIN, "ffmpeg" + EXE)
    shutil.copy2(binary, dest)
    if not IS_WIN:
        os.chmod(dest, 0o755)
    print("ffmpeg ->", dest)


def make_windows_ico():
    """Create assets/icon.ico from assets/icon.png (needs Pillow)."""
    out = os.path.join(ROOT, "assets", "icon.ico")
    try:
        from PIL import Image
        img = Image.open(os.path.join(ROOT, "assets", "icon.png"))
        img.save(out, sizes=[(16, 16), (32, 32), (48, 48), (64, 64),
                             (128, 128), (256, 256)])
        print("icon.ico ->", out)
    except Exception as e:  # noqa
        print("WARNING: could not build .ico, app will use default icon:", e)
        return None
    return out


def bundle():
    print("=== PyInstaller bundle ===")
    sep = ";" if IS_WIN else ":"
    args = [
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
        "--add-data", f"static{sep}static",
        "--add-binary", f"{os.path.join(BIN, 'ffmpeg' + EXE)}{sep}bin",
        "--add-binary", f"{os.path.join(BIN, 'whisper-cli' + EXE)}{sep}bin",
    ]
    if IS_MAC:
        # A real .app, no console, quit from the Dock. Opens the browser itself.
        args += ["--windowed", "--name", "Whisper Transcript",
                 "--icon", os.path.join(ROOT, "assets", "AppIcon.icns"),
                 "--osx-bundle-identifier", "com.favouryusuf.whispertranscript"]
    else:
        # Windows: keep a small console window (close it to quit). A tray-icon
        # "quit" is a future polish.
        ico = make_windows_ico()
        args += ["--name", "WhisperTranscript"]
        if ico:
            args += ["--icon", ico]
    args += ["server.py"]
    run(args)


def package():
    print("=== zipping distributable ===")
    os.makedirs(os.path.join(ROOT, "dist"), exist_ok=True)
    if IS_MAC:
        app = os.path.join(ROOT, "dist", "Whisper Transcript.app")
        out = os.path.join(ROOT, "dist", "WhisperTranscript-macOS.zip")
        # ditto preserves the bundle (symlinks, permissions, resource forks).
        run(["ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", app, out])
    else:
        folder = os.path.join(ROOT, "dist", "WhisperTranscript")
        out = os.path.join(ROOT, "dist", "WhisperTranscript-Windows.zip")
        if os.path.exists(out):
            os.remove(out)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for dp, _, files in os.walk(folder):
                for fn in files:
                    full = os.path.join(dp, fn)
                    z.write(full, os.path.relpath(full, os.path.join(ROOT, "dist")))
    print("packaged ->", out)


def main():
    fresh_dir(BIN)
    build_whisper()
    fetch_ffmpeg()
    bundle()
    package()
    print("\nDONE. Artifacts in dist/")


if __name__ == "__main__":
    main()
