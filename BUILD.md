# Building & distributing Whisper Transcript

The app is plain Python + two external tools (ffmpeg, whisper.cpp) + a model
the app downloads on first run. CI packages it into self-contained downloads
for macOS and Windows — users install nothing.

## How the build works

`build.py` runs on each OS and does four things:

1. **Builds whisper.cpp from source, statically** (`BUILD_SHARED_LIBS=OFF`) so
   there are no loose `.dll`/`.dylib` files to ship. On macOS it links the
   system Metal/Accelerate frameworks (on every Mac); on Windows it links the
   system runtime.
2. **Downloads a static ffmpeg** (single self-contained binary).
3. **Bundles** `server.py` + `static/` + `bin/` with PyInstaller, including the
   Python runtime — so users don't need Python installed.
4. **Zips** the result into `dist/`.

The Whisper **model is not bundled** (it's 75MB–1.5GB). The app shows a model
picker on first launch and downloads the chosen one to a per-user data folder.

## Running it via GitHub Actions (the normal path)

The workflow in `.github/workflows/build.yml` builds both OSes in the cloud.

- **Manual run:** Actions tab → "Build apps" → "Run workflow". Download the
  `WhisperTranscript-macOS` / `WhisperTranscript-Windows` artifacts.
- **Release:** push a version tag and it also publishes the zips to a GitHub
  Release (shareable download links):

  ```bash
  git tag v1.0.0
  git push origin v1.0.0
  ```

## Building locally (optional)

Needs `cmake`, `git`, and Python with `pip install pyinstaller pillow`.

```bash
python build.py        # outputs dist/WhisperTranscript-<os>.zip
```

## What users get

- **macOS:** `Whisper Transcript.app` — double-click, browser opens, quit from
  the Dock.
- **Windows:** a `WhisperTranscript` folder with `WhisperTranscript.exe` — double-click,
  browser opens. A small console window stays open; close it to quit. (A tray
  "quit" button is a planned polish.)

On first launch either one asks which model to download, then works fully
offline.

## Caveats / first-run notes

- **Unsigned apps.** Downloaded apps are unsigned, so macOS shows
  "unidentified developer" (right-click → Open) and Windows SmartScreen shows
  "More info → Run anyway". Code-signing (Apple Developer $99/yr; a Windows
  cert) removes this later if the app takes off.
- **First CI run may need a tweak.** `build.py` downloads upstream binaries and
  source; if an upstream URL or output path changes, adjust the constants at the
  top of `build.py`. Read the Actions logs to see exactly where.
- **Licensing.** whisper.cpp and the Whisper models are MIT. ffmpeg is the LGPL
  build on Windows; on macOS it's a static build from evermeet.cx — include its
  license if you distribute widely.
