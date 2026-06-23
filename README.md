# Whisper Transcript

Drag in a video (or audio) file → get an **`.srt`** and a timestamped **`.md`** transcript.
Everything runs locally on your Mac using whisper.cpp and your downloaded model. Nothing is uploaded anywhere.

## How to start it

**Easiest:** double-click **`Whisper Transcript.app`**.
The server starts in the background (no Terminal window) and a browser tab opens by itself at <http://127.0.0.1:8756>.

- **To stop it:** quit the app from the Dock (right-click its icon → Quit), or press ⌘Q.
- **First launch only:** if macOS says the app is from an unidentified developer, right-click the app → **Open** → **Open**. After that, double-click works normally.
- You can drag the `.app` into your **Applications** folder or onto the **Dock** — it still finds everything it needs.

**Alternative launchers:**

- `Start Whisper Transcript.command` — opens a Terminal window that hosts the server (Ctrl+C to stop).
- From Terminal:
  ```bash
  cd "/Users/phronesis/Claude Cowork/Whisper Transcript app"
  python3 server.py
  ```

## How to use

1. Drag a video onto the drop zone (or click to browse). Works with mp4, mov, mkv, mp3, wav, m4a, and more.
2. Watch the progress bar: **extracting audio → transcribing**.
3. When it's done, preview the Markdown or SRT, then click **Download .md** / **Download .srt**.

## What's under the hood

```
video  →  ffmpeg (extract 16kHz mono audio)  →  whisper-cli (transcribe)  →  .srt + .md
```

- **Engine:** `whisper-cli` (whisper.cpp), Metal-accelerated on your M3 Pro.
- **Model:** `models/ggml-large-v3-turbo.bin` (high accuracy, fast).
- **Language:** auto-detected.

## Folders

- `models/` — the Whisper model file(s). Drop other `ggml-*.bin` models here to swap; the app picks one up automatically.
- `outputs/` — generated `.srt` / `.md` files are kept here too (in case you want them later).
- `uploads/` — temporary; the source video is deleted after transcription to save space.

## Notes

- No Python packages to install — it uses only the standard library.
- To change the port, run `PORT=9000 python3 server.py`.
