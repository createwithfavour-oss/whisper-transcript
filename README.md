# 🎙️ Whisper Transcript

Drag in a video or audio file → get a subtitle file (`.srt`) and a clean, timestamped transcript (`.md`). It runs **entirely on your own computer** — nothing is uploaded anywhere, and it works offline.

Powered by [whisper.cpp](https://github.com/ggerganov/whisper.cpp) (OpenAI's Whisper, running locally).

---

## Download

Grab the latest version for your computer from the **[Releases page →](https://github.com/createwithfavour-oss/whisper-transcript/releases/latest)**

- **Windows** → `WhisperTranscript-Windows.zip`
- **Mac** → `WhisperTranscript-macOS.zip`

No installation, no accounts, no Python — just download, unzip, and open.

---

## How to use it

### 1. Open the app

**Windows**
1. Unzip `WhisperTranscript-Windows.zip`.
2. Open the `WhisperTranscript` folder and double-click **`WhisperTranscript.exe`**.
3. First time only: Windows may show a blue **"Windows protected your PC"** box. Click **More info → Run anyway**. (This appears because the app isn't code-signed yet — it's safe.)
4. A small black window opens (that's the engine — leave it open) and your browser opens automatically.

**Mac**
1. Unzip `WhisperTranscript-macOS.zip`.
2. **Right-click** `Whisper Transcript.app` → **Open** → **Open**. (First time only. macOS blocks double-clicking unsigned apps, but right-click → Open works. After this, you can open it normally.)
3. Your browser opens automatically.

### 2. Pick a model (first launch only)

The first time you run it, you choose a transcription model. It downloads once, then works offline forever.

| Model | Size | Best for |
|---|---|---|
| Tiny | 75 MB | Quick drafts, any old laptop |
| Base | 142 MB | Fast, clear speech |
| Small | 466 MB | A good balance |
| **Large v3 Turbo** | 1.5 GB | **Best accuracy (recommended)** |

Bigger = more accurate but a larger download and a bit slower. You can add or switch models later with the **"change"** link.

### 3. Transcribe

1. **Drag a video or audio file** onto the box (or click to browse).
2. Watch the progress bar: *extracting audio → transcribing*.
3. When it's done, click **Download .md** and **Download .srt**.

That's it. Works with mp4, mov, mkv, mp3, wav, m4a, and most other formats.

### To quit

- **Mac:** right-click the app in the Dock → Quit.
- **Windows:** close the small black window.

---

## Good to know

- **100% private.** Your files never leave your computer. No internet needed after the model downloads.
- **Longest file:** up to ~37 hours of audio per file.
- **It's open source.** Code and build pipeline are in this repo.

---

## For developers

Want to build it yourself or see how the cloud builds work? See **[BUILD.md](BUILD.md)**.

## License & credits

Built on [whisper.cpp](https://github.com/ggerganov/whisper.cpp) and the Whisper models (MIT), with [ffmpeg](https://ffmpeg.org) for audio extraction.
