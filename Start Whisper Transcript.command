#!/bin/bash
# Double-click this file in Finder to launch the Whisper Transcript app.
cd "$(dirname "$0")"
echo "Starting Whisper Transcript… a browser tab will open shortly."
exec python3 server.py
