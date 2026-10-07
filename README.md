<p align="center">
  <img src="assets/solomon-christ-logo-white.jpg" width="360" alt="Solomon Christ logo">
</p>

# Solomon Pocket AI

A simple, private voice assistant that runs on your Windows PC. Talk naturally, type a message, import a document, inspect a photo, or ask it to look through the camera once. The conversation model, speech recognition, speech generation, memory, and camera analysis run locally.

## Start in one command

Requirements: Windows 10 or 11, an internet connection for the first setup, a microphone, and roughly 5 GB of free disk space.

1. Download or clone this repository.
2. Open the folder in Terminal.
3. Run:

```bat
setup.bat
```

Setup creates an isolated Python environment, installs Ollama if needed, downloads the local language and speech models, verifies the engines, and launches the app. Later, double-click `start-solomon-pocket-ai.bat` to run it again.

## What it can do

- Hold private spoken or typed conversations using `qwen3.5:4b` through local Ollama.
- Transcribe microphone input with local Whisper and speak answers with local Kokoro.
- Keep short Obsidian-compatible Markdown memory.
- Import and read bounded TXT, Markdown, JSON, CSV, PDF, PNG, JPEG, and WebP files.
- Capture one camera frame only when you press the camera action or explicitly ask.
- Get live weather from Open-Meteo or a bounded current fact from English Wikipedia when requested.

## Settings

Open **Settings** in the main window to see and test the exact local stack:

- Microphone input and speaker output, including device refresh, a two-second microphone meter, and a spoken speaker test.
- Any conversation/thinking model already installed in Ollama.
- Whisper speech-to-text size, with an in-app download button for additional supported models.
- Installed Kokoro voice model, all 54 available voice styles, language/gender labels, and voice speed.

The app adapts microphone sample rates and automatically resamples Kokoro audio for speakers that do not accept its native 24 kHz output. Audio choices are saved by device name and Windows audio backend rather than unstable device numbers. If a monitor, dock, headset, or speaker was just connected, click **Refresh devices**. If Windows has no active default speaker, Solomon Pocket AI prefers a device explicitly named **Speakers** and lets you test or override it.

## Privacy boundary

Runtime files live only in `SolomonPocketAIData/`, and Git ignores that entire directory. The model cannot browse arbitrary files, execute shell commands, or continuously access the camera. Files must be selected by you and are copied into its protected workspace before use.

Weather and current-fact requests are the only optional online app features. Everything else continues to work offline after setup. See [PRIVACY.md](PRIVACY.md) for the exact boundary.

## Useful commands

```powershell
# Launch
.\start-solomon-pocket-ai.bat

# Verify all local engines without opening the UI
.\.venv\Scripts\python.exe .\SolomonPocketAI.py --self-test

# Run the capability tests
.\.venv\Scripts\python.exe -m unittest discover -s python_tests -v

# Check what is safe to publish
powershell -NoProfile -File .\scripts\check-public-repo.ps1
```

## Project status

This is the first useful Windows release. Linux and macOS packaging are planned after the Windows experience is stable. The Android/phone version comes later, based on the same proven local conversation system.

## Contribute or support the project

Ideas, testing help, code contributions, and support for continued development are welcome.

- Website: [www.SolomonChrist.com](https://www.solomonchrist.com)
- Contact or donation inquiries: [solomon@solomonchrist.com](mailto:solomon@solomonchrist.com)

No open-source license has been selected yet. Do not publish the repository as open source until a license is added.
