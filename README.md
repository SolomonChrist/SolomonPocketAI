<p align="center">
  <img src="assets/solomon-christ-logo-white.jpg" width="360" alt="Solomon Christ logo">
</p>

# Solomon Pocket AI

A simple, private voice assistant that runs on your Windows PC. Talk naturally, type a message, import a document, inspect a photo, or ask it to look through the camera once. The conversation model, speech recognition, speech generation, memory, and camera analysis run locally.

## Install from scratch

You do not need to install Git, Python, Python packages, or Ollama manually. The setup checks them and installs the tested versions when needed. You need Windows 10 or 11, Windows Package Manager (`winget`, included with the Microsoft Store **App Installer**), an internet connection for the first setup, a microphone, and roughly 7 GB of free disk space.

1. On GitHub, choose **Code > Download ZIP**. This route does not require Git.
2. Extract the ZIP, then open the extracted `SolomonPocketAI` folder.
3. Double-click **`setup.bat`**.

If you prefer Windows Terminal, open it in that folder and run this exact command:

```powershell
.\setup.bat
```

Do not type only `setup` in PowerShell. `setup` is the name of an unrelated PowerShell/Pester command on some computers, which produces “The Setup command may only be used inside a Describe block.” The `.\setup.bat` form explicitly runs this project's installer.

Setup installs or verifies Git, the tested Python 3.11 runtime, all packages in `desktop-requirements.txt`, Ollama, and the local language and speech models. It creates an isolated Python environment, verifies the engines, and launches the app. If an earlier attempt created an environment with the wrong Python version, setup safely moves that folder aside and rebuilds it. Later, double-click `start-solomon-pocket-ai.bat` to run the app again.

## What it can do

- Hold private spoken or typed conversations using `qwen3.5:4b` through local Ollama.
- Detect and transcribe English or Mandarin microphone input with local Whisper, and speak answers with local Kokoro.
- Stream speech through one continuous audio connection while the next phrase renders ahead, avoiding stop-and-restart gaps between generated sentences.
- Replay the three most recent spoken responses instantly from the local audio cache.
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
- Standard, Large, and Extra Large interface text sizes that apply throughout the app and persist locally.
- An optional single trusted folder that replaces the private app workspace as the only file area the assistant can use.

The app adapts microphone sample rates and automatically resamples Kokoro audio for speakers that do not accept its native 24 kHz output. Audio choices are saved by device name and Windows audio backend rather than unstable device numbers. If a monitor, dock, headset, or speaker was just connected, click **Refresh devices**. If Windows has no active default speaker, Solomon Pocket AI prefers a device explicitly named **Speakers** and lets you test or override it.

## Privacy boundary

Runtime files live only in `SolomonPocketAIData/`, and Git ignores that entire directory. The model cannot browse arbitrary files, execute shell commands, or continuously access the camera. By default, files must be selected by you and are copied into its protected workspace before use.

You can optionally choose exactly one **Trusted Folder** in Settings. Solomon Pocket AI then sees only supported files inside that folder through logical IDs rather than Windows paths. It may read bounded TXT, MD, JSON, CSV, PDF, PNG, JPG/JPEG, and WebP files and create new TXT or MD files. Programs, scripts, shortcuts, symlinks, junctions/reparse points, hard-linked files, traversal paths, and files outside the selected folder are blocked. Choosing an entire drive or your whole home folder is also refused. This is an application-level safety boundary; a packaged release still needs the planned operating-system sandbox for defense in depth.

The three-response replay cache also stays inside `SolomonPocketAIData/`. It rotates automatically and is erased when you clear the conversation.

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

## License

Solomon Pocket AI is open-source software released under the [Apache License 2.0](LICENSE). The accompanying [NOTICE](NOTICE) contains the attribution that must be preserved when applicable under the license. Third-party models, dependencies, and services retain their respective licenses and terms.
