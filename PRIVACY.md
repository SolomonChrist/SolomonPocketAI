# Privacy

Solomon Pocket AI is designed to keep the assistant and its working data inside its own folder.

## Local data

The app stores downloaded models, conversation history, Markdown memory, imported copies, generated notes, settings, and one-shot camera captures under `SolomonPocketAIData/`. That directory is excluded from Git.

The model receives logical item identifiers and bounded content. It does not receive arbitrary host paths and has no shell or general filesystem tool.

The user may optionally replace the built-in file workspace with one explicitly selected trusted folder. Only its path is stored in the ignored local settings file. Model-facing operations receive logical IDs, never that host path. Safe-document reads are bounded to TXT, Markdown, JSON, CSV, PDF, PNG, JPEG, and WebP, while writes remain create-only TXT/Markdown. Link/reparse escapes, hard links, traversal, executable/script formats, drive roots, and the whole home directory are refused at the application layer.

## Camera

Camera access happens only after an explicit camera action. The app opens camera device 0, captures one frame, releases the device, and analyzes the saved local frame. It does not provide background or continuous camera access.

## Network

After initial model and dependency downloads, normal conversation, speech recognition, speech generation, memory, document reading, image understanding, and camera analysis can work offline.

The app makes a network request only when the user requests one of these features:

- Weather: Open-Meteo geocoding and forecast hosts.
- Current fact: English Wikipedia endpoints.
- Additional speech model: the user presses **Download selected** in Settings to fetch an official Whisper model into the ignored local model folder.

There is no general web browser, account login, cookie store, advertising, analytics, or telemetry in the app.

## Before publishing

Run `scripts/check-public-repo.ps1`. It checks every file Git could include for machine-specific paths, email addresses, common secret formats, private keys, runtime data, and oversized model files. Review the resulting file list before any commit or push.
