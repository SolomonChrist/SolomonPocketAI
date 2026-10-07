# Contributing

Keep Solomon Pocket AI simple, local-first, and safe by default.

1. Do not add general filesystem, shell, process, browser, or continuous-camera access.
2. Keep model-facing file operations inside `SolomonPocketAIData/` and use logical IDs instead of host paths.
3. Do not commit personal information, credentials, private documents, runtime data, or model files.
4. Add or update tests for capability and security-boundary changes.
5. Run the unit tests and `scripts/check-public-repo.ps1` before opening a pull request.

Large features should first prove a small, usable vertical slice.
