"""Download the local speech models into the app-owned data directory."""

from __future__ import annotations

import argparse
import hashlib
import os
import urllib.request
from pathlib import Path


KOKORO_FILES = (
    (
        "kokoro-v1.0.fp16.onnx",
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
        "model-files-v1.1/kokoro-v1.0.fp16.onnx",
        "f3a290d384fbb27966d462905c71a46cef9e5fd00516b40df32a0b4afe77ac96",
    ),
    (
        "voices-v1.0.bin",
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
        "model-files-v1.1/voices-v1.0.bin",
        "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_hash: str) -> None:
    if destination.is_file() and sha256(destination) == expected_hash:
        print(f"Already verified: {destination.name}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    temporary.unlink(missing_ok=True)
    print(f"Downloading {destination.name}...")
    request = urllib.request.Request(url, headers={"User-Agent": "SolomonPocketAI-Setup/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    actual_hash = sha256(temporary)
    if actual_hash != expected_hash:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for {destination.name}: expected {expected_hash}, got {actual_hash}"
        )
    os.replace(temporary, destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    args = parser.parse_args()
    data_root = args.data_root.resolve()

    whisper_root = data_root / "models" / "whisper"
    whisper_root.mkdir(parents=True, exist_ok=True)
    whisper_path = whisper_root / "tiny.pt"
    if whisper_path.is_file():
        print("Already present: tiny.pt")
    else:
        print("Downloading Whisper tiny...")
        import whisper

        whisper.load_model("tiny", device="cpu", download_root=str(whisper_root))

    kokoro_root = data_root / "models" / "kokoro"
    for name, url, expected_hash in KOKORO_FILES:
        download(url, kokoro_root / name, expected_hash)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
