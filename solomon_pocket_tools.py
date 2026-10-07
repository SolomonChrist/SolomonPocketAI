"""Narrow, user-invoked capabilities for the Solomon Pocket AI app.

The model never receives host paths. Human imports are copied into a fixed
vault, writes are limited to notes/outbox, camera access is one-shot, and the
only network client is an exact-host weather adapter.
"""

from __future__ import annotations

import json
import io
import os
import re
import stat
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath


MAX_IMPORT_BYTES = 20 * 1024 * 1024
MAX_TEXT_BYTES = 2 * 1024 * 1024
MAX_TEXT_CHARS = 40_000
MAX_PDF_PAGES = 50
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
MAX_NOTE_CHARS = 100_000
MAX_LISTED_FILES = 500
ALLOWED_EXTENSIONS = {".txt", ".md", ".json", ".csv", ".pdf", ".png", ".jpg", ".jpeg", ".webp"}
TEXT_EXTENSIONS = {".txt", ".md", ".json", ".csv"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL", "CLOCK$", "CONIN$", "CONOUT$",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


class PocketToolError(RuntimeError):
    pass


class SolomonPocketTools:
    def __init__(self, data_root: Path, trusted_root: str | Path | None = None) -> None:
        self.data_root = data_root.resolve()
        self.trusted_root = self.validate_trusted_root(trusted_root) if trusted_root else None
        self.vault_root = self.trusted_root or (self.data_root / "vault")
        self.inbox_root = self.trusted_root or (self.vault_root / "inbox")
        self.notes_root = self.trusted_root or (self.vault_root / "notes")
        self.outbox_root = self.trusted_root or (self.vault_root / "outbox")
        self.camera_root = self.data_root / "derived" / "camera"
        self.ensure_layout()

    def ensure_layout(self) -> None:
        folders = (self.camera_root,) if self.trusted_root else (
            self.inbox_root,
            self.notes_root,
            self.outbox_root,
            self.camera_root,
        )
        for folder in folders:
            folder.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _is_link_or_reparse(path: Path) -> bool:
        """Reject symlinks and Windows junction/reparse points without following them."""
        try:
            details = os.lstat(path)
        except OSError:
            return True
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        return stat.S_ISLNK(details.st_mode) or bool(
            getattr(details, "st_file_attributes", 0) & reparse_flag
        )

    @classmethod
    def validate_trusted_root(cls, selected: str | Path) -> Path:
        candidate = Path(str(selected).strip()).expanduser()
        if not str(selected).strip() or not candidate.exists() or not candidate.is_dir():
            raise PocketToolError("Choose an existing folder for trusted file access.")
        if cls._is_link_or_reparse(candidate):
            raise PocketToolError("A shortcut, symbolic link, junction, or reparse point cannot be trusted.")
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise PocketToolError("The trusted folder could not be resolved safely.") from exc
        if resolved == Path(resolved.anchor):
            raise PocketToolError("Choose a specific folder, not an entire drive.")
        try:
            if resolved == Path.home().resolve(strict=True):
                raise PocketToolError("Choose a specific folder, not your entire home folder.")
        except OSError:
            pass
        return resolved

    @property
    def workspace_name(self) -> str:
        return "trusted folder" if self.trusted_root else "protected app workspace"

    @property
    def workspace_path(self) -> Path:
        return self.trusted_root or self.vault_root

    def _active_trusted_root(self) -> Path | None:
        if self.trusted_root is None:
            return None
        current = self.validate_trusted_root(self.trusted_root)
        if current != self.trusted_root:
            raise PocketToolError("The trusted folder changed identity and was refused.")
        return current

    @staticmethod
    def _safe_name(name: str, default_extension: str | None = None) -> str:
        candidate = re.sub(r"\s+", " ", Path(str(name)).name).strip().rstrip(". ")
        if default_extension and not Path(candidate).suffix:
            candidate += default_extension
        if not candidate or candidate in {".", ".."} or len(candidate) > 120:
            raise PocketToolError("The file name is empty or too long.")
        if candidate != str(name).strip() and any(mark in str(name) for mark in ("/", "\\", ":")):
            raise PocketToolError("Folder paths are not accepted; use a file name only.")
        if any(ord(char) < 32 for char in candidate) or any(char in '<>:"/\\|?*' for char in candidate):
            raise PocketToolError("The file name contains unsupported characters.")
        if Path(candidate).stem.upper() in WINDOWS_RESERVED:
            raise PocketToolError("That reserved Windows file name is not allowed.")
        return candidate

    @staticmethod
    def _unique_destination(folder: Path, name: str) -> Path:
        destination = folder / name
        if not destination.exists():
            return destination
        stem = destination.stem
        suffix = destination.suffix
        for index in range(2, 1000):
            candidate = folder / f"{stem} ({index}){suffix}"
            if not candidate.exists():
                return candidate
        raise PocketToolError("Too many files have the same name.")

    @staticmethod
    def _validate_signature(path: Path, extension: str) -> None:
        with path.open("rb") as handle:
            head = handle.read(16)
        if extension == ".pdf" and not head.startswith(b"%PDF-"):
            raise PocketToolError("The selected file is not a valid PDF signature.")
        if extension == ".png" and not head.startswith(b"\x89PNG\r\n\x1a\n"):
            raise PocketToolError("The selected file is not a valid PNG signature.")
        if extension in {".jpg", ".jpeg"} and not head.startswith(b"\xff\xd8\xff"):
            raise PocketToolError("The selected file is not a valid JPEG signature.")
        if extension == ".webp" and not (head.startswith(b"RIFF") and head[8:12] == b"WEBP"):
            raise PocketToolError("The selected file is not a valid WebP signature.")

    def import_selected(self, selected_path: str | Path) -> str:
        """Trusted UI boundary: copy one explicitly selected file into inbox."""
        source = Path(selected_path)
        if not source.is_file() or source.is_symlink():
            raise PocketToolError("The selected item is not a regular file.")
        extension = source.suffix.casefold()
        if extension not in ALLOWED_EXTENSIONS:
            raise PocketToolError("Supported imports: TXT, MD, JSON, CSV, PDF, PNG, JPEG, and WebP.")
        size = source.stat().st_size
        if size <= 0 or size > MAX_IMPORT_BYTES:
            raise PocketToolError("The selected file is empty or exceeds the 20 MB import limit.")
        self._validate_signature(source, extension)
        if extension in TEXT_EXTENSIONS:
            if size > MAX_TEXT_BYTES:
                raise PocketToolError("Text imports are limited to 2 MB.")
            try:
                source.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise PocketToolError("Text imports must be UTF-8.") from exc
        if extension in IMAGE_EXTENSIONS:
            self._validate_image(source)
        name = self._safe_name(source.name)
        destination_root = self._active_trusted_root() or self.inbox_root
        destination = self._unique_destination(destination_root, name)
        temporary = destination.with_suffix(destination.suffix + ".importing")
        try:
            with source.open("rb") as reader, temporary.open("xb") as writer:
                copied = 0
                while True:
                    block = reader.read(1024 * 1024)
                    if not block:
                        break
                    copied += len(block)
                    if copied > MAX_IMPORT_BYTES:
                        raise PocketToolError("The file changed during import and exceeded the limit.")
                    writer.write(block)
                writer.flush()
                os.fsync(writer.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return f"trusted/{destination.name}" if self.trusted_root else f"inbox/{destination.name}"

    def _validate_image(self, path: Path) -> None:
        if path.stat().st_size > MAX_IMAGE_BYTES:
            raise PocketToolError("Images are limited to 12 MB.")
        try:
            from PIL import Image

            with Image.open(path) as image:
                width, height = image.size
                if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                    raise PocketToolError("The image dimensions exceed the 16-megapixel limit.")
                image.verify()
        except PocketToolError:
            raise
        except Exception as exc:
            raise PocketToolError("The image could not be safely decoded.") from exc

    def list_items(self) -> list[dict[str, object]]:
        if self.trusted_root:
            return self._list_trusted_items()
        items: list[dict[str, object]] = []
        for label, folder in (
            ("inbox", self.inbox_root),
            ("notes", self.notes_root),
            ("outbox", self.outbox_root),
        ):
            for path in folder.iterdir():
                if not path.is_file() or path.is_symlink() or path.suffix.casefold() not in ALLOWED_EXTENSIONS:
                    continue
                stat = path.stat()
                items.append(
                    {
                        "id": f"{label}/{path.name}",
                        "size": stat.st_size,
                        "modified": stat.st_mtime,
                    }
                )
        return sorted(items, key=lambda item: (-float(item["modified"]), str(item["id"]).casefold()))

    def _list_trusted_items(self) -> list[dict[str, object]]:
        root = self._active_trusted_root()
        if root is None:
            return []
        items: list[dict[str, object]] = []
        for current, directories, names in os.walk(root, topdown=True, followlinks=False):
            current_path = Path(current)
            directories[:] = [
                name
                for name in directories
                if self._trusted_directory_allowed(current_path / name, root)
            ]
            for name in names:
                path = current_path / name
                if not self._trusted_file_allowed(path, root):
                    continue
                details = path.stat()
                relative = path.relative_to(root).as_posix()
                items.append(
                    {
                        "id": f"trusted/{relative}",
                        "size": details.st_size,
                        "modified": details.st_mtime,
                    }
                )
                if len(items) >= MAX_LISTED_FILES:
                    return sorted(
                        items,
                        key=lambda item: (-float(item["modified"]), str(item["id"]).casefold()),
                    )
        return sorted(items, key=lambda item: (-float(item["modified"]), str(item["id"]).casefold()))

    @classmethod
    def _trusted_directory_allowed(cls, path: Path, root: Path) -> bool:
        if cls._is_link_or_reparse(path):
            return False
        try:
            return path.is_dir() and path.resolve(strict=True).is_relative_to(root)
        except (OSError, RuntimeError):
            return False

    @classmethod
    def _trusted_file_allowed(cls, path: Path, root: Path) -> bool:
        if path.suffix.casefold() not in ALLOWED_EXTENSIONS or cls._is_link_or_reparse(path):
            return False
        try:
            details = os.lstat(path)
            resolved = path.resolve(strict=True)
            return (
                stat.S_ISREG(details.st_mode)
                and details.st_nlink == 1
                and resolved.is_relative_to(root)
            )
        except (OSError, RuntimeError):
            return False

    def format_item_list(self) -> str:
        items = self.list_items()
        if not items:
            if self.trusted_root:
                return "The trusted folder has no supported safe files."
            return "The protected workspace is empty. Use Import file to copy something into it."
        lines = ["Trusted folder files:" if self.trusted_root else "Protected workspace files:"]
        for item in items[:50]:
            lines.append(f"- {item['id']} ({int(item['size']):,} bytes)")
        return "\n".join(lines)

    def _resolve_item(self, item_id: str | None) -> tuple[str, Path]:
        items = self.list_items()
        if not items:
            message = (
                "The trusted folder has no supported safe files."
                if self.trusted_root
                else "The protected workspace is empty. Import a file first."
            )
            raise PocketToolError(message)
        requested = (item_id or "").strip()
        if not requested:
            chosen = items[0]
        elif "/" in requested:
            if "\\" in requested or requested.startswith("/"):
                raise PocketToolError("Use a logical file ID shown by /files.")
            matches = [item for item in items if str(item["id"]).casefold() == requested.casefold()]
            if len(matches) != 1:
                raise PocketToolError("That workspace file was not found.")
            chosen = matches[0]
        else:
            safe = self._safe_name(requested)
            matches = [item for item in items if Path(str(item["id"])).name.casefold() == safe.casefold()]
            if len(matches) != 1:
                raise PocketToolError("The file name was not found or is ambiguous; use the ID shown by /files.")
            chosen = matches[0]
        label, name = str(chosen["id"]).split("/", 1)
        if label == "trusted" and self.trusted_root:
            relative = PurePosixPath(name)
            if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
                raise PocketToolError("The trusted-folder item failed its confinement check.")
            path = self.trusted_root.joinpath(*relative.parts)
            if not self._trusted_file_allowed(path, self.trusted_root):
                raise PocketToolError("The trusted-folder item failed its confinement check.")
            return str(chosen["id"]), path
        folder = {"inbox": self.inbox_root, "notes": self.notes_root, "outbox": self.outbox_root}[label]
        path = folder / self._safe_name(name)
        if path.is_symlink() or not path.is_file() or path.parent.resolve() != folder.resolve():
            raise PocketToolError("The workspace item failed its confinement check.")
        return str(chosen["id"]), path

    def read_for_model(self, item_id: str | None = None) -> dict[str, object]:
        logical_id, path = self._resolve_item(item_id)
        extension = path.suffix.casefold()
        if extension in IMAGE_EXTENSIONS:
            self._validate_signature(path, extension)
            self._validate_image(path)
            return {"id": logical_id, "kind": "image", "bytes": self._image_for_model(path)}
        if extension == ".pdf":
            return {"id": logical_id, "kind": "text", "text": self._read_pdf(path)}
        if extension in TEXT_EXTENSIONS:
            if path.stat().st_size > MAX_TEXT_BYTES:
                raise PocketToolError("The text file exceeds the 2 MB read limit.")
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise PocketToolError("The text file is not valid UTF-8.") from exc
            return {"id": logical_id, "kind": "text", "text": text[:MAX_TEXT_CHARS]}
        raise PocketToolError("That file type cannot be read.")

    @staticmethod
    def _image_for_model(path: Path) -> bytes:
        """Bound vision input cost without altering the user's imported copy."""
        try:
            from PIL import Image

            with Image.open(path) as image:
                image.thumbnail((1600, 1600))
                converted = image.convert("RGB")
                output = io.BytesIO()
                converted.save(output, format="JPEG", quality=88, optimize=True)
                data = output.getvalue()
            if not data or len(data) > 3 * 1024 * 1024:
                raise PocketToolError("The image could not be bounded for local vision.")
            return data
        except PocketToolError:
            raise
        except Exception as exc:
            raise PocketToolError("The image could not be prepared for local vision.") from exc

    @staticmethod
    def _read_pdf(path: Path) -> str:
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path), strict=True)
            if reader.is_encrypted:
                raise PocketToolError("Encrypted PDFs are not supported.")
            if len(reader.pages) > MAX_PDF_PAGES:
                raise PocketToolError(f"PDFs are limited to {MAX_PDF_PAGES} pages.")
            output: list[str] = []
            used = 0
            for index, page in enumerate(reader.pages, start=1):
                page_text = (page.extract_text() or "").strip()
                if not page_text:
                    continue
                remaining = MAX_TEXT_CHARS - used
                if remaining <= 0:
                    break
                chunk = page_text[:remaining]
                output.append(f"\n--- Page {index} ---\n{chunk}")
                used += len(chunk)
            text = "".join(output).strip()
            if not text:
                raise PocketToolError("The PDF contains no extractable text. Scanned-PDF OCR is not available yet.")
            return text
        except PocketToolError:
            raise
        except Exception as exc:
            raise PocketToolError(f"The PDF could not be parsed safely: {exc}") from exc

    def write_note(self, name: str, content: str, *, outbox: bool = False) -> str:
        cleaned = str(content).strip()
        if not cleaned or len(cleaned) > MAX_NOTE_CHARS:
            raise PocketToolError("Notes must contain 1 to 100,000 characters.")
        safe_name = self._safe_name(name, ".md")
        if Path(safe_name).suffix.casefold() not in {".md", ".txt"}:
            raise PocketToolError("Solomon Pocket AI may write only Markdown or text files.")
        folder = self._active_trusted_root() or (self.outbox_root if outbox else self.notes_root)
        destination = self._unique_destination(folder, safe_name)
        descriptor, temporary_name = tempfile.mkstemp(prefix=".solomon-", suffix=".tmp", dir=folder)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(cleaned + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        label = "trusted" if self.trusted_root else ("outbox" if outbox else "notes")
        return f"{label}/{destination.name}"

    def capture_camera(self, device_index: int = 0) -> tuple[str, bytes]:
        try:
            import cv2

            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
            camera = cv2.VideoCapture(device_index, backend)
            if not camera.isOpened():
                raise PocketToolError("The camera could not be opened.")
            frame = None
            try:
                for _ in range(8):
                    ok, candidate = camera.read()
                    if ok:
                        frame = candidate
                    time.sleep(0.03)
            finally:
                camera.release()
            if frame is None:
                raise PocketToolError("The camera did not return a frame.")
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                raise PocketToolError("The camera frame could not be encoded.")
            data = bytes(encoded)
            if len(data) > MAX_IMAGE_BYTES:
                raise PocketToolError("The camera snapshot exceeded the image limit.")
            name = time.strftime("camera-%Y%m%d-%H%M%S.jpg")
            destination = self._unique_destination(self.camera_root, name)
            temporary = destination.with_suffix(".jpg.tmp")
            temporary.write_bytes(data)
            os.replace(temporary, destination)
            return f"camera/{destination.name}", data
        except PocketToolError:
            raise
        except Exception as exc:
            raise PocketToolError(f"Camera capture failed: {exc}") from exc

    @staticmethod
    def _fetch_json(url: str, exact_host: str) -> dict[str, object]:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != exact_host:
            raise PocketToolError("The live-data request failed its network allowlist.")
        request = urllib.request.Request(url, headers={"User-Agent": "SolomonPocketAI/0.1 read-only-live-data"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=12) as response:
                if urllib.parse.urlparse(response.geturl()).hostname != exact_host:
                    raise PocketToolError("The weather service redirected outside its allowlist.")
                raw = response.read(512 * 1024 + 1)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise PocketToolError("Live weather is unavailable. Check the internet connection and try again.") from exc
        if len(raw) > 512 * 1024:
            raise PocketToolError("The weather response exceeded its size limit.")
        try:
            result = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PocketToolError("The weather service returned invalid data.") from exc
        if not isinstance(result, dict):
            raise PocketToolError("The weather service returned an unexpected response.")
        return result

    def current_weather(self, location: str) -> str:
        place = re.sub(r"\s+", " ", str(location)).strip(" .,?\t\r\n")
        if len(place) < 2 or len(place) > 100 or any(ord(char) < 32 for char in place):
            raise PocketToolError("Provide a city or postal code, for example: /weather Orlando, FL")
        geo_query = urllib.parse.urlencode({"name": place, "count": 1, "language": "en", "format": "json"})
        geo = self._fetch_json(
            f"https://geocoding-api.open-meteo.com/v1/search?{geo_query}",
            "geocoding-api.open-meteo.com",
        )
        results = geo.get("results")
        if not isinstance(results, list) or not results:
            raise PocketToolError(f"No weather location matched “{place}”.")
        match = results[0]
        if not isinstance(match, dict):
            raise PocketToolError("The weather location response was malformed.")
        latitude = float(match["latitude"])
        longitude = float(match["longitude"])
        forecast_query = urllib.parse.urlencode(
            {
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,apparent_temperature,precipitation,weather_code,cloud_cover,wind_speed_10m,wind_gusts_10m",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,sunrise,sunset",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "precipitation_unit": "inch",
                "forecast_days": 1,
                "timezone": "auto",
            }
        )
        forecast = self._fetch_json(
            f"https://api.open-meteo.com/v1/forecast?{forecast_query}",
            "api.open-meteo.com",
        )
        current = forecast.get("current")
        daily = forecast.get("daily")
        if not isinstance(current, dict) or not isinstance(daily, dict):
            raise PocketToolError("The weather forecast response was incomplete.")
        code = int(current.get("weather_code", -1))
        condition = {
            0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
            45: "foggy", 48: "freezing fog", 51: "light drizzle", 53: "drizzle",
            55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
            71: "light snow", 73: "snow", 75: "heavy snow", 80: "rain showers",
            81: "rain showers", 82: "heavy rain showers", 95: "thunderstorms",
        }.get(code, f"weather code {code}")
        name_parts = [str(match.get("name", place))]
        if match.get("admin1"):
            name_parts.append(str(match["admin1"]))
        if match.get("country"):
            name_parts.append(str(match["country"]))
        location_name = ", ".join(name_parts)

        def first(key: str, default: object = "unknown") -> object:
            value = daily.get(key)
            return value[0] if isinstance(value, list) and value else default

        return (
            f"Live weather from Open-Meteo for {location_name}.\n"
            f"Observed/model time: {current.get('time', 'unknown')} {forecast.get('timezone_abbreviation', '')}.\n"
            f"Now: {current.get('temperature_2m', 'unknown')} degrees Fahrenheit, feels like "
            f"{current.get('apparent_temperature', 'unknown')} degrees, {condition}.\n"
            f"Wind: {current.get('wind_speed_10m', 'unknown')} mph; gusts "
            f"{current.get('wind_gusts_10m', 'unknown')} mph. Cloud cover: "
            f"{current.get('cloud_cover', 'unknown')}%.\n"
            f"Today: high {first('temperature_2m_max')} degrees, low {first('temperature_2m_min')} degrees, "
            f"maximum precipitation chance {first('precipitation_probability_max')}%.\n"
            f"Sunrise {first('sunrise')}; sunset {first('sunset')}."
        )

    def current_fact(self, query: str) -> str:
        """Read up to three bounded English Wikipedia summaries for one query."""
        cleaned = re.sub(r"\s+", " ", str(query)).strip(" ?.\t\r\n")
        if len(cleaned) < 3 or len(cleaned) > 180 or any(ord(char) < 32 for char in cleaned):
            raise PocketToolError("Provide a short factual question after /fact.")
        parameters = urllib.parse.urlencode(
            {
                "action": "query",
                "generator": "search",
                "gsrsearch": cleaned,
                "gsrlimit": 3,
                "prop": "extracts|info",
                "exintro": 1,
                "explaintext": 1,
                "exchars": 1800,
                "inprop": "url",
                "redirects": 1,
                "format": "json",
                "formatversion": 2,
            }
        )
        result = self._fetch_json(
            f"https://en.wikipedia.org/w/api.php?{parameters}",
            "en.wikipedia.org",
        )
        query_result = result.get("query")
        pages = query_result.get("pages") if isinstance(query_result, dict) else None
        if not isinstance(pages, list) or not pages:
            raise PocketToolError("The limited live fact lookup found no matching Wikipedia pages.")
        sections = [
            f"Read-only live fact lookup at {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}."
        ]
        for page in pages[:3]:
            if not isinstance(page, dict):
                continue
            title = str(page.get("title", "Untitled"))
            extract = re.sub(r"\s+", " ", str(page.get("extract", ""))).strip()
            url = str(
                page.get("fullurl")
                or f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
            )
            sections.append(f"SOURCE: {title}\nURL: {url}\nEXTRACT: {extract or '(no summary available)'}")
        if len(sections) == 1:
            raise PocketToolError("The limited live fact lookup returned no readable summaries.")
        return "\n\n".join(sections)
