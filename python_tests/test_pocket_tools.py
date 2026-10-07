from __future__ import annotations

import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
from PIL import Image
from pypdf import PdfWriter

from solomon_pocket_tools import PocketToolError, SolomonPocketTools


class SolomonPocketToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.tools = SolomonPocketTools(self.root / "SolomonPocketAIData")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_import_is_copied_and_read_by_opaque_workspace_id(self) -> None:
        source = self.root / "outside.txt"
        source.write_text("original text", encoding="utf-8")
        item_id = self.tools.import_selected(source)
        source.write_text("changed outside", encoding="utf-8")
        result = self.tools.read_for_model(item_id)
        self.assertEqual("inbox/outside.txt", item_id)
        self.assertEqual("original text", result["text"])

    def test_paths_and_unsupported_writes_fail_closed(self) -> None:
        with self.assertRaises(PocketToolError):
            self.tools.write_note("../escape.md", "no")
        with self.assertRaises(PocketToolError):
            self.tools.write_note("program.exe", "no")
        with self.assertRaises(PocketToolError):
            self.tools.read_for_model("../../outside.txt")

    def test_note_writes_only_inside_notes(self) -> None:
        item_id = self.tools.write_note("idea.md", "A local note")
        self.assertEqual("notes/idea.md", item_id)
        self.assertEqual("A local note\n", (self.tools.notes_root / "idea.md").read_text(encoding="utf-8"))

    def test_image_is_validated_and_bounded_for_local_vision(self) -> None:
        source = self.root / "photo.png"
        Image.new("RGB", (32, 24), color=(20, 40, 60)).save(source)
        item_id = self.tools.import_selected(source)
        result = self.tools.read_for_model(item_id)
        self.assertEqual("image", result["kind"])
        self.assertLess(len(result["bytes"]), 3 * 1024 * 1024)
        self.assertTrue(bytes(result["bytes"]).startswith(b"\xff\xd8\xff"))

    def test_pdf_parser_rejects_document_without_extractable_text(self) -> None:
        source = self.root / "blank.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        with source.open("wb") as output:
            writer.write(output)
        item_id = self.tools.import_selected(source)
        with self.assertRaisesRegex(PocketToolError, "no extractable text"):
            self.tools.read_for_model(item_id)

    def test_network_adapter_is_exact_host_allowlisted(self) -> None:
        with self.assertRaisesRegex(PocketToolError, "allowlist"):
            self.tools._fetch_json("https://example.com/weather", "api.open-meteo.com")

    def test_camera_is_one_shot_and_released(self) -> None:
        class FakeCamera:
            released = False

            def isOpened(self):
                return True

            def read(self):
                return True, np.zeros((12, 16, 3), dtype=np.uint8)

            def release(self):
                self.released = True

        camera = FakeCamera()
        fake_cv2 = type(
            "FakeCv2",
            (),
            {
                "CAP_DSHOW": 1,
                "CAP_ANY": 0,
                "IMWRITE_JPEG_QUALITY": 1,
                "VideoCapture": staticmethod(lambda *_args: camera),
                "imencode": staticmethod(
                    lambda *_args: (True, np.frombuffer(b"\xff\xd8\xffmock-jpeg", dtype=np.uint8))
                ),
            },
        )
        with mock.patch.dict("sys.modules", {"cv2": fake_cv2}):
            item_id, data = self.tools.capture_camera()
        self.assertTrue(camera.released)
        self.assertTrue(item_id.startswith("camera/camera-"))
        self.assertTrue(data.startswith(b"\xff\xd8\xff"))


if __name__ == "__main__":
    unittest.main()
