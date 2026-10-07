from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np

import SolomonPocketAI as app


class AudioSettingsTest(unittest.TestCase):
    def test_interface_font_scale_is_bounded_and_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            settings_file = Path(temporary) / "settings.json"
            settings = dict(app.DEFAULT_SETTINGS)
            settings["interface_scale"] = 1.3
            with (
                mock.patch.object(app, "SETTINGS_FILE", settings_file),
                mock.patch.object(app, "ensure_local_layout"),
            ):
                app.save_settings(settings)
                self.assertEqual(1.3, app.load_settings()["interface_scale"])
                settings["interface_scale"] = 4.0
                app.save_settings(settings)
                self.assertEqual(1.0, app.load_settings()["interface_scale"])

    def test_trusted_folder_setting_is_local_and_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            settings_file = temporary_path / "settings.json"
            trusted_folder = temporary_path / "Trusted Files"
            trusted_folder.mkdir()
            settings = dict(app.DEFAULT_SETTINGS)
            settings["trusted_folder"] = str(trusted_folder)
            with (
                mock.patch.object(app, "SETTINGS_FILE", settings_file),
                mock.patch.object(app, "ensure_local_layout"),
            ):
                app.save_settings(settings)
                self.assertEqual(str(trusted_folder), app.load_settings()["trusted_folder"])

    def test_font_size_scaling_preserves_pixel_font_sign(self) -> None:
        self.assertEqual(13, app._scaled_font_size(10, 1.3))
        self.assertEqual(-13, app._scaled_font_size(-10, 1.3))

    def test_replay_cache_keeps_only_three_newest_responses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            replay_root = Path(temporary)
            with mock.patch.object(app, "REPLAY_ROOT", replay_root):
                for number in range(1, 5):
                    app.save_replay_response(
                        f"Response {number}",
                        np.full(240, number, dtype=np.float32),
                        24_000,
                    )
                items = app.load_replay_responses()
        self.assertEqual(["Response 4", "Response 3", "Response 2"], [item["text"] for item in items])
        self.assertEqual(3, len(items))

    def test_streaming_speech_captures_original_audio_for_replay(self) -> None:
        engines = mock.Mock()
        engines.synthesize.return_value = (np.ones(240, dtype=np.float32), 24_000)
        speaker = app.StreamingSpeech(engines)
        speaker.feed("This short answer should be replayable.")
        speaker.finish()
        captured = speaker.captured_audio()
        self.assertIsNotNone(captured)
        samples, sample_rate = captured
        self.assertEqual(240, samples.size)
        self.assertEqual(24_000, sample_rate)
        engines.play.assert_called_once()

    def test_chinese_voice_uses_mandarin_code_and_preview(self) -> None:
        self.assertEqual("cmn", app.voice_language("zf_xiaoni"))
        preview = app.voice_preview_text("zf_xiaoni")
        self.assertIn("中文语音测试", preview)

    def test_resample_converts_kokoro_rate_for_realtek_style_output(self) -> None:
        samples = np.zeros(24_000, dtype=np.float32)
        converted = app.resample_audio(samples, 24_000, 44_100)
        self.assertEqual(44_100, converted.size)
        self.assertEqual(np.float32, converted.dtype)

    def test_missing_system_default_prefers_named_speakers(self) -> None:
        options = [
            ("System default (not currently available)", None, None),
            ("Headphones [Windows MME]", "output|Windows MME|Headphones", 5),
            (
                "Speakers (Realtek HD Audio output) [Windows MME]",
                "output|Windows MME|Speakers (Realtek HD Audio output)",
                8,
            ),
        ]
        with mock.patch.object(app, "audio_device_options", return_value=options):
            self.assertEqual(8, app.resolve_output_device(None))

    def test_stable_device_id_resolves_current_index(self) -> None:
        device_id = "output|Windows DirectSound|ARZOPA (NVIDIA High Definition Audio)"
        options = [
            ("System default - ARZOPA", None, None),
            ("ARZOPA [Windows DirectSound]", device_id, 8),
        ]
        with mock.patch.object(app, "audio_device_options", return_value=options):
            self.assertEqual(8, app.resolve_output_device(device_id))

    def test_voice_playback_resamples_to_selected_device_rate(self) -> None:
        device_id = "output|Windows DirectSound|ARZOPA (NVIDIA High Definition Audio)"
        engine = app.VoiceEngines(output_device=device_id)
        samples = np.zeros(24_000, dtype=np.float32)
        with (
            mock.patch.object(app, "resolve_output_device", return_value=8),
            mock.patch.object(app, "supported_audio_rate", return_value=44_100),
            mock.patch.object(app.sd, "play") as play,
        ):
            engine.play(samples, 24_000)
        played_samples, played_rate = play.call_args.args[:2]
        self.assertEqual(44_100, played_rate)
        self.assertEqual(44_100, played_samples.size)
        self.assertEqual(8, play.call_args.kwargs["device"])


if __name__ == "__main__":
    unittest.main()
