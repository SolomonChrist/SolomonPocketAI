from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

import SolomonPocketAI as app


class AudioSettingsTest(unittest.TestCase):
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
