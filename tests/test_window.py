import unittest
import numpy as np
import sys
import importlib
import pathlib


# Ensure our workspace version of the package is imported, not any installed one.
REPO_ROOT = str((pathlib.Path(__file__).parent.parent).resolve())
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Purge any previously imported 'verbatim' from site-packages
for mod in list(sys.modules.keys()):
    if mod == "verbatim" or mod.startswith("verbatim."):
        del sys.modules[mod]

verbatim_local = importlib.import_module("verbatim.verbatim")


class DummyConfig:
    def __init__(self, sampling_rate: int = 16000, window_duration: int = 1):
        self.sampling_rate = sampling_rate
        self.window_duration = window_duration


class TestAdvanceAudioWindow(unittest.TestCase):
    def setUp(self):
        # Small window for faster tests
        self.config = DummyConfig(sampling_rate=16000, window_duration=1)
        self.state = verbatim_local.State(self.config)

    def test_noop_when_offset_non_positive(self):
        # Seed some audio so the buffer isn't empty
        chunk = np.full(100, 0.5, dtype=np.float32)
        self.state.append_audio_to_window(chunk)

        prev_window_ts = self.state.window_ts
        prev_audio_ts = self.state.audio_ts
        prev_array = self.state.rolling_window.array.copy()

        self.state.advance_audio_window(0)
        self.assertEqual(self.state.window_ts, prev_window_ts)
        self.assertEqual(self.state.audio_ts, prev_audio_ts)
        self.assertTrue(np.array_equal(self.state.rolling_window.array, prev_array))

        self.state.advance_audio_window(-10)
        self.assertEqual(self.state.window_ts, prev_window_ts)
        self.assertEqual(self.state.audio_ts, prev_audio_ts)
        self.assertTrue(np.array_equal(self.state.rolling_window.array, prev_array))

    def test_clamp_when_offset_exceeds_available_audio(self):
        # Fill with a known pattern
        chunk = np.arange(400, dtype=np.float32)
        self.state.append_audio_to_window(chunk)

        # Request an advance larger than available (available == 400)
        self.state.advance_audio_window(1000)

        # Window should not advance beyond audio_ts
        self.assertEqual(self.state.window_ts, self.state.audio_ts)
        # After clamped advance, no residual audio remains in the window
        self.assertEqual(float(self.state.rolling_window.array.sum()), 0.0)
        # Invariant: non-negative remaining window audio
        self.assertGreaterEqual(self.state.audio_ts - self.state.window_ts, 0)

    def test_partial_advance_shifts_audio_left_and_zeros_tail(self):
        # Pattern 0..999 in the beginning of the window
        chunk = np.arange(1000, dtype=np.float32)
        self.state.append_audio_to_window(chunk)

        self.state.advance_audio_window(200)
        # Expect positions [0:800] to contain 200..999, rest zeros
        arr = self.state.rolling_window.array
        self.assertTrue(np.array_equal(arr[0:800], np.arange(200, 1000, dtype=np.float32)))
        self.assertTrue(np.all(arr[800:] == 0))
        self.assertEqual(self.state.window_ts, 200)
        self.assertEqual(self.state.audio_ts, 1000)

    def test_append_after_clamped_advance(self):
        # Ingest 500 samples, then advance beyond available (clamped)
        self.state.append_audio_to_window(np.ones(500, dtype=np.float32))
        self.state.advance_audio_window(2000)  # clamps to 500
        self.assertEqual(self.state.window_ts, self.state.audio_ts)

        # Now append new audio and ensure it inserts at start without error
        new_chunk = np.full(100, 2.0, dtype=np.float32)
        self.state.append_audio_to_window(new_chunk)

        # Check insertion at the beginning
        self.assertTrue(np.array_equal(self.state.rolling_window.array[0:100], new_chunk))
        # Window now contains exactly the new chunk
        self.assertEqual(self.state.audio_ts - self.state.window_ts, 100)


if __name__ == "__main__":
    unittest.main()
