import importlib
import pathlib
import sys
import unittest

import numpy as np

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
    def __init__(self, sampling_rate: int = 16000, window_duration: int = 1, backward_window_duration: float = 30.0):
        self.sampling_rate = sampling_rate
        self.window_duration = window_duration
        self.backward_window_duration = backward_window_duration


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
        prev_array = self.state.rolling_window.window_audio().copy()

        self.state.advance_audio_window(0)
        self.assertEqual(self.state.window_ts, prev_window_ts)
        self.assertEqual(self.state.audio_ts, prev_audio_ts)
        self.assertTrue(np.array_equal(self.state.rolling_window.window_audio(), prev_array))

        self.state.advance_audio_window(-10)
        self.assertEqual(self.state.window_ts, prev_window_ts)
        self.assertEqual(self.state.audio_ts, prev_audio_ts)
        self.assertTrue(np.array_equal(self.state.rolling_window.window_audio(), prev_array))

    def test_clamp_when_offset_exceeds_available_audio(self):
        # Fill with a known pattern
        chunk = np.arange(400, dtype=np.float32)
        self.state.append_audio_to_window(chunk)

        # Request an advance larger than available (available == 400)
        self.state.advance_audio_window(1000)

        # Window should not advance beyond audio_ts
        self.assertEqual(self.state.window_ts, self.state.audio_ts)
        # After clamped advance, no residual audio remains in the window
        self.assertEqual(float(self.state.rolling_window.window_audio().sum()), 0.0)
        # Invariant: non-negative remaining window audio
        self.assertGreaterEqual(self.state.audio_ts - self.state.window_ts, 0)

    def test_partial_advance_shifts_audio_left_and_zeros_tail(self):
        # Pattern 0..999 in the beginning of the window
        chunk = np.arange(1000, dtype=np.float32)
        self.state.append_audio_to_window(chunk)

        self.state.advance_audio_window(200)
        # Expect positions [0:800] to contain 200..999, rest zeros
        arr = self.state.rolling_window.window_audio()
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
        self.assertTrue(np.array_equal(self.state.rolling_window.window_audio()[0:100], new_chunk))
        # Window now contains exactly the new chunk
        self.assertEqual(self.state.audio_ts - self.state.window_ts, 100)

    def test_absolute_range_tracks_active_start(self):
        self.state.append_audio_to_window(np.arange(10, dtype=np.float32))
        self.state.advance_audio_window(3)
        self.assertTrue(np.array_equal(self.state.active_audio(), np.arange(3, 10, dtype=np.float32)))
        self.assertTrue(np.array_equal(self.state.audio_between(5, 8), np.arange(5, 8, dtype=np.float32)))
        self.assertTrue(np.array_equal(self.state.audio_between(2, 8), np.arange(2, 8, dtype=np.float32)))
        with self.assertRaisesRegex(ValueError, "outside the retained window"):
            self.state.audio_between(-1, 8)
        with self.assertRaisesRegex(ValueError, "outside the retained window"):
            self.state.audio_between(5, 11)

    def test_retains_thirty_seconds_without_changing_active_audio(self):
        state = verbatim_local.State(DummyConfig(sampling_rate=10, window_duration=30))
        state.append_audio_to_window(np.arange(300, dtype=np.float32))
        self.assertEqual(300, len(state.window_audio()))
        self.assertTrue(np.array_equal(state.active_audio(), np.arange(300, dtype=np.float32)))

        state.advance_audio_window(200)
        state.append_audio_to_window(np.arange(300, 500, dtype=np.float32))
        state.advance_audio_window(200)

        self.assertEqual(100, state.retained_start_ts)
        self.assertEqual(400, state.window_ts)
        self.assertEqual(500, state.audio_ts)
        self.assertTrue(np.array_equal(state.audio_between(100, 400), np.arange(100, 400, dtype=np.float32)))
        self.assertTrue(np.array_equal(state.active_audio(), np.arange(400, 500, dtype=np.float32)))
        self.assertTrue(np.array_equal(state.window_audio()[:100], state.active_audio()))
        self.assertTrue(np.all(state.window_audio()[100:] == 0))
        with self.assertRaisesRegex(ValueError, "outside the retained window"):
            state.audio_between(99, 100)

    def test_retained_audio_duration_is_independent_of_active_window(self):
        state = verbatim_local.State(DummyConfig(sampling_rate=10, window_duration=30, backward_window_duration=5.0))
        state.append_audio_to_window(np.arange(300, dtype=np.float32))
        state.advance_audio_window(200)
        state.append_audio_to_window(np.arange(300, 500, dtype=np.float32))
        state.advance_audio_window(200)

        self.assertEqual(50, state.rolling_window.history_capacity)
        self.assertEqual(350, state.retained_start_ts)
        self.assertTrue(np.array_equal(state.audio_between(350, 400), np.arange(350, 400, dtype=np.float32)))
        with self.assertRaisesRegex(ValueError, "outside the retained window"):
            state.audio_between(349, 400)

    def test_zero_retained_audio_keeps_only_active_samples(self):
        state = verbatim_local.State(DummyConfig(sampling_rate=10, window_duration=30, backward_window_duration=0.0))
        state.append_audio_to_window(np.arange(300, dtype=np.float32))
        state.advance_audio_window(200)

        self.assertEqual(0, state.rolling_window.history_capacity)
        self.assertEqual(state.window_ts, state.retained_start_ts)
        self.assertTrue(np.array_equal(state.active_audio(), np.arange(200, 300, dtype=np.float32)))
        with self.assertRaisesRegex(ValueError, "outside the retained window"):
            state.audio_between(199, 200)

        state.append_audio_to_window(np.arange(300, 500, dtype=np.float32))
        self.assertTrue(np.array_equal(state.active_audio(), np.arange(200, 500, dtype=np.float32)))
        state.append_audio_to_window(np.arange(500, 550, dtype=np.float32))
        self.assertEqual(250, state.retained_start_ts)
        self.assertTrue(np.array_equal(state.active_audio(), np.arange(250, 550, dtype=np.float32)))
        with self.assertRaisesRegex(ValueError, "outside the retained window"):
            state.audio_between(249, 250)

    def test_long_silence_keeps_only_recent_audio(self):
        state = verbatim_local.State(DummyConfig(sampling_rate=10, window_duration=30))
        state.append_audio_to_window(np.ones(300, dtype=np.float32))
        state.advance_audio_window(300)
        state.append_audio_to_window(np.zeros(300, dtype=np.float32))
        state.advance_audio_window(300)

        self.assertEqual(300, state.retained_start_ts)
        self.assertTrue(np.all(state.audio_between(300, 600) == 0))
        with self.assertRaisesRegex(ValueError, "outside the retained window"):
            state.audio_between(299, 300)

    def test_append_overflow_keeps_active_view_and_retains_shifted_audio(self):
        state = verbatim_local.State(DummyConfig(sampling_rate=10, window_duration=30))
        state.append_audio_to_window(np.arange(300, dtype=np.float32))
        state.append_audio_to_window(np.arange(300, 350, dtype=np.float32))

        self.assertEqual(50, state.window_ts)
        self.assertEqual(0, state.retained_start_ts)
        self.assertEqual(300, len(state.active_audio()))
        self.assertTrue(np.array_equal(state.active_audio(), np.arange(50, 350, dtype=np.float32)))
        self.assertTrue(np.array_equal(state.audio_between(0, 350), np.arange(350, dtype=np.float32)))

    def test_window_rejects_out_of_capacity_access(self):
        window = self.state.rolling_window
        with self.assertRaisesRegex(ValueError, "exceeds window capacity"):
            window.active_audio(window.capacity + 1)
        with self.assertRaisesRegex(ValueError, "exceeds window capacity"):
            window.write(window.capacity, np.ones(1, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
