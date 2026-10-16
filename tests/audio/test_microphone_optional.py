import builtins
import unittest
from unittest.mock import MagicMock, patch

from verbatim_audio.sources.factory import create_audio_sources


class TestOptionalMicrophoneDependencies(unittest.TestCase):
    def test_missing_microphone_dependency_has_install_hint(self):
        original_import = builtins.__import__

        def missing_pyaudio(name, globals_=None, locals_=None, fromlist=(), level=0):
            if name.endswith("micaudiosource"):
                raise ModuleNotFoundError("No module named 'pyaudio'", name="pyaudio")
            return original_import(name, globals_, locals_, fromlist, level)

        with patch("builtins.__import__", side_effect=missing_pyaudio):
            with self.assertRaisesRegex(RuntimeError, "verbatim\\[microphone\\]"):
                create_audio_sources(input_source=">", device="cpu", cache=MagicMock())


if __name__ == "__main__":
    unittest.main()
