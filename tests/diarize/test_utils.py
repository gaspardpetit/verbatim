import unittest

from verbatim_diarization.utils import sanitize_uri_component


class SanitizeUriComponentTest(unittest.TestCase):
    def test_general_cases(self):
        cases = [
            ("Av Galilée", "Av_Galilée"),
            ("leading--spaces  ", "leading--spaces"),
            ("upper/lower\\mix", "upper_lower_mix"),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(sanitize_uri_component(raw), expected)

    def test_fallback_and_length(self):
        self.assertEqual(sanitize_uri_component(""), "audio")
        long_value = "a" * 300
        self.assertEqual(sanitize_uri_component(long_value), "a" * 255)


if __name__ == "__main__":
    unittest.main()
