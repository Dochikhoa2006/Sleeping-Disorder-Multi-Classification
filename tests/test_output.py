import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.output import atomic_text_output
from src.predict import write_prediction_csv


class OutputTests(unittest.TestCase):
    def test_destination_changes_only_after_successful_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text("old result", encoding="utf-8")
            with atomic_text_output(path) as output_file:
                output_file.write('["Healthy"]\n')
                self.assertEqual(path.read_text(encoding="utf-8"), "old result")
            self.assertEqual(path.read_text(encoding="utf-8"), '["Healthy"]\n')
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_failed_write_preserves_existing_or_absent_destination(self) -> None:
        for existing in (False, True):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "result.json"
                if existing:
                    path.write_bytes(b"old result")
                with self.assertRaisesRegex(OSError, "Simulated write failure"):
                    with atomic_text_output(path) as output_file:
                        output_file.write("partial result")
                        raise OSError("Simulated write failure")
                if existing:
                    self.assertEqual(path.read_bytes(), b"old result")
                else:
                    self.assertFalse(path.exists())
                self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_failed_replace_preserves_destination_and_cleans_up(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_bytes(b"old result")
            with patch.object(Path, "replace", side_effect=OSError("Simulated replace failure")):
                with self.assertRaisesRegex(OSError, "Simulated replace failure"):
                    with atomic_text_output(path) as output_file:
                        output_file.write("new result")
            self.assertEqual(path.read_bytes(), b"old result")
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_csv_length_mismatch_does_not_publish_partial_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.csv"
            path.write_bytes(b"old CSV result")
            with self.assertRaises(ValueError):
                write_prediction_csv(path, ["Record ID"],
                                     [{"Record ID": "1"}, {"Record ID": "2"}], ["Healthy"])
            self.assertEqual(path.read_bytes(), b"old CSV result")
            self.assertEqual(list(path.parent.iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
