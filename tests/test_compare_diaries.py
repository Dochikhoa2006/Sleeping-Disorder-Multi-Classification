import csv
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.compare_diaries import compare_diaries
from src.diary import COLUMNS


class CompareDiariesTests(unittest.TestCase):
    def batch(self, entries):
        buffer = StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows({"date": day, "bedtime": "23:00", "wake_time": "07:00",
                          "awake_minutes": "30", "quality": "8", **changes} for day, changes in entries)
        return buffer.getvalue().encode("utf-8-sig")

    def test_all_change_types_sorted_without_times_or_ratings(self):
        before = self.batch([("2026-11-03", {}), ("2026-11-01", {}), ("2026-11-02", {})])
        after = self.batch([("2026-11-04", {}), ("2026-11-03", {}),
                            ("2026-11-02", {"bedtime": "22:30", "quality": "7"})])
        report = compare_diaries(before, after)
        self.assertEqual(report["counts"], {"added": 1, "removed": 1, "changed": 1, "unchanged": 1})
        self.assertEqual(report["changes"], [{"date": "2026-11-02", "fields": ["bedtime", "quality"]}])
        self.assertEqual(report["added_dates"], ["2026-11-04"])
        self.assertEqual(report["removed_dates"], ["2026-11-01"])
        self.assertNotIn("22:30", json.dumps(report))
        self.assertNotEqual(report["baseline_sha256"], report["candidate_sha256"])

    def test_reordering_is_unchanged_but_byte_hashes_differ(self):
        rows = [("2026-11-01", {}), ("2026-11-02", {})]
        report = compare_diaries(self.batch(rows), self.batch(list(reversed(rows))))
        self.assertEqual(report["counts"]["unchanged"], 2)
        self.assertEqual(report["changes"], [])
        self.assertNotEqual(report["baseline_sha256"], report["candidate_sha256"])

    def test_empty_backups_and_invalid_duplicates(self):
        empty = self.batch([])
        self.assertEqual(compare_diaries(empty, empty)["baseline_nights"], 0)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            compare_diaries(empty, self.batch([("2026-11-01", {})] * 2))
        with self.assertRaises(ValueError):
            compare_diaries(empty, b"bad header")

    def test_stdlib_cli_reports_before_failure_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline.csv"
            candidate = baseline.with_name("candidate.csv")
            output = baseline.with_name("comparison.json")
            before, after = self.batch([]), self.batch([("2026-11-01", {})])
            baseline.write_bytes(before)
            candidate.write_bytes(after)
            command = [sys.executable, "-S", "-m", "src.compare_diaries", "--baseline", str(baseline), "--candidate", str(candidate)]
            result = subprocess.run([*command, "--fail-on-change", "--output", str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(json.loads(output.read_text())["counts"]["added"], 1)
            self.assertEqual(baseline.read_bytes(), before)
            self.assertEqual(candidate.read_bytes(), after)
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
            result = subprocess.run([*command, "--output", str(baseline)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(baseline.read_bytes(), before)
            saved = output.read_bytes()
            candidate.write_text("invalid")
            result = subprocess.run([*command, "--output", str(output)], capture_output=True, text=True)
            self.assertIn("Error:", result.stderr)
            self.assertEqual(output.read_bytes(), saved)
