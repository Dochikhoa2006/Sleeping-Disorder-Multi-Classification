from contextlib import redirect_stderr
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from src.__main__ import main


class CliLauncherTests(unittest.TestCase):
    def test_help_without_third_party_packages_lists_commands(self):
        for arguments in ([], ["--help"]):
            result = subprocess.run([sys.executable, "-S", "-m", "src", *arguments], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("compare-models", result.stdout)
            self.assertIn("clean-batch", result.stdout)

    def test_arguments_and_exit_status_forwarded_and_argv_restored(self):
        arguments = ["launcher", "predict", "--input-json", "patient.json", "--help"]
        def run():
            self.assertEqual(sys.argv[1:], arguments[2:])
            raise SystemExit(17)
        module = Mock()
        module.main.side_effect = run
        with patch("sys.argv", arguments), patch("src.__main__.import_module", return_value=module) as load:
            with self.assertRaises(SystemExit) as raised:
                main()
            self.assertEqual(raised.exception.code, 17)
            self.assertIs(sys.argv, arguments)
            load.assert_called_once_with("src.predict")

    def test_unknown_command_never_imports_a_tool(self):
        with patch("sys.argv", ["launcher", "unknown"]), patch("src.__main__.import_module") as load, redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as raised:
                main()
            self.assertEqual(raised.exception.code, 2)
            load.assert_not_called()

    def test_real_converter_dispatch_returns_json(self):
        result = subprocess.run([sys.executable, "-m", "src", "convert", "--input-json", "examples/patient.json", "--to", "json"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [json.loads(Path("examples/patient.json").read_text())])

    def test_diary_dispatch_remains_stdlib_only_and_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.csv"
            result = subprocess.run([sys.executable, "-S", "-m", "src", "diary", "--file", str(path),
                                     "summary", "--end", "2026-11-17", "--days", "7"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["unrecorded_nights"], 7)
            self.assertFalse(path.exists())
