import logging
import os
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from gena_expression.inference.model import _log_git_commit_comparison


class GitCommitComparisonTests(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger(f"{__name__}.{self._testMethodName}")

    def test_matching_commit_is_logged(self):
        with patch.dict(os.environ, {"GENALM_HOME": "/genalm"}), patch(
            "gena_expression.inference.model.subprocess.check_output",
            return_value=b"abc123\n",
        ) as check_output:
            with self.assertLogs(self.logger, level="INFO") as captured:
                _log_git_commit_comparison({"git_commit_hash": "abc123"}, self.logger)

        check_output.assert_called_once_with(
            ["git", "rev-parse", "HEAD"],
            cwd=Path("/genalm") / "GENA_LM",
            stderr=subprocess.DEVNULL,
        )
        self.assertIn("matches model config", captured.output[0])

    def test_mismatching_commit_warns_without_raising(self):
        with patch.dict(os.environ, {"GENALM_HOME": "/genalm"}), patch(
            "gena_expression.inference.model.subprocess.check_output",
            return_value=b"local123\n",
        ):
            with self.assertLogs(self.logger, level="WARNING") as captured:
                _log_git_commit_comparison({"git_commit_hash": "trained456"}, self.logger)

        self.assertIn("does not match", captured.output[0])

    def test_missing_config_commit_warns_without_running_git(self):
        with patch(
            "gena_expression.inference.model.subprocess.check_output"
        ) as check_output, self.assertLogs(self.logger, level="WARNING") as captured:
            _log_git_commit_comparison({}, self.logger)

        check_output.assert_not_called()
        self.assertIn("does not contain", captured.output[0])

    def test_missing_genalm_home_warns(self):
        with patch.dict(os.environ, {}, clear=True), self.assertLogs(
            self.logger, level="WARNING"
        ) as captured:
            _log_git_commit_comparison({"git_commit_hash": "abc123"}, self.logger)

        self.assertIn("GENALM_HOME is not set", captured.output[0])

    def test_git_failure_warns_without_raising(self):
        with patch.dict(os.environ, {"GENALM_HOME": "/genalm"}), patch(
            "gena_expression.inference.model.subprocess.check_output",
            side_effect=subprocess.CalledProcessError(128, ["git"]),
        ):
            with self.assertLogs(self.logger, level="WARNING") as captured:
                _log_git_commit_comparison({"git_commit_hash": "abc123"}, self.logger)

        self.assertIn("Could not check", captured.output[0])

    def test_missing_git_executable_warns_without_raising(self):
        with patch.dict(os.environ, {"GENALM_HOME": "/genalm"}), patch(
            "gena_expression.inference.model.subprocess.check_output",
            side_effect=FileNotFoundError("git was not found"),
        ):
            with self.assertLogs(self.logger, level="WARNING") as captured:
                _log_git_commit_comparison({"git_commit_hash": "abc123"}, self.logger)

        self.assertIn("git was not found", captured.output[0])


if __name__ == "__main__":
    unittest.main()
