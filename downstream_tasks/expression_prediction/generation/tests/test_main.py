"""Command-line entry-point tests."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import run_generation


class MainTests(unittest.TestCase):
    """Verify the intentionally small command-line surface."""

    def test_parse_args_requires_only_config(self):
        with patch("sys.argv", ["run_generation.py", "--config", "experiment.yaml"]):
            args = run_generation.parse_args()
        self.assertEqual(args.config, Path("experiment.yaml"))

    def test_main_loads_config_runs_and_lists_outputs(self):
        config = object()
        output = Path("result.json")
        with (
            patch("sys.argv", ["run_generation.py", "--config", "experiment.yaml"]),
            patch("run_generation.RunConfig.load", return_value=config) as load,
            patch("run_generation.run_experiment", return_value=[output]) as run,
            patch("builtins.print") as printer,
        ):
            run_generation.main()
        load.assert_called_once_with(Path("experiment.yaml"))
        run.assert_called_once_with(config)
        printer.assert_any_call("Written files:")
        printer.assert_any_call(f"  {output}")


if __name__ == "__main__":
    unittest.main()
