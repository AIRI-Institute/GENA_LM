"""Focused tests for configurable condition-description formatting."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from collections import UserDict
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from gena_expression.conditions import Condition, resolve_description_formatter
from gena_expression.inference.description import (
    get_configured_description_formatter,
    get_make_description_from_json,
)
from gena_expression.inference import SequenceModel


class DescriptionFormatterTests(unittest.TestCase):
    """Verify runtime configuration, model isolation, and callable adaptation."""

    def tearDown(self) -> None:
        """Keep the process-wide formatter isolated between tests."""

        Condition.clear_description_formatter()

    def test_structured_description_requires_an_explicit_configuration(self):
        """There is no implicit metadata formatter fallback."""

        condition = Condition(name="K562", description={"cell_type": "K562"})
        with self.assertRaisesRegex(ValueError, "No description formatter"):
            condition.text()
        with self.assertRaisesRegex(ValueError, "No description formatter"):
            SequenceModel._resolve_description_formatter(None)

        self.assertEqual(
            Condition(name="K562", description="cell type is K562.").text(),
            "cell type is K562.",
        )

    def test_runtime_formatter_receives_dict_and_none_for_extra_parameters(self):
        """Fixed parameters after metadata receive None, including keyword-only ones."""

        calls = []

        def formatter(meta, description_id="unused", *, meta_path="unused"):
            calls.append((meta, description_id, meta_path))
            return f"cell type is {meta['cell_type']}."

        Condition.set_description_formatter(formatter)
        condition = Condition(
            name="K562",
            description=UserDict({"cell_type": "K562"}),
        )

        self.assertEqual(condition.text(), "cell type is K562.")
        self.assertIsInstance(calls[0][0], dict)
        self.assertIsNone(calls[0][1])
        self.assertIsNone(calls[0][2])

    def test_none_clears_the_runtime_formatter(self):
        """Passing None to the runtime setter restores the unconfigured state."""

        Condition.set_description_formatter(lambda meta: str(meta["cell_type"]))
        Condition.set_description_formatter(None)

        self.assertIsNone(Condition.get_description_formatter())
        with self.assertRaisesRegex(ValueError, "No description formatter"):
            SequenceModel._resolve_description_formatter(None)

    def test_model_snapshots_runtime_formatter_and_uses_it_for_grouping(self):
        """Later notebook configuration changes do not alter an existing model."""

        Condition.set_description_formatter(lambda meta: f"first:{meta['value']}")
        model = SequenceModel.__new__(SequenceModel)
        model.description_formatter = SequenceModel._resolve_description_formatter(None)
        condition = Condition(name="condition", description={"value": "x"})

        Condition.set_description_formatter(lambda meta: f"second:{meta['value']}")

        self.assertEqual(condition.text(), "second:x")
        self.assertEqual(model.make_description(condition), "first:x")
        self.assertEqual(model._grouping_key_for_condition(condition), "first:x")
        self.assertEqual(model.make_description_from_json({"value": "x"}), "first:x")

    def test_per_model_formatter_overrides_runtime_formatter(self):
        """An explicit model formatter takes precedence over runtime configuration."""

        Condition.set_description_formatter(lambda meta: "runtime")
        model = SequenceModel.__new__(SequenceModel)
        model.description_formatter = SequenceModel._resolve_description_formatter(
            lambda meta: "model"
        )

        self.assertEqual(model.make_description({"value": "x"}), "model")
        self.assertEqual(model.make_description("already rendered"), "already rendered")

    def test_explicit_and_runtime_formatters_bypass_config_loading(self):
        """Config inspection is only the final fallback in the precedence chain."""

        with patch(
            "gena_expression.inference.model.get_configured_description_formatter"
        ) as from_config:
            explicit = SequenceModel._resolve_description_formatter(
                lambda meta: "explicit",
                config_path="model.yaml",
            )
            self.assertEqual(explicit({}), "explicit")
            from_config.assert_not_called()

            Condition.set_description_formatter(lambda meta: "runtime")
            runtime = SequenceModel._resolve_description_formatter(
                None,
                config_path="model.yaml",
            )
            self.assertEqual(runtime({}), "runtime")
            from_config.assert_not_called()

    def test_model_resolution_adapts_the_configured_dataset_formatter(self):
        """The config fallback uses the same dict-and-None invocation contract."""

        calls = []

        def legacy_formatter(meta, description_id, meta_path):
            calls.append((meta, description_id, meta_path))
            return f"configured:{meta['cell_type']}"

        with patch(
            "gena_expression.inference.model.get_configured_description_formatter",
            return_value=legacy_formatter,
        ) as from_config:
            formatter = SequenceModel._resolve_description_formatter(
                None,
                config_path="model.yaml",
            )

        self.assertEqual(formatter(UserDict({"cell_type": "K562"})), "configured:K562")
        self.assertIsInstance(calls[0][0], dict)
        self.assertIsNone(calls[0][1])
        self.assertIsNone(calls[0][2])
        from_config.assert_called_once_with("model.yaml")

    def test_genalm_root_is_visible_only_while_dataset_class_is_loaded(self):
        """The automatic loader temporarily exposes GENALM_HOME/GENA_LM."""

        with tempfile.TemporaryDirectory() as temp_dir:
            genalm_root = (Path(temp_dir) / "GENA_LM").resolve()
            genalm_root.mkdir()
            root_text = str(genalm_root)

            def load_from_config(config_path):
                self.assertEqual(config_path, "model.yaml")
                self.assertEqual(sys.path[0], root_text)
                return lambda meta, description_id, meta_path: "configured"

            with patch.dict(os.environ, {"GENALM_HOME": temp_dir}), patch(
                "gena_expression.inference.description.get_make_description_from_json",
                side_effect=load_from_config,
            ):
                formatter = get_configured_description_formatter("model.yaml")

            self.assertNotIn(root_text, sys.path)
            self.assertEqual(formatter({}, None, None), "configured")

    def test_dataset_formatter_is_read_from_the_first_train_config(self):
        """Hydra resolves the selected class without constructing the dataset."""

        class Dataset:
            @staticmethod
            def make_description_from_json(meta, description_id, meta_path):
                return "configured"

        targets = []
        hydra_module = ModuleType("hydra")
        hydra_utils = ModuleType("hydra.utils")

        def get_class(target):
            targets.append(target)
            return Dataset

        hydra_utils.get_class = get_class
        hydra_module.utils = hydra_utils

        omegaconf = ModuleType("omegaconf")

        class FakeOmegaConf:
            @staticmethod
            def load(config_path):
                self.assertEqual(config_path, "model.yaml")
                return {
                    "valid_dataset_first": {"_target_": "pkg.ValidDataset"},
                    "train_dataset_selected": {"_target_": "pkg.TrainDataset"},
                }

            @staticmethod
            def select(config, key):
                return config.get(key)

        omegaconf.DictConfig = dict
        omegaconf.OmegaConf = FakeOmegaConf

        with patch.dict(
            sys.modules,
            {
                "hydra": hydra_module,
                "hydra.utils": hydra_utils,
                "omegaconf": omegaconf,
            },
        ):
            formatter = get_make_description_from_json("model.yaml")

        self.assertIs(formatter, Dataset.make_description_from_json)
        self.assertEqual(targets, ["pkg.TrainDataset"])

    def test_static_method_reference_is_loaded_and_adapted(self):
        """The file/class/static-method syntax supports legacy three-argument code."""

        with tempfile.TemporaryDirectory() as temp_dir:
            formatter_path = Path(temp_dir) / "dataset_file.py"
            formatter_path.write_text(
                """
class DatasetDescriptions:
    @staticmethod
    def make_description_from_json(meta, description_id, meta_path):
        return f\"{type(meta).__name__}:{meta['cell_type']}:{description_id}:{meta_path}\"
""".lstrip(),
                encoding="utf-8",
            )
            reference = (
                f"{formatter_path}::DatasetDescriptions::make_description_from_json"
            )

            Condition.set_description_formatter(reference)

            self.assertEqual(
                Condition(
                    name="K562",
                    description=UserDict({"cell_type": "K562"}),
                ).text(),
                "dict:K562:None:None",
            )
            self.assertEqual(Condition.get_description_formatter_source(), reference)

    def test_string_reference_requires_a_static_method(self):
        """An ordinary instance method cannot accidentally consume metadata as self."""

        with tempfile.TemporaryDirectory() as temp_dir:
            formatter_path = Path(temp_dir) / "dataset_file.py"
            formatter_path.write_text(
                """
class DatasetDescriptions:
    def make_description_from_json(self, meta):
        return str(meta)
""".lstrip(),
                encoding="utf-8",
            )
            reference = (
                f"{formatter_path}::DatasetDescriptions::make_description_from_json"
            )

            with self.assertRaisesRegex(TypeError, "@staticmethod"):
                Condition.set_description_formatter(reference)

    def test_formatter_must_return_string(self):
        """Invalid formatter output fails before reaching the tokenizer."""

        formatter = resolve_description_formatter(lambda meta: {"not": "text"})

        with self.assertRaisesRegex(TypeError, "must return str"):
            formatter({"cell_type": "K562"})


if __name__ == "__main__":
    unittest.main()
