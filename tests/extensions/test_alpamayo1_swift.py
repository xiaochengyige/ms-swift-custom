from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MS_SWIFT_ROOT = REPO_ROOT / "ms-swift-3.12.0"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MS_SWIFT_ROOT) not in sys.path:
    sys.path.insert(0, str(MS_SWIFT_ROOT))

from extensions.alpamayo1_swift.runtime.data import (  # noqa: E402
    ALPAMAYO1_IMAGE_SCHEME,
    parse_chunk_spec,
    parse_image_uri,
    parse_subset_spec,
)


def _install_swift_stubs():
    swift_module = types.ModuleType("swift")
    llm_module = types.ModuleType("swift.llm")
    plugin_module = types.ModuleType("swift.plugin")

    class ModelMeta:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class DatasetMeta:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    llm_module.MODEL_MAPPING = {}
    llm_module.DATASET_MAPPING = {}

    def register_model(meta, exist_ok=False):
        del exist_ok
        llm_module.MODEL_MAPPING[meta.model_type] = meta

    def register_dataset(meta, exist_ok=False):
        del exist_ok
        llm_module.DATASET_MAPPING[meta.dataset_name] = meta

    llm_module.ModelMeta = ModelMeta
    llm_module.DatasetMeta = DatasetMeta
    llm_module.register_model = register_model
    llm_module.register_dataset = register_dataset
    plugin_module.optimizers_map = {}

    sys.modules["swift"] = swift_module
    sys.modules["swift.llm"] = llm_module
    sys.modules["swift.plugin"] = plugin_module
    return llm_module, plugin_module


def _install_plugin_runtime_stubs():
    modeling_module = types.ModuleType("extensions.alpamayo1_swift.runtime.modeling")
    data_module = types.ModuleType("extensions.alpamayo1_swift.runtime.data")

    modeling_module.ALPAMAYO1_STAGE1_MODEL_TYPE = "alpamayo1_stage1"
    modeling_module.ALPAMAYO1_STAGE2_MODEL_TYPE = "alpamayo1_stage2"
    modeling_module.get_model_tokenizer = lambda *args, **kwargs: (None, None)

    data_module.ALPAMAYO1_DATASET_NAME = "alpamayo1_pai"
    data_module.install_alpamayo1_image_loader = lambda: None
    data_module.load_alpamayo1_pai_dataset = lambda *args, **kwargs: []

    sys.modules["extensions.alpamayo1_swift.runtime.modeling"] = modeling_module
    sys.modules["extensions.alpamayo1_swift.runtime.data"] = data_module


class Alpamayo1DataTests(unittest.TestCase):

    def test_parse_chunk_spec(self):
        self.assertEqual(parse_chunk_spec("0-3"), [0, 1, 2])
        self.assertEqual(parse_chunk_spec("1, 3,5"), [1, 3, 5])
        self.assertEqual(parse_chunk_spec("[4, 6]"), [4, 6])
        self.assertIsNone(parse_chunk_spec(""))

    def test_parse_subset_spec(self):
        self.assertEqual(parse_subset_spec("train@0-99"), ("train", "0-99"))
        self.assertEqual(parse_subset_spec("val"), ("val", "99-100"))
        self.assertEqual(parse_subset_spec(None), ("train", "0-99"))

    def test_parse_image_uri(self):
        uri = (
            f"{ALPAMAYO1_IMAGE_SCHEME}"
            "eyJjYW1lcmFfZmVhdHVyZSI6ImNhbWVyYS9jYW1lcmFfZnJvbnRfd2lkZV8xMjBmb3YiLCJjbGlwX2lkIjoiY2xpcC0xIiwidGltZXN0YW1wX3VzIjoxMjN9"
        )
        payload = parse_image_uri(uri)
        self.assertEqual(payload["clip_id"], "clip-1")
        self.assertEqual(payload["timestamp_us"], 123)


class Alpamayo1RegisterTests(unittest.TestCase):

    def test_registers_models_dataset_and_optimizer(self):
        llm_module, plugin_module = _install_swift_stubs()
        _install_plugin_runtime_stubs()

        for module_name in (
            "extensions.alpamayo1_swift.register",
            "extensions.alpamayo1_swift.plugin",
        ):
            sys.modules.pop(module_name, None)

        register_module = importlib.import_module("extensions.alpamayo1_swift.register")
        plugin_impl = importlib.import_module("extensions.alpamayo1_swift.plugin")

        register_module.register_all()

        self.assertIn("alpamayo1_stage1", llm_module.MODEL_MAPPING)
        self.assertIn("alpamayo1_stage2", llm_module.MODEL_MAPPING)
        self.assertIn("alpamayo1_pai", llm_module.DATASET_MAPPING)
        self.assertIn("alpamayo1_stage1", plugin_module.optimizers_map)
        self.assertIs(plugin_module.optimizers_map["alpamayo1_stage1"], plugin_impl.create_alpamayo1_stage1_optimizer)


if __name__ == "__main__":
    unittest.main()
