import json
import threading
from pathlib import Path

import numpy as np
import pytest

from laya_coreml.agent import Agent
from laya_coreml.batching import assemble_batch_bundle, variant_shapes


def export(root, capacity):
    root.mkdir()
    for name in ("model.mlpackage", "tokenizer", "encoder"):
        (root / name).mkdir()
        (root / name / "config.json").write_text("{}")
    (root / "rl_agent_config.json").write_text('{"max_len":128}')
    manifest = {
        "format": "laya-coreml",
        "format_version": 1,
        "source_weights_sha256": "same-weights",
        "precision": "float16",
        "attention": "sdpa",
        "shape": {
            "batch_size": capacity,
            "max_length": 128,
            "max_options": 8,
            "flexible": True,
            "lengths": [16, 32, 64, 128],
            "min_length": 16,
        },
        "files": {},
    }
    (root / "coreml_config.json").write_text(json.dumps(manifest))
    return root


def test_bundle_checks_identity_and_preserves_smallest_graph(tmp_path):
    small, large = export(tmp_path / "small", 1), export(tmp_path / "large", 4)
    # Previous reports must not be copied to a newly composed artifact.
    (small / "validation.json").write_text('{"passed":true}')
    output = assemble_batch_bundle([large, small], tmp_path / "bundle")
    manifest = json.loads((output / "coreml_config.json").read_text())
    assert manifest["shape"]["batch_size"] == 1
    assert list(variant_shapes(manifest, output)) == [4]
    assert not (output / "validation.json").exists()
    assert "batches/b4/model.mlpackage/config.json" in manifest["files"]
    with pytest.raises(FileExistsError):
        assemble_batch_bundle([small, large], output)
    (large / "rl_agent_config.json").write_text('{"max_len":64}')
    with pytest.raises(ValueError, match="identical"):
        assemble_batch_bundle([small, large], tmp_path / "wrong")
    assert not (tmp_path / "wrong").exists()


@pytest.mark.parametrize("change", ["capacity", "context", "outside"])
def test_variant_rejects_incompatible_graph(tmp_path, change):
    source = export(tmp_path / "source", 1)
    manifest = json.loads((source / "coreml_config.json").read_text())
    shape = {**manifest["shape"], "batch_size": 4}
    package = "model.mlpackage"
    if change == "capacity":
        shape["batch_size"] = True
    elif change == "context":
        shape["max_length"] = 64
    else:
        package = str(tmp_path)
    manifest["batch_variants"] = [{"shape": shape, "package": package}]
    with pytest.raises(ValueError):
        variant_shapes(manifest, source)


def test_native_graph_selection_preserves_rows_masks_and_padding():
    agent = Agent.__new__(Agent)
    agent.shape = {
        "batch_size": 1,
        "max_length": 32,
        "max_options": 4,
        "flexible": True,
        "min_length": 16,
        "lengths": [16, 32],
    }
    agent.tok = type("Tokenizer", (), {"pad_token_id": 0})()
    agent._batch_variants = {n: ({**agent.shape, "batch_size": n}, Path(str(n))) for n in (4, 8)}
    agent._batch_lock = threading.RLock()
    calls = []

    class Model:
        def predict(self, arrays):
            calls.append(arrays)
            # Depend on the actual masked content so padding mistakes are visible.
            sums = (arrays["input_ids"] * arrays["attention_mask"]).sum(1)
            logits = np.repeat(sums[:, None], 4, axis=1).astype(np.float32)
            return {"logits": logits, "action_logits": np.zeros((len(sums), 2), np.float32)}

    agent.model = Model()
    agent._batch_models = {4: Model(), 8: Model()}
    assert agent._select_graph(1)[0]["batch_size"] == 1
    assert agent._select_graph(2)[0]["batch_size"] == 4
    assert agent._select_graph(6)[0]["batch_size"] == 8
    assert agent._select_graph(12)[0]["batch_size"] == 8
    batch = {
        "input_ids": np.arange(10 * 17).reshape(10, 17),
        "attention_mask": np.ones((10, 17), np.int32),
        "marker_pos": np.tile([1, 2], (10, 1)),
        "marker_mask": np.ones((10, 2), np.int32),
        "qtype": np.zeros(10, np.int32),
    }
    batch["attention_mask"][0, 5:] = 0
    logits, actions = agent._forward(batch)
    expected = (batch["input_ids"] * batch["attention_mask"]).sum(1)
    np.testing.assert_array_equal(logits[:, 0], expected)
    np.testing.assert_array_equal(actions, np.full((10, 2), 0.5))
    assert [c["input_ids"].shape for c in calls] == [(8, 32), (4, 32)]
    assert calls[0]["attention_mask"][0].sum() == 5
    assert calls[1]["attention_mask"][2:].sum() == 2  # safe dummy rows
    agent.__exit__(None, None, None)
    assert agent.model is None and not agent._batch_models
