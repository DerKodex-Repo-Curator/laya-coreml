import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from laya_coreml.shape_cache import ShapeCache


@pytest.mark.parametrize("capacity", [-1, 33, True, 1.5, None])
def test_invalid_capacity(capacity):
    with pytest.raises(ValueError, match="shape_cache_size"):
        ShapeCache(capacity)


def test_frequency_admission_identity_values_and_cleanup(monkeypatch):
    made = []

    class Model:
        def __init__(self, path, compute_units):
            made.append((path, compute_units))

        def predict(self, arrays):
            return arrays["input_ids"].sum()

    monkeypatch.setattr("coremltools.models.CompiledMLModel", Model)
    model = SimpleNamespace(
        package_path="a",
        compute_unit="cpu_gpu",
        get_compiled_model_path=lambda: "a.mlmodelc",
        predict=lambda arrays: arrays["input_ids"].sum(),
    )
    cache = ShapeCache(2)

    def call(length, value=1):
        return cache.predict(model, {"input_ids": np.full((1, length), value)})

    assert call(16) == 16
    assert made == []  # first shape reuses the already loaded graph
    assert call(32) == 32
    assert call(16, 3) == 48  # cache execution instances, never model outputs
    assert made == [("a.mlmodelc", "cpu_gpu")]
    assert call(64) == 64
    assert len(made) == 2 and len(cache._models) == 2
    assert call(128) == 128  # a long-tail miss reshapes the base, without a clone
    assert call(128, 2) == 256
    assert len(made) == 2
    assert call(32) == 32
    model = SimpleNamespace(
        package_path="b",
        compute_unit="cpu_gpu",
        predict=lambda arrays: 1000 + arrays["input_ids"].sum(),
    )
    assert call(32) == 1032  # identical tensor shapes in another graph stay distinct
    assert len(made) == 2
    cache.clear()
    assert not cache._models and not cache._frequency and not cache._misses
    assert not cache._base_shapes


def test_concurrent_same_shape_loads_once(monkeypatch):
    counts = {"loads": 0, "active": 0, "maximum": 0}
    lock = threading.Lock()

    class Model:
        def __init__(self, *args, **kwargs):
            counts["loads"] += 1

        def predict(self, arrays):
            with lock:
                counts["active"] += 1
                counts["maximum"] = max(counts["maximum"], counts["active"])
            time.sleep(0.002)
            with lock:
                counts["active"] -= 1
            return arrays["input_ids"].sum()

    monkeypatch.setattr("coremltools.models.CompiledMLModel", Model)
    model = Model()
    model.package_path, model.compute_unit = "a", "cpu_gpu"
    cache = ShapeCache(1)
    with ThreadPoolExecutor(max_workers=4) as pool:
        result = list(
            pool.map(lambda n: cache.predict(model, {"input_ids": np.full((1, 16), n)}), range(8))
        )
    assert result == [16 * n for n in range(8)]
    assert counts == {"loads": 1, "active": 0, "maximum": 1}


def test_disabled_uses_original_instance():
    model = SimpleNamespace(predict=lambda batch: batch["x"] + 1)
    assert ShapeCache(0).predict(model, {"x": 3}) == 4


def test_long_tail_does_not_reload_every_request(monkeypatch):
    loads = []

    class Model:
        package_path, compute_unit = "a", "cpu_gpu"

        def get_compiled_model_path(self):
            return "a.mlmodelc"

        def __init__(self, *args, **kwargs):
            loads.append(True)

        def predict(self, arrays):
            return arrays["input_ids"].shape[1]

    monkeypatch.setattr("coremltools.models.CompiledMLModel", Model)
    base = Model()
    cache = ShapeCache(2)
    for _ in range(12):
        for length in (16, 32, 64, 128, 256, 512):
            assert cache.predict(base, {"input_ids": np.ones((1, length))}) == length
    assert len(cache._models) == 2
    assert len(loads) < 10  # 72 requests must not cause 72 clone loads
    for length in range(1, 100):
        cache.predict(base, {"input_ids": np.ones((1, length))})
    assert len(cache._misses) <= 16


def test_each_physical_graph_reuses_its_first_instance(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("First shape must not create an additional model")

    monkeypatch.setattr("coremltools.models.CompiledMLModel", unexpected)
    cache = ShapeCache(1)
    for graph in range(4):
        model = SimpleNamespace(
            package_path=str(graph),
            compute_unit="cpu_gpu",
            predict=lambda arrays: arrays["input_ids"].sum(),
        )
        assert cache.predict(model, {"input_ids": np.ones((1, 16))}) == 16
    assert not cache._models
    assert len(cache._base_shapes) == 4


def test_close_releases_compiled_instances_before_base_directories():
    from laya_coreml.runtime import RuntimeMixin

    base, variant = object(), object()
    agent = SimpleNamespace(model=base, _batch_models={2: variant})

    def clear():
        assert agent.model is base
        assert agent._batch_models == {2: variant}

    agent._shape_cache = SimpleNamespace(clear=clear)
    assert RuntimeMixin.__exit__(agent, None, None, None) is False
    assert agent.model is None and not agent._batch_models
