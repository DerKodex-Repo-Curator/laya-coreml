"""The scheduling-safe upstream fixture must still reject actual early unloads."""

import ast
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_official_test import measure_idle_window_on_worker


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("elapsed", [0.1, 0.199, 0.201, 2.0])
def test_worker_deadline_rejects_early_unload_and_allows_late_observer(fail, elapsed):
    source = Path(__file__).resolve().parents[1] / ".upstream/tests/test_serve.py"
    assert source.exists(), "pinned upstream checkout required"
    tree = measure_idle_window_on_worker(ast.parse(source.read_text()))
    fixture = next(
        n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "BlockingRouter"
    )
    clock = SimpleNamespace(now=10.0)

    class IdleRouter:
        def __init__(self):
            self.unloaded = threading.Event()

        def predict(self, *args, **kwargs):
            return {}

        def unload(self):
            self.unloaded.set()

    namespace = {
        "IdleRouter": IdleRouter,
        "threading": threading,
        "time": SimpleNamespace(monotonic=lambda: clock.now),
        "fail": fail,
    }
    module = ast.fix_missing_locations(ast.Module(body=[fixture], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    router = namespace["BlockingRouter"]()
    router.release.set()
    if fail:
        with pytest.raises(ValueError, match="test failure"):
            router.predict()
    else:
        router.predict()
    assert router.finished.is_set()
    clock.now += elapsed
    if elapsed < 0.2:
        with pytest.raises(AssertionError, match="unload preceded idle deadline"):
            router.unload()
        assert not router.unloaded.is_set()
    else:
        router.unload()
        assert router.unloaded.is_set()
