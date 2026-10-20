import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np


def environment():
    versions = {}
    for name in ("coremltools", "numpy", "torch", "tokenizers", "mlx", "laya-mlx"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chip": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip(),
        "memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)),
        "os": platform.platform(),
        "python": platform.python_version(),
        "packages": versions,
        "source_sha256": source_hash(),
    }


def source_hash():
    h = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for folder in ("laya_coreml", "benchmarks"):
        for path in sorted((root / folder).glob("*.py")):
            h.update(str(path.relative_to(root)).encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def softmax(x):
    x = np.asarray(x, np.float64)
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def calibrated(agent, z, item):
    from laya_coreml.common import temp_bucket

    k, qt = len(item["markers"]), item["qtype"]
    t = agent.temperature_by_options.get(temp_bucket(qt, k), agent.temperature[qt])
    return softmax(np.asarray(z[:k], np.float64) / max(1e-3, float(t)))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def stats(values):
    return {
        "samples": len(values),
        "mean_ms": float(np.mean(values)),
        **{f"p{p}_ms": float(np.percentile(values, p)) for p in (50, 95, 99)},
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
    }


def compute_plan(agent):
    from collections import Counter, defaultdict

    import coremltools as ct

    from laya_coreml.agent import COMPUTE_UNITS

    plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        agent.model.get_compiled_model_path(),
        compute_units=getattr(ct.ComputeUnit, COMPUTE_UNITS[agent.compute_units]),
    )
    preferred, supported = Counter(), Counter()
    costs = defaultdict(float)
    operations = []

    def visit(block):
        for op in block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            cost = plan.get_estimated_cost_for_mlprogram_operation(op)
            device = type(usage.preferred_compute_device).__name__ if usage else "unknown"
            devices = [type(d).__name__ for d in usage.supported_compute_devices] if usage else []
            preferred[device] += 1
            supported.update(devices)
            if cost:
                costs[device] += cost.weight
            operations.append(
                {
                    "operator": op.operator_name,
                    "preferred": device,
                    "supported": devices,
                    "estimated_cost_weight": cost.weight if cost else None,
                }
            )
            for nested in op.blocks:
                visit(nested)

    for function in plan.model_structure.program.functions.values():
        visit(function.block)
    return {
        "kind": "Core ML anticipated execution plan, not a runtime hardware trace",
        "preferred_operation_counts": dict(preferred),
        "supported_operation_counts": dict(supported),
        "estimated_cost_by_device": dict(costs),
        "operations": operations,
    }


def public_differences(expected, actual, tolerance, path=""):
    """Compare complete public results, allowing only bounded float drift."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences = [] if expected.keys() == actual.keys() else [path + ": keys differ"]
        for key in expected.keys() & actual.keys():
            differences += public_differences(
                expected[key], actual[key], tolerance, path + "/" + key
            )
        return differences
    if isinstance(expected, list) and isinstance(actual, list):
        differences = [] if len(expected) == len(actual) else [path + ": lengths differ"]
        for i, (left, right) in enumerate(zip(expected, actual)):
            differences += public_differences(left, right, tolerance, path + "/" + str(i))
        return differences
    if isinstance(expected, float) and type(actual) in (float, int):
        if not math.isfinite(expected) or not math.isfinite(actual):
            return [path + ": non-finite public value"]
        # Public outputs are serialized decimal values; keep the stated tolerance
        # exact at a rounding boundary. Raw tensor error checks are unchanged.
        error = abs(Decimal(str(expected)) - Decimal(str(actual)))
        return (
            []
            if error <= Decimal(str(tolerance))
            else [f"{path}: float drift {error} > {tolerance!r}; {expected!r} != {actual!r}"]
        )
    return (
        [] if type(expected) is type(actual) and expected == actual else [path + ": values differ"]
    )
