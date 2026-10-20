"""Compose fixed-batch exports of identical weights into one adaptive native bundle."""

import json
import shutil
from pathlib import Path

from .artifacts import file_digest, verify_files


def _inventory(root):
    return {
        str(path.relative_to(root)): {"bytes": path.stat().st_size, "sha256": file_digest(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "coreml_config.json"
    }


def assemble_batch_bundle(directories, output):
    """Copy matching fixed-B exports, retaining the smallest as the base graph.

    Only physical batch capacity may differ. No weights are merged or converted,
    and no calibration, tokenizer, context limit or option layout is changed.
    The output is local and is never uploaded by this operation.
    """
    directories = [Path(path) for path in directories]
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if len(directories) < 2:
        raise ValueError("Supply at least two different batch capacities")
    entries = []
    for directory in directories:
        manifest = json.loads((directory / "coreml_config.json").read_text())
        if manifest.get("format") != "laya-coreml" or manifest.get("format_version") != 1:
            raise ValueError("Only ordinary Core ML format version 1 exports can be combined")
        if manifest.get("batch_variants"):
            raise ValueError("Supply individual exports, not an already combined bundle")
        verify_files(directory, manifest["files"])
        entries.append((manifest["shape"]["batch_size"], directory, manifest))
    entries.sort(key=lambda entry: entry[0])
    if len({entry[0] for entry in entries}) != len(entries):
        raise ValueError("Batch capacities must be distinct")
    _, primary_dir, primary = entries[0]
    signature = _signature(primary_dir, primary)
    if any(_signature(directory, manifest) != signature for _, directory, manifest in entries):
        raise ValueError(
            "Exports must have identical weights, tokenizer, config, precision and shapes except batch_size"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.mkdir()
        for name in ("model.mlpackage", "tokenizer", "encoder"):
            shutil.copytree(primary_dir / name, output / name)
        shutil.copy2(primary_dir / "rl_agent_config.json", output / "rl_agent_config.json")
        variants = []
        for capacity, directory, manifest in entries[1:]:
            relative = f"batches/b{capacity}/model.mlpackage"
            shutil.copytree(directory / "model.mlpackage", output / relative)
            variants.append({"package": relative, "shape": manifest["shape"]})
        # Publication metadata and previous validation attest to the old artifact,
        # not to this newly composed bundle.
        for key in ("package_sha256", "repository", "source_revision"):
            primary.pop(key, None)
        primary["batch_variants"] = variants
        primary["files"] = _inventory(output)
        (output / "coreml_config.json").write_text(json.dumps(primary, indent=2) + "\n")
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return output


def _signature(directory, manifest):
    host_files = [directory / "rl_agent_config.json", directory / "encoder/config.json"]
    host_files += sorted((directory / "tokenizer").rglob("*"))
    return {
        "weights": manifest["source_weights_sha256"],
        "precision": manifest["precision"],
        "layout": manifest.get("option_layout", "sequential"),
        "attention": manifest["attention"],
        "mask": manifest.get("attention_mask_construction"),
        "shape": {k: v for k, v in manifest["shape"].items() if k != "batch_size"},
        "host": {str(p.relative_to(directory)): file_digest(p) for p in host_files if p.is_file()},
    }


def variant_shapes(manifest, directory):
    """Validate declared native variants before loading any extra Core ML graph."""
    directory = Path(directory).resolve()
    base = manifest["shape"]
    entries = {}
    for variant in manifest.get("batch_variants", []):
        shape = variant["shape"]
        capacity = shape["batch_size"]
        if (
            type(capacity) is not int
            or not base["batch_size"] < capacity <= 64
            or capacity in entries
        ):
            raise ValueError("Invalid or duplicate native batch capacity")
        if {k: v for k, v in shape.items() if k != "batch_size"} != {
            k: v for k, v in base.items() if k != "batch_size"
        }:
            raise ValueError("Batch variant changes a non-batch shape limit")
        package = (directory / variant["package"]).resolve()
        if not package.is_relative_to(directory) or not package.is_dir():
            raise ValueError("Batch variant package must be inside the bundle")
        if package.suffix != ".mlpackage":
            raise ValueError("Batch variant must reference an mlpackage")
        entries[capacity] = (shape, package)
    return entries
