"""Prepare validated 0.5.0 adaptive bundles; upload only with explicit --upload."""

import argparse
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi

from laya_coreml.artifacts import verify_files
from laya_coreml.convert import REVISIONS

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("laya", "laya-multilingual", "laya-typed-decisions")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--account", default="aac6fef")
    args = parser.parse_args()
    receipts = []
    api = HfApi() if args.upload else None
    if api is not None:
        assert api.whoami()["name"] == args.account
    for name in MODELS:
        for short, precision in (("fp16", "float16"), ("fp32", "float32")):
            source = ROOT / f"models/v0412/{name}-{short}-dense-adaptive"
            manifest = json.loads((source / "coreml_config.json").read_text())
            evidence = ROOT / f"benchmarks/results/v0412-{name}-{precision}-cached-validation.json"
            validation = json.loads(evidence.read_text())
            assert validation["passed"] and validation["stability"]["calls"] == 100
            assert validation["export"]["files"] == manifest["files"]
            assert manifest["precision"] == precision
            verify_files(source, manifest["files"])
            repo = f"{args.account}/{name}-coreml-adaptive-{short}"
            output = ROOT / "models/hub-v0.5.0" / repo.split("/")[-1]
            if not output.exists():
                output.mkdir(parents=True)
                for relative in manifest["files"]:
                    target = output / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.link(source / relative, target)
            verify_files(output, manifest["files"])
            manifest.update(
                source=f"convaiinnovations/{name}",
                source_revision=REVISIONS[name],
                runtime_min_version="0.5.0",
            )
            (output / "coreml_config.json").write_text(json.dumps(manifest, indent=2) + "\n")
            shutil.copyfile(evidence, output / "validation.json")
            for filename in ("LICENSE", "NOTICE"):
                shutil.copyfile(ROOT / filename, output / filename)
            card = f'''---
license: apache-2.0
library_name: coreml
pipeline_tag: text-classification
base_model: convaiinnovations/{name}
tags:
- coreml
- laya
- apple-silicon
- adaptive-batch
---

# {name} Core ML adaptive {short.upper()}

Native Core ML conversion of [convaiinnovations/{name}](https://huggingface.co/convaiinnovations/{name}),
pinned source revision `{REVISIONS[name]}`. Requires **laya-coreml >= 0.5.0**,
Apple silicon and macOS 15+. Tested on M3 Max. This is an independent conversion.

```python
import laya_coreml as laya
with laya.load("{repo}") as agent:
    result = agent.predict("The customer requests a refund.", {{
        "refund": {{"type": "noul", "instructions": "Does the customer request a refund?"}}
    }})
```

Install with `pip install laya-coreml==0.5.0`. Pin this repository's commit SHA
with `revision=` for reproducibility. Inference requires no PyTorch, MLX or Transformers.

The bundle contains B=1/2/4/16 graphs, dense short sequence buckets,
{manifest["shape"]["max_length"]}-token capacity and up to 32 options per graph.
Logical batch/window behavior follows upstream Laya v0.4.12. Compute precision is
**{precision}**, running on CPU/GPU; this is not the fixed-shape ANE model.

Validation: 63/63 selected answers match upstream FP32, maximum calibrated
probability error {validation["probability_max_abs_error"]:.9g}, 100 stable repeated
calls, and full public-output checks. See `validation.json`. The model files are
byte-identical to the validated dense export; packaging changes provenance metadata
and adds this card/license. These finite checks do not prove universal model accuracy.

Adaptive graphs load lazily. The runtime's default cache retains at most four
extra execution instances, never answers. This improves repeated-shape latency
at a substantial resident-memory cost; `shape_cache_size=0` disables extra instances.
Cold/new shapes can be slower. FP16 is not numerically identical to FP32.
The matching FP32 benchmark recorded Core ML process peak RSS up to 13.51 GiB;
that is not a full unified/device memory measure or an FP16 memory estimate.

[Runtime and qualification evidence](https://github.com/mizorewww/laya-coreml/blob/main/docs/QUALIFICATION_0_4_12.md).
Original single-graph Hub repositories remain unchanged.
'''
            (output / "README.md").write_text(card)
            record = {"repository": repo, "precision": precision, "local_path": str(output)}
            if api is not None:
                api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
                commit = api.upload_folder(
                    repo_id=repo,
                    repo_type="model",
                    folder_path=output,
                    commit_message="Publish validated laya-coreml 0.5.0 adaptive bundle",
                )
                record["revision"] = commit.oid
                record["commit_url"] = commit.commit_url
            receipts.append(record)
            receipt_path = ROOT / "benchmarks/results/hub-v050-release.json"
            receipt_path.write_text(json.dumps(receipts, indent=2) + "\n")
            print("Uploaded" if api is not None else "Prepared", repo, flush=True)


if __name__ == "__main__":
    main()
