# Conversion notes

This page describes the ordinary Core ML export. The separately rewritten ANE
graph and optional weight palettization are documented in
[ANE_ENGINEERING.md](ANE_ENGINEERING.md).

The export loads **original Laya checkpoints** into FP32 PyTorch modules, strictly
checks all state-dict keys, traces an inference-only implementation, and saves a Core ML ML Program.
The published checkpoint files themselves contain predominantly FP16 tensors;
FP32 here describes the export/reference computation, not higher-precision source weights.
No training, pruning or weight quantization is performed. FP16 is a conversion
precision choice; FP32 can be selected for diagnostics.

The runtime uses the checkpoint's tokenizer, prompt layout, option markers,
question-type embedding, decision head, action head, and calibration temperatures.
Choice, score, noul, structured criteria, token accounting, and zero generated
tokens follow the upstream API. The encoder is bidirectional: every question
still runs its own encoder sequence. There is no shared-state hidden-state cache.

## Validated conversion choices

- `coremltools==9.0`, `torch==2.7.0`, `numpy==2.1.3`, Python 3.12.
- TorchScript tracing with graph checking, evaluation mode, original weights loaded into FP32 modules.
- ML Program, macOS 15 / iOS 18 deployment target. Actual execution was tested
  on an M3 Max running macOS 27.2; iPhone/iPad and older macOS execution were not tested.
- Default sequence lengths are selected from 16, 32, 64, 96, 128, 192, 256, 384,
  512, 768, 1024, capped by the checkpoint's context limit. The runtime pads to
  the smallest available length and masks those added tokens.
- Default batch size is one, with 32 marker slots. More questions run in chunks.
  `--batch-size` and `--max-options` produce different exported signatures.
- Fixed shapes are available for a known workload. Inputs exceeding an export's
  length or option capacity raise an error; they are not silently truncated to
  fit a smaller export. Original checkpoint context truncation is preserved.

Apple documents [TorchScript conversion](https://apple.github.io/coremltools/docs-guides/source/convert-pytorch-workflow.html)
and [enumerated input shapes](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html).
Multiple enumerated inputs need the same number of shapes, matched by index;
this export pairs input IDs and attention masks accordingly.

## Failures retained for reproducibility

These are observations on this machine and OS, not claims about every Core ML version.

1. PyTorch's `__or__` boolean operator was not converted. Explicit
   `torch.logical_or` / `torch.logical_and` preserve the same mask semantics.
2. NumPy 2.5 rejected a deprecated array-to-scalar conversion inside coremltools
   9.0. The supported project dependency is pinned below NumPy 2.2. PyTorch was
   pinned to the converter's tested 2.7.0 version instead of 2.7.1.
3. `RangeDim` with forced `CPU_AND_GPU` produced large numerical errors and
   different results on repeated identical inputs. The original SDPA export
   matched only 47/63 reference answers, and explicit matmul/softmax attention
   matched 20/63. FP32 did not resolve the observed short-input GPU failure.
   CPU / automatic selection gave correct outputs for the SDPA graph.
4. Enumerated lengths restored GPU fidelity and repeatability. A separate tiny
   regression test then exposed an MPSGraph compiler `SIGTRAP` when slicing a
   constant boolean local-attention matrix. The diagnostic named
   `ElementsAttr::getValues<bool>` / `FoldStridedSliceOp`.
5. The final implementation slices **integer positions** and constructs the
   boolean local mask afterwards. This removes the compiler trap. It does not
   cure the general RangeDim GPU failure: the subsequent experiment still
   matched only 49/63 and was not repeatable. Enumerated lengths remain the default.

The runtime rejects RangeDim + `cpu_gpu` unless explicitly allowed for a
diagnostic experiment. To reproduce that failed configuration:

```bash
laya-coreml convert laya-multilingual models/range-experiment --shape-mode range
python -m benchmarks.validate models/range-experiment \
  --name laya-multilingual --compute-units cpu_gpu --allow-unvalidated-gpu \
  --repeats 10 --output artifacts/range-experiment.json
```

The test is expected to fail on the measured environment. Raw failed and
successful reports are retained in `benchmarks/results/`; reports containing
`"passed": false` must not be cited as validated configurations.

## Device evidence

`CPU_AND_NE` means CPU and Neural Engine are *allowed*, not that every operator
runs on the Neural Engine. The benchmark records the Core ML compute plan's
preferred/supported devices and estimated costs. This is an anticipated plan,
not an Instruments runtime hardware trace, power measurement, or proof of
exclusive Neural Engine execution.

## Loading Hub snapshots

The release smoke test found a separate packaging issue: loading a symbolic-link
weight file from the shared Hugging Face cache caused Core ML's native compiler
to report a missing `model.mlmodelc/weights/weight.bin`. All six equivalent local
bundles loaded successfully. The runtime now copies symlink-backed packages to
a content-addressed cache of regular files before constructing `MLModel`.
Hashes are checked before and after copying and on reuse; a changed or damaged
cache raises an error. Regular-file local bundles do not take this copy path.
See [USAGE.md](USAGE.md) for the cache location and override.

## Reproducibility

Every export includes `coreml_config.json`: original weight SHA256, source
revision, shapes, precision, attention implementation, tool versions, conversion
time and hashes of every package/tokenizer/config file. Exports refuse to
overwrite existing directories. A failed export removes only its newly created
output directory.

The committed golden reference was generated from unmodified upstream Laya
revision `573e5b62696ba441230cd6be71d593331b5d23af` using FP32 PyTorch MPS.
It includes full input token IDs and unrounded logits. Validation compares those
tokens exactly and checks selected answers, calibrated probabilities, action
probabilities, token accounting, and repeated public results.

To regenerate the golden reference in an upstream-compatible environment:

```bash
git clone https://github.com/NandhaKishorM/laya .upstream
git -C .upstream checkout 6a5819129eb220570792e417e49723d697efd76f
python -m benchmarks.reference --upstream .upstream --model-root /path/to/original/checkpoints
```

The exact reference dependencies are recorded in the generated JSON. They are
separate from the pinned export environment; Transformers is not a runtime or
export dependency of laya-coreml.

## Adaptive native batch bundles (unreleased v0.4.12 migration)

A fixed B=1 graph cannot execute several question rows together. Export multiple
physical batch sizes from exactly the same checkpoint, then combine them locally:

```bash
python -m laya_coreml convert SOURCE exports/b1 --batch-size 1
python -m laya_coreml convert SOURCE exports/b2 --batch-size 2
python -m laya_coreml convert SOURCE exports/b4 --batch-size 4
python -m laya_coreml convert SOURCE exports/b16 --batch-size 16
python -m laya_coreml batch-bundle exports/adaptive exports/b1 exports/b2 exports/b4 exports/b16
```

Load `exports/adaptive` with the ordinary `load()` or `Agent` API. The runtime
selects the smallest available graph that fits the remaining logical rows, using
the largest graph for full chunks. A short final chunk can use a smaller graph.
The logical `predict_batch(batch_size=...)` control still counts states exactly
as upstream does. Graph selection changes only native tensor execution.

The assembler checks source weight hashes, host config and tokenizer bytes,
precision, attention implementation and shape limits. Only batch capacity may
differ. It rejects duplicate capacities, incompatible exports and an existing
output directory. It copies model packages, produces a fresh file inventory and
does not carry old validation reports or publication metadata into the result.
It does not upload or publish the bundle.

Additional graphs load on first use and stay resident until the agent is released
or its context exits. Warm all expected request shapes before measuring steady
state. Storage and resident memory grow with the selected graph set; this is a
tradeoff, not weight compression. Single-graph bundles keep their existing path.

The English migration experiment uses B=1/2/4/16. B=4 alone wastes work for the
two-row windows produced by the default long-text scanner; including B=2 avoids
that regression without changing the window policy. Qualification of the full
three-checkpoint matrix and clean-wheel validation remains part of migration
acceptance; do not treat composition alone as a validated release artifact.


Ordinary Core ML agents now retain up to four additional execution instances,
keyed by graph and every input tensor shape. Each physical graph uses its already
loaded instance for its first/current shape; only additional shapes need extra
instances. This avoids repeated expensive
reshapes when requests alternate between sequence lengths. `shape_cache_size=0`
on `Agent` or `load` disables this cache; accepted capacities are 0–32. The
frequency-based cache limit applies across all batch graphs within an agent, not separately per graph.
Prediction and eviction are serialized within the cache, and leaving the agent's
context manager releases the cached instances. This improves repeated mixed-shape
latency at the cost of additional resident model memory; it does not cache answers,
reduce input lengths, or change numeric precision. Cold misses and workloads with
more shapes than the configured capacity can still incur model-loading/reshape
cost. Long-tail misses use the original graph until their aged usage frequency
exceeds a cached shape, preventing reload thrashing. Fixed-shape ANE agents do
not use this ordinary Core ML cache.

New enumerated exports include 48, 80 and 112 token shapes in addition to the
existing sequence-length choices. This reduces unnecessary padding for short
batches; existing bundles retain their recorded shape sets. Do not assemble old
and new shape sets into one adaptive bundle: export all capacities consistently.


For strict FP32 qualification, export every capacity with `--precision float32`.
The local qualification paths are `models/v0412/MODEL-fp32-bB-dense` and the
assembled `models/v0412/MODEL-fp32-dense-adaptive`, where MODEL is `laya`,
`laya-multilingual` or `laya-typed-decisions`, and B is 1, 2, 4 or 16. Use source
weights in `../laya-mlx/models/MODEL`, pinned by the benchmark manifests. Use
empty output directories: the converter and assembler refuse to overwrite them.
The same recipe with `--precision float16` and `fp16` in the path creates the
reduced-precision variant. Existing Hub bundles are not silently replaced.

From the sibling MLX repository, `python -m benchmarks.qualification` performs
full-reference and repeated-call validation before timing all three backends
sequentially. It checks checkpoint/input hashes and explicitly disables upstream
AMP for FP32 comparison. Its final output comparison uses 0.0001 for FP32 public
floats, exact non-float values, and 0.02 for reduced precision. Raw tensor checks
retain their separate original tolerances. No finite suite proves equivalence
for every possible request.

The conversion environment deliberately pins PyTorch 2.7; do not use it as the
upstream performance baseline. The final qualification driver invokes upstream
with the sibling MLX reference environment (PyTorch 2.14) and checks its version
against `reference-v0412-fp32-torch214.json`. Native Core ML inference still runs
in its own environment and requires no PyTorch dependency in an installed wheel.

With the two sibling repositories and their validated development environments,
regenerate the final golden from this Core ML repository with:

```bash
../laya-mlx/.venv/bin/python -m benchmarks.reference \
  --upstream .upstream --model-root ../laya-mlx/models \
  --output benchmarks/results/reference-v0412-fp32-torch214.json
```

Then run `.venv/bin/python -m benchmarks.qualification` from `../laya-mlx`.
For an alternating-order performance repeat, run
`.venv/bin/python -m benchmarks.paired_features --directory benchmarks/results/v0412-final-paired`
there after all other accelerator jobs finish.
