# 0.5.1 — upstream v0.4.12 migration and compiled-instance stability

This release includes the migration described in RELEASE_0_5_0.md. The initial
0.5.0 tag failed the macOS release gate and was never published to PyPI.
Its tag remains unchanged for traceability; 0.5.1 is the corrected release.

The bounded shape cache now loads extra execution instances from the owning
model's already compiled artifact instead of repeatedly compiling its package.
The previous route triggered a fatal coremltools GIL-state error on macOS 26
when admitting additional shapes. Instance cleanup now precedes releasing the
base models that own the compiled directories. Cache size, admission policy,
input values, graph weights and host outputs are unchanged.

The macOS CI keeps uncaptured native-stage diagnostics and runs all original
conversion and inference assertions. No tests are skipped to publish this fix.
See the publication receipt and new real-model qualification results for final
validation status. Existing model repositories remain unchanged; new adaptive
FP16/FP32 bundles use separate repositories.

Final validation: 337 tests; 3 checkpoints × FP32/FP16 × 63 questions, 100 stable calls per combination. Paired FP32 performance improves P50 and mean throughput on 18/18 workloads and P95 on 16/18; the two tail regressions and memory cost remain documented in [qualification evidence](https://github.com/mizorewww/laya-coreml/blob/main/docs/QUALIFICATION_0_5_1.md).
