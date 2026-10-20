# 0.5.0 — upstream v0.4.12 migration

This release aligns applicable host APIs with NandhaKishorM/laya
`ffc142012af63d015fa19f05f29f9c5ff329a2cc`: batch and long-text inference,
structured decisions, hooks, calibration, evaluation, Router, HTTP/MCP,
and framework/tool integrations. Native execution and explicit unsupported
backend boundaries are preserved.

Compatibility changes include upstream Router defaults, validation and output
semantics. The LlamaIndex adapter supports the actual current BaseSelector and
SingleSelection contracts. See MIGRATION_0_4.md for behavior and dependency details.

The 341-entry API audit and three-checkpoint FP32/FP16 numerical/stability campaign
passed. The qualification report records unchanged applicable upstream assertions,
real SDK HTTP checks, real optional-framework checks and clean-wheel inference.
Package 0.5.0 changes version metadata relative to the recorded 0.4.0 development
qualification builds; release gates verify the final distribution separately.

Performance comparisons use upstream PyTorch 2.14, matched FP32 and fixed inputs.
MLX wins all 18 P50 comparisons in the final matrix. Core ML's alternating-order
confirmation wins all 18 P50/P95/mean-throughput comparisons; the initial four
sequential regressions are retained in the report. Core ML adaptive graph and
shape caches trade more resident memory for lower warm-call latency. Cold calls,
other hardware and arbitrary workloads are outside these finite guarantees.

See QUALIFICATION_0_4_12.md and UPSTREAM_0412_PROGRESS.json for raw evidence and
limitations. Original historical benchmark and package receipts remain unchanged.

Core ML adds verified B=1/2/4/16 adaptive bundle assembly, dense short sequence
buckets, lazy graph loading and a bounded aged-frequency shape cache. Existing
single-graph and ANE artifacts remain supported with their own capacities.
