# Historical candidate levers — inactive review material

This package preserves the twelve Opt-146-era proposals and the KV/QSA research report requested for code review. It does **not** apply their patches to the current engine, change defaults, or endorse the historical speed/occupancy claims. The original implementation line used ABI 16; current Radiance uses ABI 26.

Original README files are retained as historical documents. Their words such as "verified", gain percentages, hardware equations and recommended defaults are **unverified source claims**, not validation of this PR. Several are contradicted by later source-line experiments. No production model, benchmark baseline, binary plugin or training artifact is included.

## Coverage

| Lever | Proposal | Current review status |
|---|---|---|
| 1 | GDN/QSA projection fusion | Original sources and patch retained; not ported to ABI 26. |
| 2 | Fused MoE router | Original readable sources retained; the target already has related routing code. No speed claim. |
| 3 | KVarN / QSA offload | Incomplete historical proposal; its reduced-precision KV path must not be treated as an allowed lossless optimization. No active integration. |
| 4 | Reduced-vocabulary draft head | Requires an external checkpoint not included here; no executable serving claim. |
| 5 | Adaptive bandit controller | Historical proposal only. A different, untested implemented controller is preserved in the separate FullDrafter source-review package. |
| 6 | Fast store drain | Related implementation already exists in current Radiance; no duplicate active patch. |
| 7 | Swizzled prefill GEMM arms | Original proposal retained; target-specific support and performance untested. |
| 8 | Stager wrap prefetch | Historical proposal, not a replacement for the target's newer generalized stager. |
| 9 | RLFU VRAM cache | Original patch is structurally malformed and is preserved with an `.invalid.txt` suffix. |
| 10 | MoE decode grid clamp | Original patch is structurally malformed and is preserved with an `.invalid.txt` suffix. |
| 11 | HC write pipelining | Original proposal retained; a separately reviewed direct target port is submitted independently. |
| 12 | CPU-wait/pipelining change | Original patch is structurally malformed; relevant target state-clearing/ahead behavior already exists. |
| Research | KV-cache / QSA report | Research context, not an implemented optimization. |

`manifest.json` records original paths, published paths, byte hashes and the explicit renames of malformed patches. The original catalog and forensic audit are included under `originals/`, not promoted as a current performance report.

## Safety and validation

- Nothing in this package is added to the engine's build or selected by its kernel loader.
- Do not apply these ABI-16 patches to current `main`.
- Invalid patches are retained as text for inspection, not disguised as apply-ready changes.
- Package validation checks inventory, byte preservation, patch parsing classifications and absence of NUL-corrupt payloads. It is not a build, numerical oracle, benchmark result or gate PASS.
- The 77-NUL-byte sorted-down candidate config is not an implementation and is not published as JSON; its omission is accounted for in the companion FullDrafter review manifest.
