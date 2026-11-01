# CED predictive MoE prefetch — as-is snapshot

**Status: incomplete, unverified, not ported, not compiled.** Nothing in this directory is
referenced by any CMake target and nothing on `main` changed. This is a frozen copy of a subsystem
that lives in a different tree, published so it can be read and discussed. No performance number is
claimed anywhere.

Tracked as issue #24 in this repository.

## What CED is

Conditional / Clustered Expert Distribution. A low-rank projection of an early layer's hidden state
estimates the expert routing ΔL = 4…8 layers ahead, so the DMA of an offloaded expert overlaps the
intervening compute instead of stalling it.

```text
h_t (layer l)
  stage 1:  z = V · h_t          V: r × d,   r = 32, d = 2560     ~83% of the FLOPs
  stage 2:  Λ = U · z + b        U: E × r,   E = 512
  → P̂^{l+ΔL}(e | h_t)
  filter:   P(e) > tau (0.15), cluster top-k (4), top_k candidates (2), 8 MiB per window
  issue:    async DMA on the mover transfer stream  →  expert in VRAM before layer l + ΔL
```

Lease safety: prefetched units only take a free slab slot or one whose hold-down lease expired
(30 dispatches); otherwise the prefetcher throttles (`prefetch_skipped_leased`) rather than thrash.
Prefetched units receive a hold-down lease on registration.

## Files

```text
contrib/ced/
├── README.md                                              # this file
└── as-is/
    ├── core/place/prefetch.h                              # CedConfig, CedStats, CedProjector, CedPrefetcher
    ├── core/place/prefetch.cpp                            # projector + prefetcher implementation
    ├── core/place/ced_gemv.h                              # projection GEMV ABI
    ├── core/place/ced_gemv.hip                            # RDNA4 Wave32 kernel, two-stage projection
    └── docs/ced_projection_and_prefetch_optimization_audit.md   # the tree's own audit
```

Copied byte for byte; `sha256`:

| file | sha256 |
| --- | --- |
| `core/place/prefetch.h` | `e22a5c23ba29e85d4d33f15d48d159f5cd4f81afd3bf7fec3def93f0320d57dd` |
| `core/place/prefetch.cpp` | `23ec68598b4920f5ed7fbfc55d184085fcd7938cb4bc1854138e2de5dace630f` |
| `core/place/ced_gemv.h` | `0a7cac22a3eb62e66edd12c756239c897e438414a925a8773b661484295e82a8` |
| `core/place/ced_gemv.hip` | `68841c34ef161599aa34419d0d99bc5c7f963796ea87b9e282533c36fffb6cbc` |
| `docs/ced_projection_and_prefetch_optimization_audit.md` | `5a18a2fb3fcfe00bde0bb388dd889e5d82ad1199d881aa128c1a3b45c1677a50` |

Provenance: the `RadianceFullDrafter` tree at commit `51248ae`, where CED was introduced in
`1a4455d`. That tree and this one share no history (no merge base), and their `core/place/` files
have diverged — `heat.{h,cpp}`, `mover.*` and `planner.*` each differ by 400–1300 diff lines. This is
a snapshot for reading, not a mergeable patch.

## Wiring in the source tree (not applied here)

The subsystem is wired in six places in the tree it came from. Shown so the integration is visible
without touching this branch's tree:

`core/rad_core.h` — config fields. Note the default is **on**.

```cpp
    bool    ced_prefetch = true;           /* --ced-prefetch / --no-ced-prefetch */
    int     ced_lookahead = 4;             /* --ced-lookahead ΔL (default 4 layers ahead) */
    float   ced_threshold = 0.15f;         /* --ced-threshold probability threshold tau */
    int     ced_top_k = 2;                 /* --ced-topk candidates per lookahead */
    bool    ced_reuse_ridgefill = false;   /* --ced-reuse-ridgefill */
```

`core/config.cpp` — flag parsing. No help entry exists for any of these:

```cpp
        else if (S("--ced-prefetch"))             c->ced_prefetch = true;
        else if (S("--no-ced-prefetch"))          c->ced_prefetch = false;
        else if (S("--ced-lookahead"))            { if (!(v = need(i))) return RAD_E_INVAL; c->ced_lookahead = atoi(v); }
        else if (S("--ced-threshold"))            { if (!(v = need(i))) return RAD_E_INVAL; c->ced_threshold = (float)atof(v); }
        else if (S("--ced-topk"))                 { if (!(v = need(i))) return RAD_E_INVAL; c->ced_top_k = atoi(v); }
        else if (S("--ced-reuse-ridgefill"))      c->ced_reuse_ridgefill = true;
```

`core/engine_bringup.cpp` — construction, once per rank:

```cpp
        CedConfig ced_cfg;
        ced_cfg.enabled = cfg_.ced_prefetch;
        ced_cfg.lookahead = cfg_.ced_lookahead;
        ced_cfg.threshold = cfg_.ced_threshold;
        ced_cfg.top_k = cfg_.ced_top_k;
        ced_cfg.reuse_ridgefill = cfg_.ced_reuse_ridgefill;
        RAD_TRY(r->prefetcher.init(r->program, r->plan, &r->mover, &r->heat, ced_cfg));
```

`core/engine.cpp` — the only call site, at step boundaries (single-sequence batches only):

```cpp
            r.heat.observe(hr.layer, hr.h, n_ex, rep_wt);
            if (r.prefetcher.enabled() && b->n_seq <= 1) {
                r.prefetcher.on_step_routing(hr.layer, hr.h, n_ex);
            }
```

plus a report line in the shutdown path: `if (r->prefetcher.enabled()) RAD_INFO("%s", r->prefetcher.report().c_str());`

`core/CMakeLists.txt` — the kernel, and only the kernel:

```cmake
  target_sources(rad_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/place/ced_gemv.hip)
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/place/ced_gemv.hip PROPERTIES LANGUAGE HIP)
```

`prefetch.cpp` itself is picked up by that tree's directory glob, not listed.

## What `main` is missing for this to build

- `core/runtime/rad_ridgefill_wrapper.h` — included by `prefetch.h`, absent here.
- `#include <arch/rad_rdna4_isa.h>` — in the source tree the header is `arch/common/rad_rdna4_isa.h`.
  The include name and the path disagree; it is currently a dead include either way.
- The `Heat` lease API. `prefetch.cpp` calls `is_leased`, `lease_remaining`, `promoted_at`,
  `find_eviction_candidate`, `min_residency_dispatches`, `advance_dispatch`, `active_units`,
  `boost_unit` and `set_burst_moves`; this repository's `Heat` has none of them.
- The `Mover` / `Plan` interfaces as this tree defines them (400–1300 diff lines each).
- The `Config` fields and flags above.

## Seven known defects

From the tree's own audit (`as-is/docs/ced_projection_and_prefetch_optimization_audit.md`), restated
in issue #24, none re-verified against current code:

1. `CedProjector::project_device` passes host `std::vector` pointers to the GPU kernel — VM page
   fault on first use.
2. The GEMV kernel computes stage 1 on wave 0 only; the other three wavefronts idle through 83% of
   the work, and `V` is read with a 10 240-byte lane stride (12.5% cache-line efficiency).
3. The documented temporal cache hints (`NT_RT`, `LU`) are not in the compiled loads.
4. The lease counter `stats_.dispatches` advances once per *layer*, so a lease of 2 dispatches
   expires after 2 layers and an intra-step eviction can reclaim a just-prefetched unit.
5. `CedPrefetcher::on_layer_dispatch` is never called from `core/engine.cpp`; only the step-boundary
   hook `on_step_routing` is. Intra-pass projection is dead code.
6. Projection weights are synthetic `sin`/`cos` values; `set_weights` and `set_ridgefill` are never
   called at bringup.
7. The tie-break on `expert_id` makes the prefetcher always pick the first experts of a cluster.

## A premise that is wrong

`prefetch.h` sizes its DMA window against "PCIe Gen5 x16, about 50–55 GB/s". The measured hardware is
PCIe 4.0 x16 (2.5 MiB in 95–100 µs). The latency budget against 10–18 µs of layer compute has to be
redone before any claim about hiding the transfer holds.

## Acceptance for a future attempt

- Real projection weights, produced and checked against the router: hit rate of the predicted set
  against the actual top-k per layer.
- Lossless by construction — prefetch changes placement timing, never which expert is computed — and
  a statement of where parity stays bit-exact.
- A card measurement with the prefetcher on and off, same build, evaluated against the PCIe 4.0
  transfer budget.

## Not claimed here

No measured speedup, no correctness result, no review of the code beyond the two documents copied
into `as-is/`. Separately, this repository's `main` already ships a different, working expert
prefetch for `--placement expert_tiered|auto`; that mechanism is unrelated to CED.
