# Post-quantisation refinement of a ~4 bpw EXL3 (trellis/QTIP-style) checkpoint: what reduces KL to bf16, by how much, at what cost

Scope note on sourcing (read first): in this session arxiv.org, huggingface.co, openreview.net and every paper-mirror site (alphaxiv, themoonlight, liner, emergentmind, fugumt, papers.cool, etc.) were unreachable (DNS blocked). Primary facts below come from (a) GitHub README/doc/issue pages fetched directly and (b) web-search result snippets that quote the arXiv HTML or model cards. Where a number is only available through a search snippet of a third-party summary, it is labelled "third-party". Numbers that could not be verified at all are listed under Gaps rather than reported. Date of research: 2026-10-28.

Starting point assumed throughout: EXL3 = "a streamlined variant of QTIP from Cornell RelaxML" that "computes Hessians on the fly and uses a fused Viterbi kernel" — [exllamav3 README](https://github.com/turboderp-org/exllamav3). Model: Qwen3.8-27B (48 Gated DeltaNet + 16 full-attention layers, 26.9B params), target 4.0–4.5 bpw, metric = mean teacher-forced KL to bf16.

---

## Key Question 1 — Quantisation-aware distillation / fine-tuning after PTQ (PV-tuning, QuIP#/QTIP fine-tuning, EfficientQAT, LLM-QAT, ParetoQ, "fine-tuning under the grid", TesseraQ): what gains, which parameters, what cost for ~27B

### Takeaway
Every published trellis/VQ pipeline (QuIP#, QTIP, PV-tuning) includes a post-PTQ fine-tuning stage, but published gains are almost all reported at 2–3 bits and in perplexity, not KL; at 4 bits the published PTQ-only numbers are already within ~1–2% of fp16 perplexity so the measured headroom is small. The only directly relevant 4-bit KL datapoint found is a self-reported model card for Qwen3.8-27B (PARO + M2XFP4): one epoch of "fine-tuning under the grid" on 256 calibration samples cut mean KL vs bf16 from 0.0348 to 0.0211 (−39%). Block-wise training of all weights (EfficientQAT Block-AP) is reported to beat end-to-end training of only quantisation parameters, and the combination beats either alone.

### Cited Findings

**PV-Tuning (AQLM, NeurIPS 2024 oral; arXiv 2405.14852)**
- The AQLM README's PV-tuning example "enables updates to the quantization codes, the codebooks and scales, and the non-quantized parameters" (i.e. discrete codes are optimised, not just continuous parameters) — [AQLM README](https://github.com/Vahe1994/AQLM)
- Author-reported WikiText-2 PPL without → with PV-tuning: Llama-2-7B 1x16: 5.92 → 5.68; Llama-2-7B 2x8: 6.69 → 5.90; Llama-2-13B 1x16: 5.22 → 5.05; Llama-2-70B 1x16: 3.83 → 3.78 (all ~2-bit) — [AQLM README](https://github.com/Vahe1994/AQLM)
- Compute (quantisation + tuning, authors): 7B ≈ 1 day on one A100-80GB (≈14.5 h on two); 70B 10–14 days on one GPU, ≈3 days 18 h on 8×A100; `--offload_activations` reduces VRAM at ~10% slowdown — [AQLM README](https://github.com/Vahe1994/AQLM)
- NeurIPS listing: "first Pareto-optimal quantization for Llama-2 family models at 2 bits per parameter" — [NeurIPS 2024](https://neurips.cc/virtual/2024/oral/97970)
- PV-tuned 70B checkpoints: Llama-3-70B 1x16g8 at 21.9 GB vs 141 GB fp16 — [HF ISTA-DASLab card via search](https://huggingface.co/ISTA-DASLab/Meta-Llama-3.1-70B-Instruct-AQLM-PV-2Bit-1x16)

**QuIP# fine-tuning stage (arXiv 2402.04396)**
- Two steps: fine-tuning during quantisation (`quantize_finetune_llama.py`) and a second end-to-end step (`finetune_e2e_llama.py`) that "tunes the sign vectors (SU/SV), layernorms, and language model head". `--ft_epochs 0` reproduces no-FT results; `--ft_grad_ckpt` and `--ft_train_mode` (materialise the full dequantised matrix) control memory/speed — [quip-sharp README](https://github.com/Cornell-RelaxML/quip-sharp)
- Paper (via search snippet): Llama-2-70B "can be quantized without fine-tuning in under 10 GPU-hours, and with fine-tuning in about 100 GPU-hours"; the paper defines a QuIP#-NoFT variant; 2-bit 70B fits in <20 GB — [QuIP# arXiv via search](https://arxiv.org/pdf/2402.04396)

**QTIP fine-tuning (arXiv 2406.11235, NeurIPS 2024)**
- The QuIP# fine-tuning code "have been merged into `lib/algo/finetune.py`"; main script `quantize_llama/quantize_finetune_llama.py` — [QTIP README](https://github.com/Cornell-RelaxML/qtip)
- Paper (via search snippet): the trellis "codes themselves cannot be tuned, though they remain compatible with QuIP#-style fine-tuning"; FT hyper-parameters match QuIP# "except that QTIP skips matrix fusion, so it quantizes seven matrices per block rather than four" — [QTIP arXiv via search](https://arxiv.org/pdf/2406.11235)
- Author-reported no-FT Llama-2-7B 2-bit WikiText-2 PPL 6.89 (HYB). Third-party review: 4-bit Llama-2-7B 1MAD/3INST 5.17 vs QuIP# 5.19 vs AQLM 5.21; 70B 2-bit C4 5.48 vs QuIP# 5.71, AQLM 5.62 — [Liner review via search (third-party)](https://liner.com/review/qtip-quantization-with-trellises-and-incoherence-processing)

**EfficientQAT (arXiv 2407.11062, ACL 2025)**
- Two phases: Block-AP = "Block-wise training of all parameters" (lr 1e-5 for 3/4-bit, 4096 train samples), then E2E-QP = "end-to-end training of quantization parameters" (step sizes only; lr 1e-5 for 3/4-bit) — [EfficientQAT README](https://github.com/OpenGVLab/EfficientQAT)
- Author-reported 4-bit (w4g128) WikiText-2 PPL: Llama-2-7B 5.53 vs fp16 5.47; 13B 4.93 vs 4.88; 70B 3.39 vs 3.32; w3g128 70B 3.61 — [EfficientQAT README](https://github.com/OpenGVLab/EfficientQAT)
- Cost (authors): 2-bit Llama-2-70B "on a single A100-80GB GPU in just 41 hours"; comparison table lists 900 h for LLM-QAT, 300 h for QuIP#, 336 h for AQLM — [EfficientQAT arXiv via search](https://arxiv.org/html/2407.11062v2)
- Ablation (Llama-2-7B w2g64, averaged PPL / avg acc): no training 453.49 / 40.69; Block-AP only 8.53 / 58.99; E2E-QP only 9.33 / 55.71; both 7.68 / 60.14. Training only the weights inside Block-AP (not scales) gives 14.32 vs 8.53. Authors: "Block-AP outperforms E2E-QP, aligning with findings from BRECQ" — [EfficientQAT arXiv Table 4 via search](https://arxiv.org/pdf/2407.11062)

**ParetoQ (arXiv 2502.02631)**
- Unified 1–4-bit QAT framework starting from a full-precision model; README reports only ParetoQ numbers at 3/4 bits (e.g. MobileLLM-1B 4-bit 62.5 acc / 7.9 PPL vs BF16 62.1 / 8.0) and no PTQ baseline at 3–4 bits; authors claim "sub-4-bit quantization, including binary, ternary, 2-bit, and 3-bit, often outperform 4-bit" in their scaling-law sense — [ParetoQ README](https://github.com/facebookresearch/ParetoQ)

**TesseraQ (Yale)**
- "Block reconstruction-based PTQ" with progressive adaptive rounding (PAR) of rounding variables and optional dequantisation-scale tuning; calibration C4, 512 samples × 2048. LLaMA-2 W4A16 WikiText-2 PPL 5.56 / 4.96 / 3.40 (7B/13B/70B); W3A16 5.84 / 5.16 / 3.68; W2A16 8.05 / 6.55 / 5.26 — [TesseraQ README](https://github.com/Intelligent-Computing-Lab-Yale/TesseraQ)

**"Fine-tuning under the grid" on Qwen3.8-27B (M2XFP4 + ParoQuant rotations) — self-reported model card**
- hugypufy/Qwen3.8-27B-PARO-M2XFP4: "a radiance .rad container of Qwen3.8-27B quantised to M2XFP4 (arXiv 2601.19213): MXFP4 blocks plus a learned per-subgroup metadata plane and learned Givens rotations, with the weights fine-tuned under the quantization grid" (4.25-bit); "ParoQuant, 1 epoch, 256 calibration samples, donor rotations frozen"; KL 0.0211 mean / 0.0063 median vs bf16, 94.15% top-1 agreement over 32,367 full-vocabulary positions, GSM8K 94.0% (custom 200-question harness); 83 tok/s on one Radeon AI PRO R9700, single-GPU only — [HF card via search](https://huggingface.co/hugypufy/Qwen3.8-27B-PARO-M2XFP4)
- RTN baseline hugypufy/Qwen3.8-27B-M2XFP4: KL 0.0348 mean / 0.0121 median, 92.39% top-1, GSM8K 92.5%; card states the fine-tuned variant "improves on by 39 % KL" — [HF card via search](https://huggingface.co/hugypufy/Qwen3.8-27B-M2XFP4)
- M2XFP itself (ASPLOS '26, Weiming Hu et al., 2026-01-28): metadata-augmented microscaling, "reduces accuracy loss by 70.63% compared to MXFP4 and 37.30% compared to NVFP4, achieving near-FP16 accuracy at an effective 4.5-bit precision"; element-level encoding for activations, subgroup-level for weights; evaluated on LLaMA-2/3, OPT, Mistral, Falcon, DeepSeek-R1-Distill-Qwen — [M2XFP arXiv via search](https://arxiv.org/abs/2601.19213)

**Other grid-constrained fine-tuning**
- Quantized Evolution Strategies (arXiv 2602.03120): fine-tunes INT4/INT8 Qwen2.5-1.5B/3B directly in low-precision space (Countdown accuracy 3.50% → 16.00% for INT4 1.5B; third-party summary) — gradient-free, task reward, not a KL objective — [Cognizant blog via search](https://www.cognizant.com/us/en/ai-lab/blog/quantized-evolution-strategies-qes-llm-fine-tuning)
- INT2.1 (arXiv 2306.08162): trains LoRA on frozen quantised weights with a loss combining KL to the full-precision model and CE — [arXiv via search](https://arxiv.org/pdf/2306.08162)

### Inferences
- For EXL3 the trellis codes are the only thing that cannot be gradient-tuned (QTIP: "codes themselves cannot be tuned"); the QuIP#/QTIP recipe therefore tunes the continuous parts (sign vectors of the RHT, per-tensor/group scales, norms, lm_head) with STE-free gradients. PV-tuning's discrete code updates would require a Viterbi re-encode step per update, which no published code does for trellis codes.
- Memory for a 27B student: full-weight Adam fine-tuning of a dequantised 27B model is ~16 bytes/param ≈ 430 GB and is not single-GPU; Block-AP style training (one ~0.4B-param block at a time, teacher activations cached) and QuIP#-style e2e tuning of only scales/signs/norms/head both fit on one 80 GB GPU (EfficientQAT does 70B 2-bit on one A100-80GB; QuIP# 70B FT ≈ 100 GPU-h). Estimate, not measured on this model.
- The 39% KL reduction on Qwen3.8-27B is the single most relevant datapoint but is self-reported, used a 256-sample calibration set, a different grid (MXFP4 subgroup metadata, 4.25 bpw), and a custom harness; it should be treated as an upper-bound indication that one epoch of grid-constrained fine-tuning is worth ~30–40% of 4-bit KL on this exact model, pending replication.
- Headroom at 4 bits is small in PPL terms (EfficientQAT 70B w4: 3.39 vs 3.32), so KL (which is more sensitive) is the right metric; none of the QAT papers report KL at 4 bits.

### Gaps
- QTIP paper's with-FT vs without-FT table at 3/4 bits could not be retrieved (arXiv blocked); only the 2-bit 7B no-FT number and third-party 4-bit numbers were found.
- PV-tuning has no published 4-bit results in the README; the paper's 4-bit numbers (if any) were not retrievable.
- LLM-QAT: only the EfficientQAT comparison figure (900 GPU-h) was found; no 4-bit KL numbers.
- No published pipeline fine-tunes an EXL3 model: the exllamav3 repo (README, `science/` folder = codebook_eval.py, gumbel_eval.py, gumbel_tail_eval.py, kv_quant_exp.py, qgemm_benchmark.py; `doc/` = convert.md, optimize.md, env_vars.md) and its issue tracker contain no fine-tuning/distillation feature or discussion — [science/](https://github.com/turboderp-org/exllamav3/tree/master/science), [doc/](https://github.com/turboderp-org/exllamav3/tree/master/doc), [issues search](https://github.com/turboderp-org/exllamav3/issues?q=finetune+OR+fine-tune+OR+distill+OR+training+OR+QAT)
- The hugypufy card's hardware/time for the one-epoch run was not visible in the snippet.

---

## Key Question 2 — Block-wise / layer-wise reconstruction (BRECQ, OmniQuant, AffineQuant, OBQ-style sequential compensation, "calibrated layer walk" / asymmetric calibration): recovery vs end-to-end, and handling of recurrent layers

### Takeaway
Calibrating each layer against the full-precision model's output while feeding it the quantised stream (GPTAQ/GPTQv2 "asymmetric calibration", and its 2026 successors REAL-Q and ReQuant) is the cheapest refinement with measured KL gains at W4A16: on Llama-3.1-8B W4A16 WikiText-2 KL falls from 0.0495 (GPTQ) → 0.0398 (GPTAQ) → 0.0367 (GuidedQuant) → 0.0336 (REAL-Q). Block reconstruction with all block weights trainable (BRECQ/EfficientQAT Block-AP) recovers more than end-to-end tuning of quantisation parameters alone, at single-GPU cost. No source handles recurrent/GDN layers specially inside a reconstruction loop; the one Qwen3.8-27B study found GDN layers unusually robust.

### Cited Findings
- BRECQ (ICLR 2021): block reconstruction "achieves a good balance between cross-layer dependency and generalization error"; net-wise reconstruction "suffers from bad generalization"; authors note "our analysis does not give the optimal configuration of the reconstruction granularity" — [BRECQ arXiv via search](https://arxiv.org/pdf/2102.05426v2)
- OmniQuant: learnable weight clipping (`--lwc`, lr 1e-2) and learnable equivalent transformation (`--let`, lr 5e-3), 128 calibration samples, example 20 epochs; README text gives no perplexity numbers (images only) — [OmniQuant README](https://github.com/OpenGVLab/OmniQuant)
- GPTAQ / GPTQv2 (ICML 2025, arXiv 2504.02692): "matches each quantized layer's output to the full-precision model's output" (asymmetric calibration) to "reduce the quantization error accumulated in previous layers"; "20 more lines of code than GPTQ"; integrated in GPTQModel via `v2=True` — [GPTAQ README](https://github.com/Intelligent-Computing-Lab-Yale/GPTAQ)
- GPTAQ paper (via search snippet): reduces GPTQ perplexity "by 20% to 90%"; Llama-3-70B 4-bit GPTQ 9.44 → 6.93; on rotated (SpinQuant) models the gain shrinks: 7.19 vs 7.26 at 4-bit — [GPTAQ arXiv via search](https://arxiv.org/pdf/2504.02692)
- REAL-Q (arXiv 2609.00049, submitted 30 Aug 2026): "applies Block-wise Gradient Descent after each 128-column block, using a single Adam step within the GPTQ sweep", with a sliding window to limit error propagation; KL is the primary metric. Llama-3.1-8B W4A16 WikiText-2 KL (×10⁻²): GPTQ 4.95, GPTAQ 3.98, GuidedQuant 3.67, REAL-Q 3.36; "reduces end-to-end KL divergence by up to ∼49%" (largest on Qwen3-1.7B W4A16); models LLaMA-3.1 8B/70B, Qwen3 0.6B–32B (70B in Appendix D.2) — [REAL-Q arXiv via search](https://arxiv.org/html/2609.00049v2)
- ReQuant (arXiv 2608.07019, 7 Aug 2026, Yongge Ma et al., PKU): "backpropagation-free post-processing step" that "revisits the integer weight codes without changing the grid, bit-width, or scales"; each accepted move strictly reduces the row loss; "per-sweep complexity comparable to a single GPTAQ pass"; evaluated on Llama-3 8B/70B and Qwen3-14B at W4A16/W4A4/W3A4/W2A4; Llama-3-70B W4A4 (QuaRot) RTN 60.14 → 6.00, AWQ 14.26 → 6.07 (FP 2.86); "a few entries favor the unrefined baseline, especially for GPTQ and GPTAQ"; refining RTN over several sweeps "can approach or surpass GPTAQ" — [ReQuant arXiv via search](https://arxiv.org/html/2608.07019); a third-party review calls the "consistent improvement" claim "slightly too strong" — [pith.science via search](https://pith.science/paper/2608.07019)
- EfficientQAT ablation: Block-AP (all block weights + scales, teacher-block targets) 8.53 avg PPL vs E2E-QP (scales only, end-to-end) 9.33; combination 7.68 (Llama-2-7B w2g64) — [EfficientQAT arXiv via search](https://arxiv.org/pdf/2407.11062)
- exllamav3 conversion calibrates on 250 rows × 2048 tokens by default (`-cr`, `-cc`), `-cd` substitutes a custom packed safetensors trace, `-rcp` applies a per-tensor bitrate recipe, `-hb` head bits default 6 — [convert.md](https://github.com/turboderp-org/exllamav3/blob/master/doc/convert.md)
- exllamav3 self-calibration pipeline (doc/optimize.md): `sc_measure.py` injects seeded noise per Linear into the unquantised model and records final-logit KLD; `--shaped` noise following the LDLQ error distribution correlates with ground-truth swap attribution at Spearman 0.84 vs 0.61 unshaped; isotropic noise overestimates sensitivity up to ~6× (v_proj worst); `sc_optimize.py` fits kld(t,K)=S_t·rfn_t(K)^α (fitted α≈1.7–1.9, real transitions ≈2.0, demotion exponent ≈2.35 measured on Qwen3.8-27B) and allocates integer bitrates greedily; full run for a ~27B model ≈2 h (measurement ≈35 min); calibration from ~140 seeded conversations in 19 languages sampled with the model's own template, reasoning on — [optimize.md](https://github.com/turboderp-org/exllamav3/blob/master/doc/optimize.md)
- Old exllamav3 optimizer could pessimise KL: GLM-4.6 uniform EXL3 KL vs FP16 = 0.326 (3 bpw), 0.156 (4 bpw), 0.113 (5), 0.082 (6), 0.068 (8); the optimizer's 4.16 bpw mix gave ≈0.18 (worse than uniform 4 bpw), a manual recipe with 6/8-bit for sensitive tensors gave ≈0.133; closed 2026-09-05 as "superseded by the self-calibration pipeline" — [exllamav3 issue #108](https://github.com/turboderp-org/exllamav3/issues/108)
- Hybrid/recurrent sensitivity: "A KL Lens on Quantization" (arXiv 2604.13440, 15 Apr 2026) proposes a "gradient-free sensitivity method that operates entirely via forward-pass signals" for SSM–Transformer hybrids, argues SQNR-based sensitivity "fails", and reports KL-guided mixed precision reaching "near-FP16 perplexity with model sizes and throughput competitive with Uniform INT4" — [arXiv via search](https://arxiv.org/html/2604.13440v1)
- Qwen3.8-27B GDN robustness: Maiboroda & Kozyrev (arXiv 2609.04098, 3 Sep 2026) quantised all 496 linear layers (GDN included) to NVFP4 W4A4; recurrent state error "stayed flat, around 12.3% at 32K tokens versus 13% at 256 tokens"; perplexity gap +0.73 at 4K context and +0.15 at 32K; 5-task average −0.52 vs BF16; GDN gate projections least sensitive (≈2.1% / 2.6% output change when fully quantised); noise on the write-strength (β) term "nearly harmless"; attributed to the delta rule overwriting state along the current key direction — [arXiv via search](https://arxiv.org/pdf/2609.04098), [HF card via search](https://huggingface.co/emee-ai/mnma_qwen3.8_27b_nvfp4)

### Inferences
- A "calibrated layer walk" is exactly GPTAQ's asymmetric objective; on an EXL3 pipeline it means re-running LDLQ/Viterbi for layer ℓ with Hessian H_q = Σ x̂ x̂ᵀ (quantised-stream inputs) and cross-term G = Σ x̂ xᵀ against the bf16 targets. The user's stored input Hessians are the bf16-stream H; the quantised-stream statistics must be recomputed per layer during the walk (one forward pass through the partially-quantised model over the calibration corpus).
- REAL-Q/ReQuant numbers show the ordering GPTQ > GPTAQ > GuidedQuant > REAL-Q in KL at W4A16, with the whole chain worth ~32% KL on Llama-3.1-8B; these are scalar-grid results, and the fraction that carries over to a trellis quantiser (which already has lower per-layer error) is unknown.
- The GDN study implies that for Qwen3.8-27B the long-context recurrent state does not amplify weight-quantisation noise, so block reconstruction can treat a GDN layer as an ordinary block (teacher inputs → block outputs) without special state handling, provided the chunked-recurrent forward is used for the block's calibration pass.

### Gaps
- No source compared block reconstruction vs end-to-end distillation on a trellis-coded model or reported KL for BRECQ/OmniQuant/AffineQuant at 4 bits on ≥27B dense models.
- REAL-Q and ReQuant runtime/VRAM and their 70B tables were not retrievable.
- Whether EXL3's `convert.py` already feeds quantised-stream activations to later layers (GPTAQ-style) could not be verified from the docs fetched.

---

## Key Question 3 — Output-side / Fisher-weighted Hessians (YAQA, GuidedQuant, Kronecker-factored curvature) vs plain input Hessians: KL gains, gradient-pass cost, scaling to ~27B

### Takeaway
YAQA replaces the input Hessian Σxxᵀ with a Kronecker-factored approximation of the full-model KL Hessian and, with the QTIP trellis quantiser and no fine-tuning, reduces KL to the original model by ≈1/3 across Llama/Gemma/Qwen at 2–4 bits (Llama-3.1-8B-Instruct 4-bit: 0.019 → 0.013–0.014, third-party readings). The price is a backward pass over ~134M tokens (Sketch B: 65,536 seqs × 2048) on a multi-GPU node; models under ~20B fit one 8×80 GB node, 70B needed two. GuidedQuant is the cheaper Fisher-diagonal variant (70B gradients in 2.7 h on 6–7 A100s) and is reported to be weaker than YAQA.

### Cited Findings
- YAQA README: "reduces the KL divergence to the original model by a factor of 1/3 over LDLQ/GPTQ"; "based on the QTIP codebase"; per-model results only as an image — [YAQA README](https://github.com/Cornell-RelaxML/yaqa)
- YAQA Hessian collection (`hessian_llama/`): Sketch A `--power_iters 6 --ctx_size 8192 --n_seqs 4096`; Sketch B (recommended "if you can afford it") `--power_iters 1 --ctx_size 2048 --n_seqs 65536`; "Models under ~20B fit on one 8x80G node. A 70B model was run across two 8x80G nodes" (layers 0–39 / 40–79); layers processed independently; per-sample Hessians in FP32, optional `--fp64_accum`; `--cpu_offload` discouraged unless faster than "recomputing gradients" — [yaqa/hessian_llama](https://github.com/Cornell-RelaxML/yaqa/tree/main/hessian_llama)
- Paper abstract (via search): "YAQA empirically reduces the KL divergence to the original model by ≈30% while achieving state of the art performance on downstream tasks"; Llama and Gemma in the body, Qwen in the appendix; KL chosen because it "considers the entire distribution over the output space, whereas perplexity and zeroshot accuracy only consider the mass of a single 'ground truth' token" — [YAQA arXiv via search](https://arxiv.org/abs/2505.22988v1)
- Third-party numbers: Llama-3.1-70B-Instruct 2-bit KL: YAQA-B 0.335 vs LDLQ 0.497 vs GuidedQuant 0.383 (Table 1: incoherence processing + QTIP quantiser, no fine-tuning) — [Liner review via search](https://liner.com/review/modelpreserving-adaptive-rounding); Llama-3.1-8B 4-bit: YAQA-B 0.013 vs LDLQ 0.019 — [Lacuna via search](https://lacuna.tiptreesystems.com/work/model-preserving-adaptive-rounding/wrk_47f838fded3199d73994cf232e1f7f2e); the arXiv HTML row reads 0.014 for YAQA-B 4-bit 8B-Instruct — [arXiv HTML via search](https://arxiv.org/html/2505.22988v1). "YAQA-B outperforming YAQA-A due to being a better (but more expensive) Hessian estimate" — same source.
- GuidedQuant (ICML 2025, arXiv 2505.07004): layer-wise output objective "equivalent to a refined quadratic approximation assuming a block-diagonal Hessian, approximated by the empirical Fisher information matrix"; gradients averaged into g groups (g=2 for Llama-2-70B); plug-in for GPTQ/SqueezeLLM/QTIP — [arXiv via search](https://arxiv.org/pdf/2505.07004)
- GuidedQuant cost (README): gradients Llama-2-7B ≈0.3 h on 1 A100; 13B ≈0.6 h on 2 A100; 70B ≈2.7 h on 6–7 A100; Hessian collection runs on one RTX 3090; calibration RedPajama 1024 × 4096 tokens for QTIP; "YAQA achieves better performance but needs more GPU resources"; code supports Qwen3 (dense), Gemma3, Llama — [GuidedQuant README](https://github.com/snu-mllab/GuidedQuant)
- REAL-Q places GuidedQuant between GPTAQ and itself in KL at W4A16 (Llama-3.1-8B: 3.98 → 3.67 → 3.36 ×10⁻²) — [REAL-Q arXiv via search](https://arxiv.org/html/2609.00049v2)
- exllamav3 has no YAQA support: README and docs mention only on-the-fly input Hessians; search for "yaqa"/"hessian" in the repo issues returned nothing — [exllamav3 README](https://github.com/turboderp-org/exllamav3)

### Inferences
- YAQA's ~30% KL reduction is measured with exactly the user's quantiser family (QTIP/trellis, incoherence processing, no FT) and holds at 4 bits (8B: 0.019 → 0.013–0.014), making it the best-attested single intervention for this setting. The user's existing Σxxᵀ Hessians serve as the LDLQ baseline; YAQA requires new sketches that involve backward passes through the bf16 model with gradients of the KL (sampled from the model's own output distribution).
- Cost scaling for 27B: with the README's node sizing ("under ~20B fit on one 8×80G node"), a 27B model with hidden 5120 / FFN 17408 will likely need two 8×80 GB nodes or a layer-split run; the Sketch-B token budget (≈134M tokens) is ~13× the user's 10M-token calibration corpus, so using the in-domain corpus alone would be an under-sampled sketch unless repeated epochs/sampled targets are used.
- GuidedQuant's diagonal-Fisher weighting is a ~10× cheaper approximation (hours on a few A100s for 70B) that captures part of YAQA's gain; it already supports dense Qwen3 but not the GDN architecture.

### Gaps
- YAQA's full per-model/per-bitrate table, Gemma/Qwen rows, and GPU-hours for the sketches were not retrievable (arXiv blocked; README results are an image).
- No source applies YAQA or GuidedQuant to a hybrid linear-attention model or to Qwen3.8-27B.

---

## Key Question 4 — Learned rotations/transforms before quantisation (SpinQuant, ParoQuant, OSTQuant, FlatQuant, DuQuant): gains over fixed Hadamard at 4-bit weight-only, and compatibility with trellis coding

### Takeaway
For weight-only 4-bit, learned rotations buy at most a few hundredths of perplexity or ~1 point of accuracy over a fixed Hadamard/QuaRot baseline on Llama-2/3 (e.g. OSTQuant 5.64 vs QuaRot 5.62 vs SpinQuant 5.58 PPL on Llama-2-7B W4A16); the big wins of these methods are in W4A4. ParoQuant (ICLR 2026) is the exception designed for weight-only reasoning models (+2.4% avg over AWQ on reasoning tasks, <10% overhead) and already supports Qwen3.8-27B. No source combines learned rotations with trellis coding; the only combination found pairs ParoQuant rotations with M2XFP4 microscaling on Qwen3.8-27B (self-reported).

### Cited Findings
- SpinQuant README (W4A16KV16, 0-shot acc / WikiText-2 PPL): LLaMA-3-8B GPTQ 66.5/7.2, QuaRot 68.4/6.4, SpinQuant 68.5/6.4; LLaMA-3-70B GPTQ 35.7/1e5, QuaRot 70.3/7.9, SpinQuant 71.6/4.8; LLaMA-2-70B GPTQ 71.9/3.9, QuaRot 72.2/3.5, SpinQuant 72.6/3.5; rotations learned by Cayley optimisation; 70B uses an FSDP script — [SpinQuant README](https://github.com/facebookresearch/SpinQuant)
- OSTQuant README (W4A16, avg 0-shot9 acc / WikiText-2 PPL): LLaMA-2-7B OSTQuant 64.37/5.64, SpinQuant 63.59/5.58, QuaRot 64.30/5.62, GPTQ 60.86/9.84, AWQ 63.89/5.83; LLaMA-2-13B 67.31/4.94 vs SpinQuant 67.14/5.00; LLaMA-3-8B 67.80/6.53 vs SpinQuant 66.54/6.49, QuaRot 67.27/6.53, GPTQ 61.03/7.43, AWQ 67.03/7.36; training `--max_steps=100`, bs 4, 4 GPUs; `--loss_type=kl_top` flag — [OSTQuant README](https://github.com/BrotherHappy/OSTQuant)
- OSTQuant paper: "KL-Top loss that computes KL divergence over only the top-k classes with the highest probabilities", argued to give cleaner gradients; weight-only "retains 99.5% of the floating-point accuracy" — [OSTQuant arXiv via search](https://arxiv.org/pdf/2501.13987)
- FlatQuant README: per-layer learnable affine (Kronecker) transforms; only W4A4 numbers given (Llama-3-8B GPTQ: FlatQuant 6.90 vs SpinQuant 7.39 vs QuaRot 8.16; 70B 3.77 vs 6.21 vs 6.60); calibration 15 epochs, bs 4 on Llama-3-8B; decode speedup at bs=1 0.71 vs Int4 0.81 (i.e. online transform costs decode throughput) — [FlatQuant README](https://github.com/ruikangliu/FlatQuant)
- ParoQuant (arXiv 2511.10645, ICLR 2026; NVIDIA/MIT/UCSD): "scaled pairwise rotation, built from independent Givens rotations plus channel-wise scaling"; "average 2.4% accuracy improvement over AWQ on reasoning tasks, with less than 10% overhead"; "retaining only the top-10% channel pairs suffices to match full rotation performance" — [arXiv via search](https://arxiv.org/abs/2511.10645)
- ParoQuant Qwen3-8B 4-bit reasoning (MMLU/GPQA/AIME24/AIME25, avg): AWQ 73.5/60.2/72.2/61.1 (59.5) vs ParoQuant 74.1/57.7/75.6/63.3 (61.9); on non-reasoning commonsense tasks ParoQuant beats AWQ, EfficientQAT, QTIP by 0.9%, 0.7%, 0.2% ("the gap is smaller there because those benchmarks generate only a few tokens"); ≈10% slower than AWQ — [arXiv HTML via search](https://arxiv.org/html/2511.10645v1)
- ParoQuant repo lists Qwen3.8-27B, Qwen3.6-27B/35B-A3B, Qwen3.5 family, Gemma 4, Llama as supported; rotations optimised by `experiments/optimize/4bit.sh`; backends vLLM/Transformers/MLX — [paroquant README](https://github.com/z-lab/paroquant)
- z-lab publishes Qwen3.8-27B-PARO; a separate Launch80 PARO-MXFP4 build "reuses z-lab's rotations" — [search result](https://huggingface.co/z-lab/Qwen3.8-27B-PARO)
- The hugypufy card combines "learned Givens rotations" (ParoQuant donor rotations, frozen) with M2XFP4 and notes a tensor-parallel split "won't work with the per-piece rotations" — [HF card via search](https://huggingface.co/hugypufy/Qwen3.8-27B-PARO-M2XFP4)
- GPTAQ's gain shrinks on rotated models (SpinQuant+GPTAQ 7.19 vs SpinQuant+GPTQ 7.26 at 4-bit) — [GPTAQ arXiv via search](https://arxiv.org/pdf/2504.02692)

### Inferences
- QTIP/EXL3 already apply a random Hadamard (incoherence processing) before trellis coding; the weight-only published deltas between learned rotations and Hadamard/QuaRot (≤0.06 PPL on Llama-2-7B W4A16) suggest the KL headroom from swapping in a learned rotation ahead of the trellis is small and unmeasured.
- ParoQuant's sparse Givens pairs (top-10% channel pairs) and per-channel scaling are mathematically composable with a subsequent Hadamard+trellis stage, but the inference kernel must apply the pairwise rotation online (≈10% overhead reported) and it breaks naive tensor-parallel splits.
- The only evidence that rotation + a second-stage format helps at 4.25 bpw on Qwen3.8-27B is the self-reported card; the rotation's contribution vs the fine-tuning's contribution cannot be separated from the published numbers.

### Gaps
- No paper reports KL (rather than PPL/accuracy) for learned rotations at W4A16; no paper combines SpinQuant/OSTQuant/FlatQuant/ParoQuant with QTIP/EXL3 trellis coding.
- DuQuant and AffineQuant were not reachable; no 4-bit weight-only numbers were gathered for them.
- ParoQuant's rotation-optimisation time and memory for a 27B model were not found (a LAQuant table reportedly lists it, but the snippet was unreadable).

---

## Key Question 5 — Small-parameter post-hoc fixes: per-channel scale re-fitting, RMSNorm-only training, LoRA/low-rank error correction (LoftQ, LQ-LoRA, QA-LoRA, CALDERA, LQER): gains and byte cost

### Takeaway
Scale/sign/norm/head tuning is what QuIP#/QTIP's end-to-end stage already does and is cheap; low-rank error correction (LQER, CALDERA, LoftQ) is reported mainly as a 2-bit or W4A8 "near-lossless" tool, and its adapter cost is not free: a rank-64 bf16 adapter on every linear of Qwen3.8-27B would add roughly 0.2–0.25 bpw. No source reports KL-to-bf16 for any of these at 4 bits.

### Cited Findings
- QuIP# end-to-end stage tunes "the sign vectors (SU/SV), layernorms, and language model head" — [quip-sharp README](https://github.com/Cornell-RelaxML/quip-sharp)
- EfficientQAT E2E-QP (step sizes only) gives 9.33 vs 8.53 (Block-AP) vs 7.68 (both), Llama-2-7B w2g64 avg PPL — [EfficientQAT arXiv via search](https://arxiv.org/pdf/2407.11062)
- TesseraQ includes optional "dequantization scale tuning" on top of block reconstruction — [TesseraQ README](https://github.com/Intelligent-Computing-Lab-Yale/TesseraQ)
- LQER (ICML 2024, arXiv 2402.02446): SVD of the quantisation error W − W_q kept as a high-precision low-rank term; "nearly-lossless W4A8"; L2QER scales the error by activation magnitudes so a smaller rank suffices and reaches W4A6 matching OmniQuant W6A6, with "no expensive knowledge distillation, hyper-parameter search, or other forms of iterative optimization" — [LQER arXiv via search](https://arxiv.org/html/2402.02446v3)
- CALDERA (NeurIPS 2024): W ≈ Q + LR with Q, L, R all quantised, QuIP# quantiser, alternating minimisation; "outperforms existing compression techniques in the regime of less than 2.5 bits per parameter"; supports LoRA fine-tuning of part of the low-rank factors — [caldera README](https://github.com/pilancilab/caldera)
- LoftQ: alternating quantisation + low-rank init; LLaMA-2-7B/13B GSM8K 4-bit 35.0/45.0 vs 16-bit 36.9/43.1 (rank 64); Phi-2 4-bit LoftQ 64.1±0.7 vs QLoRA 60.2±0.6; adapters stored separately, not merged — [LoftQ README](https://github.com/yxli2123/LoftQ)
- LQ-LoRA: Fisher-weighted, ILP mixed-precision budget (example 2.75 bits), rank 64, precomputed Fisher for LLaMA-2 7B/70B; README gives no accuracy numbers — [lq-lora README](https://github.com/HanGuo97/lq-lora)
- RILQ (arXiv 2412.01129): "Rank-Insensitive LoRA-based Quantization Error Compensation for Boosting 2-bit LLM Accuracy" — [arXiv via search](https://arxiv.org/pdf/2412.01129)
- INT2.1 (arXiv 2306.08162): LoRA on frozen quantised weights trained with KL-to-FP + CE loss — [arXiv via search](https://arxiv.org/pdf/2306.08162)
- exllamav3 self-calibration guide notes that tying k_proj+v_proj and gate_proj+up_proj to a shared bitrate costs "a small KLD" for inference speed — [optimize.md](https://github.com/turboderp-org/exllamav3/blob/master/doc/optimize.md)

### Inferences
- Byte cost of a low-rank adapter (own arithmetic, not from a source): a rank-r bf16 adapter on a d_in×d_out linear adds 16·r·(d_in+d_out) bits. For Qwen3.8-27B (hidden 5120, FFN 17408), the three FFN matrices per layer at r=64 add 3·64·(5120+17408)·2 B ≈ 8.7 MB per layer ≈ 0.55 GB over 64 layers, plus attention/GDN projections; the total is of order 0.7–0.9 GB ≈ 0.2–0.25 bpw on 26.9B params. At r=16 this is ≈0.05–0.06 bpw. Any KL gain must beat simply spending the same bits on the trellis (e.g. 4.0 → 4.25 bpw).
- Closed-form per-channel scale refit against the stored input Hessian (min_s ‖(W − diag(s)·Ŵ)‖²_H) is a zero-byte fix and is a strict subset of what QuIP#/QTIP e2e fine-tuning does; no source quantified it alone.
- RMSNorm-only training appears only as part of QuIP#'s e2e stage ("layernorms"); no source isolates its contribution.

### Gaps
- No KL numbers at 4 bits were found for LQER, CALDERA, LoftQ, LQ-LoRA or QA-LoRA (QA-LoRA was not retrievable at all).
- CALDERA's and LQ-LoRA's bit accounting formulas and 4-bit results could not be fetched (arXiv/OpenReview blocked).

---

## Key Question 6 — Risks: overfitting to the calibration distribution, long-context recurrent state, top-512 KL vs full-vocabulary KL

### Takeaway
GPTQ-style calibration is documented to overfit its calibration data, and exllamav3's own maintainers moved to a disjoint self-sampled evaluation trace because calibration/measurement/eval overlap "overstates gains"; for Qwen3.8-27B specifically, the one published study found the GDN recurrent state does not accumulate weight-quantisation error with context (state error ≈13% at 256 tokens vs ≈12.3% at 32K; PPL gap +0.73 at 4K → +0.15 at 32K). Training on top-K-truncated teacher distributions is reported to bias the student (over-confident on the top-K, miscalibrated tail), and several 2025–2026 reports found full-vocabulary or tail-corrected objectives necessary.

### Cited Findings
**Calibration overfitting**
- "On the Impact of Calibration Data in Post-training Quantization and Pruning" (arXiv 2311.09755): prior work "found GPTQ tends to overfit to the calibration data when assessing perplexity on several datasets"; downstream performance varies substantially with calibration data — [arXiv via search](https://arxiv.org/html/2311.09755v2)
- exllamav3 issue #108: "both quantization calibration and perplexity use the same dataset in EXL3, hence we have overfitting"; quants scoring lower PPL than the FP16 base on wikitext-2 cited as the symptom — [issue #108](https://github.com/turboderp-org/exllamav3/issues/108)
- optimize.md: "The same trace is used for calibration, measurement, and optimization, so evaluating on it would overstate gains"; evaluation uses a disjoint trace with new prompts and `--tool_frac` 0.3 so "overfitting [can] be measured rather than hidden"; self-sampled traces "transfer better than raw web text, which transfers poorly even with a chat template" — [optimize.md](https://github.com/turboderp-org/exllamav3/blob/master/doc/optimize.md)
- Multilingual/mixed calibration improves PPL by up to 3.52 on Llama-3.1-8B/Qwen2.5-7B vs English-only (EACL 2026) — [ACL Anthology via search](https://aclanthology.org/2026.eacl-long.223)

**Long-context recurrent state (Qwen3.8-27B)**
- NVFP4 W4A4 of all 496 linears incl. GDN: state error ≈12.3% at 32K vs 13% at 256 tokens; PPL gap +0.73 (4K) → +0.15 (32K); gate projections least sensitive (≈2.1%/2.6%); β (write-strength) noise "nearly harmless" — [arXiv 2609.04098 via search](https://arxiv.org/pdf/2609.04098)
- Contrasting work on quantising the recurrent *state* (not weights): LeapQuant (arXiv 2609.38166) argues "rounding errors accumulate" in the state and fixes it with per-window quantisation and outlier retention (near-lossless at 8 bits on Qwen/Kimi/GLM); STEPQuant (arXiv 2609.38169) finds 6-bit state ≈ FP32 on Qwen3.8-27B and Kimi-Linear-48B with AWQ weights — [opentrain.ai via search (third-party)](https://www.opentrain.ai/papers/stepquant-when-and-where-errors-matter-in-delta-rule-recurrent-state-quantizatio--arxiv-2609.38169/)
- exllamav3 self-calibration is "untested on sparse models" and the optimizer pipeline is "experimental and still a work in progress" — [optimize.md](https://github.com/turboderp-org/exllamav3/blob/master/doc/optimize.md)

**Top-K truncated KL**
- Sparse Logit Sampling (arXiv 2503.16870): caching top-K teacher probabilities "gives biased estimates of teacher probability distribution to the student, resulting in suboptimal performance and calibration"; with KL loss "the non Top-K tokens are pushed to probability 0", the student "becomes over-confident in the Top-K tokens, and under-confident for the remaining tokens"; "calibration error worsens as K is decreased"; top-K methods "still require a large number of logits (6400) to be stored" — [arXiv via search](https://arxiv.org/pdf/2503.16870v2)
- LFM2 technical report (arXiv 2511.23404): truncating the teacher then applying temperature over the full vocabulary "is unstable" (KL grows as τ²); proposes a decoupled objective matching mass on the top-K set and shape within it — [arXiv via search](https://arxiv.org/pdf/2511.23404)
- KDRL (arXiv 2506.02208) uses top-K teacher logits "re-normalize[d] … for the full-vocabulary KL"; Reward-Gated On-Policy Distillation (arXiv 2607.04037) adds a tail term collapsing all out-of-top-K tokens into one aggregate token with a binary KL on the residual mass — [arXiv via search](https://arxiv.org/pdf/2607.04037)
- "Distilling Sequential Computation" (arXiv 2609.27233): top-K settings "produced [no] reliable convergence or useful teacher–student alignment in our preliminary runs, whereas full-vocabulary distillation trained successfully" — [arXiv via search](https://arxiv.org/pdf/2609.27233)
- OSTQuant deliberately uses KL-Top (top-k only) for rotation learning, arguing cleaner gradients — [arXiv via search](https://arxiv.org/pdf/2501.13987)

### Inferences
- Because the private scoring corpus is drawn from the same distribution as the 10M-token calibration corpus, in-distribution "overfitting" is partly what the user wants; the risk is narrower: fitting the specific 10M tokens (or their self-sampled responses) rather than the distribution. The exllamav3 recipe (held-out disjoint trace, tool-call fraction) is the practical guard; for a 27B model a 1-epoch pass over 10M tokens with few trainable parameters (scales/signs/norms) is far from the regime where PV-tuning-style full-code optimisation memorises.
- For a top-512 training objective, the tail-mass correction (one aggregate "rest" token with mass 1 − Σ top-512) is the cheapest fix with literature support; evaluating on full-vocab KL against the bf16 model on a held-out slice is necessary because top-512 KL can decrease while full KL increases (tail collapse).
- The GDN study used NVFP4 (block-32 FP4), a coarser format than 4-bpw trellis; its finding that state error does not grow with context is therefore conservative for an EXL3 student, but it was measured on one model/format pair only.

### Gaps
- No source measured KL at 256K-token context for any 4-bit Qwen3.8-27B quant; the longest reported context is 32K.
- No source quantified how much a quantisation fine-tune overfits an in-domain calibration corpus when the test set is in-distribution but disjoint (the exllamav3 docs provide the methodology but publish no numbers).
- No quantisation paper trains against top-K-truncated teacher distributions, so the size of the top-512 bias specifically for QAT is unmeasured.

---

## Consolidated cost/benefit sheet for a 27B EXL3 student (from the findings above; "est." = own inference)

| Technique | Reported effect (model, bits, date) | Trainable params | Compute/VRAM reported | Bytes added |
|---|---|---|---|---|
| YAQA Kronecker KL-Hessian + QTIP | KL −≈30% (Llama-3.1-8B 4-bit 0.019→0.013/0.014; 70B 2-bit 0.497→0.335), 2025, author + third-party | none (replaces Hessian) | Sketch B: 65,536×2048 tokens with backward passes; <20B on one 8×80 GB node, 70B on two; 27B est. 1–2 nodes | 0 |
| GuidedQuant (Fisher-diag) | between GPTAQ and REAL-Q in KL; "better performance" claimed for YAQA | none | 70B gradients 2.7 h on 6–7 A100 | 0 |
| Asymmetric calibration (GPTAQ) → REAL-Q | Llama-3.1-8B W4A16 KL 0.0495→0.0398→0.0336, 2025–2026 | none / none | "20 lines"; one GPTQ-sweep-scale pass (REAL-Q adds one Adam step per 128-col block) | 0 |
| ReQuant fixed-grid code refinement | RTN→≈GPTAQ after sweeps; mixed vs GPTAQ init, Aug 2026 | integer codes only | ≈1 GPTAQ pass per sweep | 0 |
| QuIP#/QTIP e2e fine-tune | PPL gains at 2-bit; 70B ≈100 GPU-h vs <10 without FT, 2024 | RHT signs, scales, norms, lm_head | gradient checkpointing option; single node | 0 |
| Block-AP (EfficientQAT) | 2-bit 7B avg PPL 453→8.53; 4-bit 70B 3.39 vs 3.32 fp16 | all weights of one block + scales | 70B 2-bit 41 h on one A100-80GB | 0 |
| "Fine-tune under the grid" (hugypufy, M2XFP4+PARO) | Qwen3.8-27B 4.25-bit KL 0.0348→0.0211 (−39%), 1 epoch, 256 samples, 2026, self-reported | weights re-projected to grid | not stated | 0 |
| Learned rotation (ParoQuant) | +2.4% reasoning acc over AWQ (Qwen3-8B 4-bit), ICLR 2026 | Givens angles + channel scales | <10% inference overhead; no TP split | small (angles) |
| Low-rank error adapter (LQER/CALDERA/LoftQ) | W4A8 "near-lossless" (LQER); <2.5-bit regime (CALDERA); no 4-bit KL | L, R | SVD only (LQER) | est. 0.2–0.25 bpw at r=64 bf16 |
| exllamav3 self-calibration (per-tensor bits) | predicted-KLD allocation; old optimizer could worsen KL (GLM-4.6 4 bpw 0.156 → 0.18) | none | ≈2 h for 27B | 0 (reallocates) |
