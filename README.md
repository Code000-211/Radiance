This is strictly just a Development fork, every Credit goes to https://codeberg.org/StillDeadcode

## What this fork changes

Upstream is the project credited above and in [`NOTICE`](NOTICE) (Deadcode and the radiance
contributors); this repository is a development fork of it. The upstream history in this tree, the
[`LICENSE`](LICENSE) and the [`NOTICE`](NOTICE) are kept as they are. The changes below sit on top
of the last commit in this history authored outside this fork (`cb5a69a`, 2026-10-17); everything
else is the upstream tree.

Merged on `main`:

- [#12 — feat(sampler): maximal coupling for linear MTP](https://github.com/Code000-211/Radiance/pull/12):
  an opt-in mode (`RADIANCE_MTP_COUPLED=1`) in which the linear MTP drafter samples its proposal and
  the target sampler maximally couples it.
- [#13 — perf(sampler): cooperative FP64 mass-cache in the coupled verify](https://github.com/Code000-211/Radiance/pull/13):
  the coupling's verify kernel produces the FP64 masses cooperatively; the ordered sums and every
  arithmetic contract stay as they were.

Experimental, **not merged and not approved**:

- [#14 — sched: the adaptive draft policy's contracts, fakes and profile/manifest formats (P00)](https://github.com/Code000-211/Radiance/pull/14):
  the contract basis for an adaptive draft scheduler (cycle views, immutable plans, observations,
  the four hooks, and the profile and manifest formats), still under review. The sched, runtime and
  test PRs stacked on it (#15–#20) target that branch, not `main`. None of this is part of the
  merged tree, and this fork makes no performance or correctness claim for it.

# radiance

An inference server for AMD GPUs, in C++ and HIP: language models, and image, video, speech and
music generation. Its HTTP API speaks OpenAI's chat, completions, Responses, images, audio and video
APIs, Anthropic's Messages API and llama-server's native one, so their clients and SDKs work
unchanged. Model architectures, kernel libraries and quantisers are `.so` plugins; the ones in this
tree are built and loaded the same way a third party's are ([`docs/PLUGIN.md`](docs/PLUGIN.md)).

The bundled device kernels (`libr4d`) target AMD RDNA4 (`gfx1200`, `gfx1201`). The serve flags
below are for two 32 GB cards (Radeon AI PRO R9700). The engine starts on any AMD GPU family;
another card needs a kernel library for it. A machine with no GPU builds the host backend, whose
AVX-512/AVX2 kernels (`libavx`) serve every model here: the small ones (EmbeddingGemma 2,
Qwen3-ASR, the 0.8B-3B models) at interactive speed on a desktop CPU, the 27B and larger far more
slowly than a card.

To support the development of radiance: [Patreon](https://patreon.com/StillDeadCode).

## Models

| model | container | get it | compose file |
|---|---|---|---|
| Qwen3.8-Flash-Next: 4-bit experts, int8 trunk, MTP, vision | `qwen3.8-next-flash-fp8-iq4r-moe.rad` (114 GiB) | [download](https://huggingface.co/StillDeadcode/qwen3.8-next-flash-fp8-iq4r-moe) | `flashnext.yaml` |
| Qwen3.8-27B FP8, DFlash2 drafter, vision | `qwen3.8-27b-fp8.rad` (29 GiB) | [download](https://huggingface.co/StillDeadcode/qwen3.8-27b-fp8) | `qwen3.8-27b-fp8.yaml` |
| Qwen3.8-27B MXFP4 (AMD Quark AWQ), DFlash2 drafter, vision | `qwen3.8-27b-mxfp4.rad` (19 GiB) | [download](https://huggingface.co/StillDeadcode/qwen3.8-27b-mxfp4) | `qwen3.8-27b-mxfp4.yaml` |
| Qwen3.6-35B-A3B FP8, MTP | `qwen3.6-35b-a3b-fp8.rad` (35 GiB) | [download](https://huggingface.co/StillDeadcode/qwen3.6-35b-a3b-fp8) | `qwen3.6-35b-a3b.yaml` |
| Ornith-1.5-35B-A3B FP8, MTP, vision | `ornith-1.5-35b-a3b-fp8.rad` (35 GiB) | [download](https://huggingface.co/keptn/ornith-1.5-35b-a3b-fp8) | `ornith-1.5-35b-a3b.yaml` |
| Ornith-1.5-35B-A3B w4a8: 4-bit experts (GPTQ), MTP, vision, fits one card | `ornith-1.5-35b-a3b-w4a8.rad` (20 GiB) | [download](https://huggingface.co/keptn/ornith-1.5-35b-a3b-w4a8) | `ornith-1.5-35b-a3b-w4a8.yaml` |
| MiniCPM5-2B FP8, DSpark drafter | `minicpm5-2b-fp8.rad` (3 GiB) | [download](https://huggingface.co/StillDeadcode/minicpm5-2b-fp8) | `minicpm5-2b.yaml` |
| MiniCPM5-2B bf16, DSpark drafter | `minicpm5-2b-bf16.rad` (5 GiB) | [download](https://huggingface.co/StillDeadcode/minicpm5-2b-bf16) | `minicpm5-2b-bf16.yaml` |
| Gemma 4 31B: GPTQ int4 trunk, int8 activations, MTP, vision | `gemma4-31b-w4a8.rad` (20 GiB) | [download](https://huggingface.co/keptn/gemma4-31b-w4a8) | `gemma4-31b-w4a8.yaml` |
| Gemma 4 31B heretic-ara (refusals removed): GPTQ int4 trunk, int8 activations, MTP, vision | `gemma4-31b-heretic-ara-w4a8.rad` (20 GiB) | [download](https://huggingface.co/keptn/gemma4-31b-heretic-ara-w4a8) | `gemma4-31b-heretic-ara-w4a8.yaml` |
| Gemma 4 31B: int4 trunk, int8 activations and lm_head, DFlash drafter, vision | `gemma4-31b-dense-w4a8-df.rad` (20 GiB) | [convert](#convert-a-checkpoint) | `gemma4-31b-w4a8-df.yaml` |
| Gemma 4 31B Q8: int8 trunk and lm_head (Q8_0's grid at group 128), int8 DFlash drafter, vision | `gemma4-31b-dense-q8-df.rad` (33 GiB) | [convert](#convert-a-checkpoint) | `gemma4-31b-q8-df.yaml` |
| Qwen-Image 2.1 FP8: text to image and image edits | `qwen-image-2.1-fp8.rad` (16 GiB) | [download](https://huggingface.co/StillDeadcode/radiance-qwen-image-2.1-fp8) | `qwen-image-2.1.yaml` |
| Qwen-Image 2.1 Turbo FP8: the same in 8 steps | `qwen-image-2.1-turbo-fp8.rad` (16 GiB) | [download](https://huggingface.co/StillDeadcode/radiance-qwen-image-2.1-turbo-fp8) | `qwen-image-2.1-turbo.yaml` |
| Krea 2 Turbo FP8: text to image, one card | `krea-2-turbo-fp8.rad` (16 GiB) | [download](https://huggingface.co/StillDeadcode/krea-2-turbo-radiance-fp8) | `krea-2-turbo.yaml` |
| Krea 2 Raw FP8: the same, undistilled | `krea-2-raw-fp8.rad` (16 GiB) | [download](https://huggingface.co/StillDeadcode/krea-2-raw-radiance-fp8) | `krea-2-raw.yaml` |
| FLUX.3 [klein] FP8: text to image and edits from up to ten pictures, one card | `flux-3-klein-fp8.rad` (9.3 GiB) | [download](https://huggingface.co/StillDeadcode/radiance-flux-3-klein-fp8) | `flux-3-klein.yaml` |
| FLUX.3 [klein] base FP8 | `flux-3-klein-base-fp8.rad` (9.3 GiB) | [download](https://huggingface.co/StillDeadcode/radiance-flux-3-klein-base-fp8) | `flux-3-klein.yaml` |
| LTX-2.5 distilled FP8: video with its stereo soundtrack from text or a first frame, one card | `ltx-2.5-distilled-fp8.rad` (37 GiB) | [download](https://huggingface.co/StillDeadcode/radiance-ltx-2.5-distilled-fp8) | `ltx-2.5.yaml` |
| LTX-2.5 base FP8 | `ltx-2.5-fp8.rad` (37 GiB) | [download](https://huggingface.co/StillDeadcode/radiance-ltx-2.5-fp8) | `ltx-2.5.yaml` |
| MiniMax-H3 FP8: video with its stereo soundtrack from text | `minimax-h3-fp8.rad` (60 GiB) | [download](https://huggingface.co/StillDeadcode/minimax-h3-fp8) | `minimax-h3.yaml` |
| FastH3 8-Step V2 FP8: the same in eight steps | `fasth3-8-step-v2-fp8.rad` (62 GiB) | [download](https://huggingface.co/StillDeadcode/fasth3-8-step-v2-fp8) | `fasth3.yaml` |
| MiniMax-Music3 bf16: songs from lyrics and a description | `minimax-music3-bf16.rad` (20 GiB) | [download](https://huggingface.co/StillDeadcode/minimax-music3-bf16) | `minimax-music3.yaml` |
| Breeze-TTS-2 bf16: speech, whole, streamed or live, one card | `breeze-tts2-bf16.rad` (5.5 GiB) | [download](https://huggingface.co/StillDeadcode/radiance-breeze-tts2-bf16) | `breeze-tts2.yaml` |
| Qwen3.8-27B ParoQuant int4, int5 or MXFP4 (learned rotations), DFlash2 drafter, vision | `qwen3.8-27b-paro-int4.rad` (19 GiB; int5 22 GiB) | [convert](#convert-a-checkpoint) | `qwen3.8-27b-paro.yaml` |
| TranslateGemma 27B MXFP4 (GPTQ), translation; optional DFlash2 drafter | `translategemma-27b-mxfp4-gptq.rad` (17 GiB) | [convert](#convert-a-checkpoint) | `translategemma-27b.yaml` |
| GLM-5.3-Flash: MXFP4 experts and block-FP8 attention (AMD Quark), MTP, eight cards | `glm53-flash-mxfp4.rad` (171 GiB) | [convert](#convert-a-checkpoint) | `glm53-flash.yaml` |
| Qwen3.8-27B bf16 | `qwen3.8-27b-bf16.rad` (50 GiB) | [convert](#convert-a-checkpoint) | `qwen3.8-27b-bf16.yaml` |
| Qwen3.8-Flash-Next bf16, a disk-streamed reference | `qwen3.8-flash-next-bf16.rad` (335 GiB) | [convert](#convert-a-checkpoint) | `flashnext-bf16.yaml` |
| Stable Diffusion XL bf16 and its finetunes (Illustrious, NoobAI, Pony): text to image, one card | `illustrious-xl-v2.0-bf16.rad` (6.4 GiB), from any single-file SDXL checkpoint | [convert](#convert-a-checkpoint) | `sdxl.yaml` |
| EmbeddingGemma 2 bf16: text, image, video and audio embeddings, one card | `embeddinggemma2-bf16.rad` (1.4 GiB) | [convert](#convert-a-checkpoint) | `embeddinggemma2.yaml` |
| Anima Aesthetic FP8: anime and illustration, text to image, one card (Base and Turbo convert the same way) | `anima-aesthetic-fp8.rad` (3.0 GiB) | [convert](#convert-a-checkpoint) | `anima.yaml` |
| Qwen3-ASR 1.7B bf16: speech to text, `/v1/audio/transcriptions` and live over `/v1/realtime`, 30 languages, one card | the checkpoint as it ships (4.1 GiB) | [Qwen/Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) | — |

A container holds everything the server needs: the weights, the tokeniser, the chat template and
any drafter. Each download link's repository has the recipe and the exact command that made the
file. Download one into your models directory (the transfer resumes if interrupted):

```sh
mkdir -p /srv/models/rad && cd /srv/models/rad
curl -LC - -O https://huggingface.co/StillDeadcode/minicpm5-2b-fp8/resolve/main/minicpm5-2b-fp8.rad
```

The Gemma 4 container is keptn42's: GPTQ calibrated in transformers, with an int8 lm_head; its
repository has the recipe and the calibration step. To convert Google's bf16 checkpoint yourself,
`data/recipes/gemma4-31b-dense-bf16-w4a8.recipe` quantises without calibration (round-to-nearest
with a scale search, bf16 lm_head); see [Convert a checkpoint](#convert-a-checkpoint). On two R9700s
that container decodes at 103 tok/s on prose and 141 on code, against 112 and 150 for keptn42's,
and accepts 0.46 of its drafts on prose against 0.50.

TranslateGemma 27B is converted from Google's bf16 checkpoint in three steps: an int8 container to
calibrate in, its Gram matrices over translation prompts, then MXFP4 with GPTQ
(`data/recipes/translategemma-27b-*.recipe`, the steps in docs/ARCHITECTURES.md). On two R9700s it
decodes 56 tok/s a stream (205 at four) and scores mean KL 0.042 against the bf16 model on held-out
translations; with a DFlash2 drafter, 152 a stream. Its chat template takes structured turns
naming the two languages and refuses the plain turn the server renders at startup, so
`translategemma-27b.yaml` serves a plain Gemma 3 template (`deploy/compose/gemma3.jinja`) and a
client sends on `/v1/completions` the prompt the checkpoint's template renders.

GLM-5.3-Flash is converted from AMD's Quark export (`amd/GLM-5.3-Flash-Quark-MXFP4`) with
`data/recipes/glm53-mxfp4.recipe`, which keeps every weight as stored except the latent
attention's small kv_b, decoded to bf16. At 171 GiB it takes eight R9700s (`--tp 8`), every expert
resident, and its own MTP layer drafts. On eight R9700s it prefills 3958 tok/s at 6k tokens,
decodes 166 tok/s on one stream and 426 over eight, and 506 under eight streams of agent turns.
It serves text only, with a bf16 KV cache. Its template knows the reasoning efforts `low` and
`high` and renders any other as its maximum, so `glm53-flash.yaml` maps `medium` and `xhigh`
to `high`.

The ParoQuant containers keep the trunk exactly as the ParoQuant releases ship it, with the learned
Givens rotations it was trained in. Convert int4 from `z-lab/Qwen3.8-27B-PARO`, int5 from
`Launch80/Qwen3.8-27B-PARO-int5` and MXFP4 from `MisterSnrub/Qwen3.8-27B-PARO-MXFP4-v2`, each with
`data/recipes/q38-27b-paro-mxfp4-df2.recipe` and the DFlash2 drafter. A ParoQuant container made
before 1.5.0 is refused at startup; convert it again with the old one as `--reuse`, which copies
every tensor that did not change.

The Qwen-Image containers serve OpenAI's images API (`/v1/images/generations`, `/v1/images/edits`):
pictures from a prompt, and edits of up to ten pictures with an optional mask, 2048x2048 by default.
On two R9700s a 2048x2048 picture takes 61 s at the base model's 40 steps and 13 s with Turbo (56 and
12 s with `--tp-wire wht6`); 1024x1024 takes 6 s and 2.6 s. Qwen-Image 2.1 is under Qwen's research
license, which is non-commercial.

The MiniMax-H3 containers serve OpenAI's video jobs (`/v1/videos`): a video with its stereo
soundtrack from a prompt, 5 to 14 s, up to 1344x768. On two R9700s a 5.17 s clip at 1344x768 takes
about 68 s with FastH3 and 8.8 minutes at MiniMax-H3's 50 steps; both read about 39 GiB of pinned
host memory. With LBH-123-AI's latent upscaler in the container, `"latent_upscale": 2` runs the first
steps at half the size and refines at the full size: MiniMax-H3's clip then takes 3.4 minutes
(docs/GUIDE.md §5.16). MiniMax-H3's license, which FastH3 carries, does not cover the EU, the UK,
the Republic of Korea or the USA.

MiniMax-Music3 and Breeze-TTS-2 answer `/v1/audio/speech`, whole or streamed as they are made: a song
from lyrics and a description, 44.1 kHz stereo, made at about the speed it plays on two cards; and
speech in a voice the request describes, made in 0.6 of its playing time on one card, its first
streamed audio 0.21 s after the request. Breeze-TTS-2 also speaks text while it is being written,
over the `/v1/audio/speech/stream` WebSocket; it is licensed for research and non-commercial use
only.

## Run with Docker

The host needs Docker and an `amdgpu` driver that supports ROCm 10.1; it does not need ROCm. The
card works best with Resizable BAR on (Above 4G Decoding and Resizable BAR in the firmware setup);
without it the engine keeps kernel arguments in pinned host memory, and every dispatch reads them
over the link. The image
is [`stilldeadcode/radiance`](https://hub.docker.com/r/stilldeadcode/radiance), with its kernels
built for `gfx1201`. Pull it, clone this repository for its compose files, then start a model from
one of them:

```sh
docker pull stilldeadcode/radiance
git clone https://codeberg.org/StillDeadcode/radiance.git
cp -r radiance/deploy/compose ~/radiance-compose && cd ~/radiance-compose
cp .env.example .env                         # set RADIANCE_MODELS, RADIANCE_API_KEY, the GIDs
docker compose -f minicpm5-2b.yaml up -d     # or any file from the table
docker compose -f minicpm5-2b.yaml logs -f
docker compose -f minicpm5-2b.yaml down
```

Every model listens on the same port and most use both cards, so run one at a time. Running without
compose, and the reason for each `docker run` option, is in [`docs/DOCKER.md`](docs/DOCKER.md).

To build the image yourself, for example for other cards, run `docker/build.sh -t 'gfx1200;gfx1201'`
in the clone (it needs `docker buildx` and tests the build as it goes). Then set
`RADIANCE_IMAGE=radiance:latest` in `.env`.

## Build from source

You need Linux, git, a C++20 compiler, CMake 3.21+ and Ninja. For the device kernels you also need
ROCm 10.1 or 7.2 (`hipcc`; set `ROCM_PATH` if it is not `/opt/rocm`). Clone the repository and
build in it:

```sh
git clone https://codeberg.org/StillDeadcode/radiance.git && cd radiance
./build.sh                                   # configure, build for the cards present, run the fast tests
./build.sh --no-hip                          # host backend only
cmake -S . -B build -G Ninja -DRAD_GPU_TARGETS='gfx1201'   # build for cards this machine lacks
```

To install, name the prefix when the build directory is first configured
(`-DCMAKE_INSTALL_PREFIX=/opt/radiance`), build, then run `cmake --install build`. The default
plugin path, `<prefix>/share/radiance`, is compiled in from that first configure, so
`cmake --install build --prefix DIR` alone gives a tree that looks for its plugins elsewhere. The
prefix gets `bin/`, `include/radiance/`, `lib/cmake/radiance/`, `libexec/radiance/`,
`share/radiance/` and `share/doc/radiance/`.

With ROCm 10.1 the build turns the compiler's VOPD pairing (dual-issue instruction pairs) off: its
LLVM 24 pairs in an order that follows heap addresses, so two builds of the same source can differ.
The Docker image builds its compiler with the fix and keeps the pairing, which is 1.4-3.5% faster
on the fp8 prefill GEMMs; a compiler that has the same fix is declared with
`-DRAD_HIP_VOPD_REPEATABLE=ON`. ROCm 7.2 is unaffected.

## Run without Docker

```sh
export RADIANCE_HOME=$PWD/build/radiance_home   # the plugins; an install finds its own
build/bin/radiance --model /srv/models/rad/minicpm5-2b-fp8.rad --port 8000 <flags>
build/bin/radiance --model StillDeadcode/minicpm5-2b-fp8 --port 8000 <flags>
```

`--model` also takes a Hugging Face repo id, as vLLM does. The server fetches the repo with `curl`
and `jq` (`scripts/hfget.sh`) into `~/.cache/radiance/hf` (`--download-dir` puts it elsewhere),
serves the container it holds, and checks it for changes at every start; a start that cannot reach
the hub serves the copy it has. The Docker image carries both tools; in a container,
`--download-dir` must name a writable mount, since the compose files mount `/models` read-only.

These rows match the compose files, minus Flash-Next's `--prefix-cache-mib`; the three- and
four-rank rows have no compose file. The other models' flags are in their compose files.

| model | flags |
|---|---|
| Qwen3.8-Flash-Next | `--tp 2 --tp-wire wht6 --max-num-seqs 8 --max-model-len 200000 --placement expert_tiered --host-pool-mib 12288 --gpu-headroom-mib 96 --kv-cache-dtype fp8 --num-speculative-tokens 3 --max-num-batched-tokens 2048` |
| Qwen3.8-Flash-Next, four ranks | `--tp 4 --max-num-seqs 8 --max-model-len 200000 --placement expert_tiered --host-pool-mib 12288 --gpu-headroom-mib 96 --kv-cache-dtype fp8 --num-speculative-tokens 3 --max-num-batched-tokens 2048` |
| Qwen3.8-Flash-Next, three ranks (the shipped container as it is, [`docs/TP3.md`](docs/TP3.md)) | `--tp 3 --max-num-seqs 8 --max-model-len 200000 --placement expert_tiered --host-pool-mib 8192 --gpu-headroom-mib 96 --kv-cache-dtype fp8 --num-speculative-tokens 3 --max-num-batched-tokens 2048` |
| Qwen3.8-27B FP8 and MXFP4, Qwen3.6-35B-A3B | `--tp 2 --max-num-seqs 32 --max-model-len 200000 --kv-cache-dtype fp8` |
| Qwen3.8-27B bf16 | `--tp 2 --max-num-seqs 8 --max-model-len 200000 --kv-cache-dtype fp8` |
| MiniCPM5-2B FP8 and bf16 | `--tp 2 --max-num-seqs 32 --max-model-len 131072 --kv-cache-dtype fp8 --gpu-headroom-mib 160 --max-num-batched-tokens 512` |
| Qwen3.8-Flash-Next bf16 | `--tp 2 --max-model-len 32768 --placement expert_tiered --weights-disk-tier --host-pool-mib 12288 --gpu-headroom-mib 768 --deterministic --max-num-seqs 32 --max-num-batched-tokens 32768 --num-speculative-tokens 0` |
| Qwen-Image 2.1 and 2.1 Turbo | `--tp 2 --tp-wire wht6` |
| MiniMax-H3 and FastH3 | `--tp 2 --vision-tower-placement ram` |
| MiniMax-Music3 | `--tp 2` |
| Breeze-TTS-2 | `--tp 1` |

Every model here runs at up to 32 sequences a step. The Flash-Next rows (except the bf16 reference)
and the 27B in bf16 keep 8: on a hybrid model each sequence holds its own recurrent state, and 32 of
them take that memory from the attention cache and, on Flash-Next, from the expert slab. See
[`docs/GUIDE.md`](docs/GUIDE.md) §4.4.

At `--tp 4`, decode and verify steps carry the exact all-reduce in the gated residual's write, with
the MoE gather in front of it; a prefill chunk runs the unfused chain. That chain's exact
all-reduce is one-shot up to 1 MiB a message and two-shot above, which sends half the bytes on a
prefill chunk; both produce the same bits. The four-rank row runs exact. At three and four ranks
`--tp-wire wht6` selects a different lossy wire, tiered int8 over the two-shot, above
`--tp-wire-min-kb`. Eight ranks have no lossy wire, so `--tp 8 --tp-wire wht6` stops at startup
naming the all-reduce.

Four ranks is a capacity configuration: its per-step rate against `--tp 2` on the same cards has
not been measured, and its tiering flags are the two-rank recipe carried over. As at two ranks, each
rank owns a slice of every expert's width: each parity's 128-column blocks are cut contiguously over
the ranks and the odd block alternates with the parity, so every rank's share is whole blocks and
every block has one owner. The resident expert bytes per rank are smaller, so the elastic window
has a different shape: read the boot log's `elastic` line and re-derive `--host-pool-mib` before
trusting it.

`--host-pool-mib` is pinned on every card, so the four-rank row pins 4 × 12 GiB = 48 GiB of RAM
that nothing else can use and the kernel cannot swap out. The amdgpu driver caps what all processes
together pin at the machine's RAM less 1/64 and 1.5 GiB; the pool is locked user memory, which
`ttm.pages_limit` does not count. A 64 GiB machine serves `flashnext-tp1.yaml` with 49 GiB pinned
([`docs/GUIDE.md`](docs/GUIDE.md) §7).

Add `--api-key KEY` to require `Authorization: Bearer KEY` (or `X-Api-Key: KEY`). `--debug-graph`
prints every op and the kernel it resolved to, then exits without serving. A chat request:

```sh
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":50}'
```

An image model answers OpenAI's images API instead, with base64 pictures:

```sh
curl -s localhost:8000/v1/images/generations -H 'Content-Type: application/json' \
  -d '{"prompt":"a lighthouse on a cliff at sunset, oil painting","size":"1024x1024"}' \
  | python3 -c 'import sys,json,base64; open("out.png","wb").write(base64.b64decode(json.load(sys.stdin)["data"][0]["b64_json"]))'
```

`http://localhost:8000/` is the dashboard. [`docs/GUIDE.md`](docs/GUIDE.md) covers the API, memory
budgets, long context, speculative decoding and operating a server.

## Convert a checkpoint

`rad-convert` turns a Hugging Face checkpoint into a container, quantising what a recipe in
`data/recipes/` names. Download checkpoints with `scripts/hfget.sh <org/repo> <dir>`.

```sh
build/bin/rad-convert /srv/models/hf/Qwen/Qwen3.8-27B -o qwen3.8-27b-bf16.rad \
    --recipe data/recipes/q38-27b-bf16.recipe
build/bin/rad-convert /srv/models/hf/Qwen/Qwen3.8-Flash-Next -o qwen3.8-flash-next-bf16.rad --max-ctx 4096
build/bin/rad-convert /srv/models/hf/google/gemma-4-31B-it -o gemma4-31b-w4a8.rad \
    --recipe data/recipes/gemma4-31b-dense-bf16-w4a8.recipe \
    --set quantization_config.quant_method=compressed-tensors \
    --draft-model /srv/models/hf/google/gemma-4-31B-it-assistant
build/bin/rad-convert /srv/models/hf/google/gemma-4-31B-it -o gemma4-31b-dense-w4a8-df.rad \
    --recipe data/recipes/gemma4-31b-dense-bf16-w4a8-df.recipe \
    --set quantization_config.quant_method=compressed-tensors \
    --draft-model /srv/models/hf/z-lab/gemma-4-31B-it-DFlash
build/bin/rad-convert /srv/models/hf/google/gemma-4-31B-it -o gemma4-31b-dense-q8-df.rad \
    --recipe data/recipes/gemma4-31b-dense-bf16-q8-df.recipe \
    --set quantization_config.quant_method=compressed-tensors \
    --draft-model /srv/models/hf/z-lab/gemma-4-31B-it-DFlash
build/bin/rad-convert /srv/models/hf/z-lab/Qwen3.8-27B-PARO -o qwen3.8-27b-paro-int4.rad \
    --recipe data/recipes/q38-27b-paro-mxfp4-df2.recipe \
    --draft-model /srv/models/hf/z-lab/Qwen3.8-27B-DFlash2
build/bin/rad-convert /srv/models/hf/ukisai/Swift-1.5-Qwen3.8-27b-W4A16-AWQ -o swift15-27b-w4a16.rad \
    --recipe data/recipes/q38-27b-w4a16.recipe
# Any single-file SDXL checkpoint (SDXL 1.0, Illustrious, NoobAI, Pony, a civitai finetune), with
# CLIP's tokenizer.json from openai/clip-vit-large-patch14
build/bin/rad-convert /srv/models/hf/OnomaAIResearch/Illustrious-XL-v2.0/Illustrious-XL-v2.0.safetensors \
    -o illustrious-xl-v2.0-bf16.rad --recipe data/recipes/sdxl-bf16.recipe \
    --tokenizer /srv/models/hf/openai/clip-vit-large-patch14/tokenizer.json
```

With Docker, use `docker run --rm -v /srv/models:/models --entrypoint rad-convert stilldeadcode/radiance ...`.
In the image the recipes are in `/opt/radiance/share/radiance/recipes/`. A checkpoint served at its
own precision needs no container: `radiance --model /srv/models/hf/Qwen/Qwen3.8-27B-FP8`. A release
someone else quantised is read as it ships when the engine serves its form (block-scaled FP8,
Quark's MXFP4), written at its own numbers when the kernels read them in another spelling (an int4
W4A16 release, compressed-tensors' AWQ and GPTQ or AutoRound's, on the 4-bit GEMMs with
`data/recipes/q38-27b-w4a16.recipe`), and converted from its exact dequantisation when no kernel
reads it: NVFP4 (compressed-tensors' and NVIDIA ModelOpt's) to the MXFP4 kernels, an AutoRound
Flash-Next's int4 block projections to int8. [docs/GUIDE.md](docs/GUIDE.md) names the recipes of record
with their `rad-convert` commands.

## Test

```sh
ctest --test-dir build -L fast -j8         # the inner loop, under 15 s; build.sh runs it
ctest --test-dir build -LE 'gpu|oot' -j8   # everything that runs without a card (about 5 minutes)
ctest --test-dir build -L gpu              # the card tests; they need the cards to themselves
ctest --test-dir build -L oot              # rebuilds every bundled plugin out of tree against an install
```

`CMakePresets.json` names the same runs (`ctest --preset fast`, `host`, `gpu`, `full`, `asan`), and
[`docs/GUIDE.md`](docs/GUIDE.md) §1.3 describes the tiers and the test binaries' options.

## Draft acceptance

Speculative decoding drafts up to `--num-speculative-tokens` tokens a step; the target model
verifies them in one pass, and a draft survives only if it matches what the target would have
produced. A position is verified only when every earlier position survived, so the same events
admit two ratios, and they differ only in the denominator:

- **Conditional** -- accepted / *reached*: of the drafts actually verified at a position, how
  many passed. The dashboard's live "conditionally accepted" figure and the per-position bars are
  conditional. Always the higher number, and it moves with the workload.
- **Marginal** -- accepted / *drafted*: of everything the drafter proposed, how much survived.
  The engine's `draft_accept` (/stats, and `radiance:draft_acceptance_ratio` on /metrics) is
  marginal, cumulative over the life of the process. Lower by construction, and slow to move.

They cannot agree, and neither is the other one broken. At depth 3 a step drafts three tokens
but reaches position 3 only when 1 and 2 both passed, so marginal is roughly the product of the
conditionals: bars reading 56/75/77% give (0.56 + 0.56x0.75 + 0.56x0.75x0.77) / 3 = 43%. A live
63% beside a lifetime 43% is one lane, two questions. Quote the conditional when asking "how
good is the drafter right now"; quote the marginal when asking "what fraction of draft work
became text." Never compare the two across that boundary.

## Documentation

| | |
|---|---|
| [`docs/GUIDE.md`](docs/GUIDE.md) | installing, converting, serving, the HTTP API, operating a server |
| [`docs/TOOLS.md`](docs/TOOLS.md) | every flag of every tool, and the recipe syntax |
| [`docs/DOCKER.md`](docs/DOCKER.md) | the image and the compose files |
| [`docs/PLUGIN.md`](docs/PLUGIN.md) | writing a kernel library, architecture or quantiser plugin, including one for another AMD card |
| [`docs/ARCHITECTURES.md`](docs/ARCHITECTURES.md) | the supported model families and their geometry |
| [`docs/TP3.md`](docs/TP3.md) | any world size: the attention world and uneven delta-net heads; three ranks |
| [`docs/OPS.md`](docs/OPS.md), [`docs/KERNELS.md`](docs/KERNELS.md) | the op vocabulary, and what `rad-kbench` last measured |
| [`spec.md`](spec.md) | the design |

## License

[Apache-2.0](LICENSE), copyright Deadcode and the radiance contributors. Keep [`NOTICE`](NOTICE)
with any copy or derivative. It credits the authors and lists the MIT-licensed code the tree
includes. The model weights are not covered: each keeps the license of the model it was converted
from. The Docker image also holds third-party libraries under their own licenses, GPL and LGPL
among them; `/usr/share/doc/THIRD-PARTY` in the image lists each one with its version and source.
