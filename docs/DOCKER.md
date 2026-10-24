# Radiance in a container

The repository builds into one Docker image that holds the engine, its plugins, its tools and the
recipes. It is assembled from scratch out of the install tree, exactly the shared libraries that
tree loads, a static busybox, and `curl`, `jq` and a CA bundle for `hfget.sh`. The host needs
Docker, an AMD GPU and an `amdgpu` kernel driver that supports ROCm 10.1. It does not need ROCm, a
compiler or FFmpeg.

The image is published as [`stilldeadcode/radiance`](https://hub.docker.com/r/stilldeadcode/radiance),
with its kernels built for `gfx1201`: `docker pull stilldeadcode/radiance`. Build it yourself for
other cards or another commit. The examples below use `radiance`, the tag a local build gets; with
the published image, write `stilldeadcode/radiance` instead.

Besides radiance (Apache-2.0, `/opt/radiance/share/doc/radiance/`), the image holds glibc,
libstdc++, the ROCm runtime, FFmpeg, busybox, curl, jq and a few other libraries, each under its own
license, GPL and LGPL among them. `/usr/share/doc/THIRD-PARTY` in the image lists each one with its
version, its license files and its source. `docker/licenses.sh` writes it during the build and fails
the build if a file in the image cannot be traced to a package.

## Build the image

```sh
docker/build.sh                        # HEAD, tagged radiance:<version>, radiance:<commit>, radiance:latest
docker/build.sh -s gpu-host            # stream the commit over ssh; gpu-host builds and keeps the image
docker/build.sh -j 8                   # cap compile jobs (a device kernel takes 2-3 GB under hipcc)
docker/build.sh -J 4                   # tests at a time (default: the compile jobs)
docker/build.sh -B 5                   # every test's CPU budget times 5, for a host slower per thread
docker/build.sh -E tokenizer_perf_test # leave tests matching a regex out of the test stage
docker/build.sh -t 'gfx1200;gfx1201'   # GPU architectures to compile kernels for (default gfx1201)
docker/build.sh -r <commit>            # any other commit
docker/build.sh --target build         # stop at a stage, tagged radiance-<stage>
```

The build context is `git archive` of the commit, so the image holds only what is committed: no
build directory, no uncommitted edit, no model file. It builds with BuildKit (`docker buildx`), which
runs independent stages in parallel and caches each one; with `-s` only the remote machine needs
buildx.

The `Dockerfile` stages:

| stage | what it does |
|---|---|
| `ffmpeg` | FFmpeg 8.1 with only the demuxers `core/mm/media.cpp` admits, the decoders their streams carry (still images, H.264, HEVC, VP8, VP9, AV1 through dav1d, MPEG-1/2/4, Theora, FLV/H.263, and audio) and the three encoders the images API writes (PNG, JPEG, WebP). The distribution's build brings about 200 packages the engine never uses |
| `llvm` | ROCm's `libLLVM` rebuilt from ROCm's llvm-project at the commit the image's `clang --version` names, configured as ROCm ships it, with `docker/llvm-vopd-pairing.patch` applied. ROCm 10.1's clang does not compile the same source to the same code twice (its VOPD pairing follows heap addresses); the patch makes it repeatable. This library is the only compiler file that changes |
| `build` | `rocm/dev-ubuntu-24.04:10.1.0-full` plus cmake, ninja and g++-14. Device code is compiled by ROCm's clang from `/opt/rocm-repeatable`, a shadow of the ROCm tree whose `libLLVM` is the `llvm` stage's (`-DRAD_HIP_VOPD_REPEATABLE=ON`, `--rocm-path=/opt/rocm`); the HIP runtime and comgr in the image are ROCm's own. Configures with kernels for `GPU_TARGETS`, builds, and installs to `/opt/radiance` with the recipes. Device code is not stored compressed: the engine's AQL backend reads the fat binaries itself and refuses a compressed bundle |
| `test` | `docker/llvm-check.sh`, then `ctest -LE 'gpu\|oot'`: every suite that runs without a card. The check recompiles one kernel file four ways: with pairing off, the image's compiler must emit ROCm's device code byte for byte; with pairing on, two compiles must match the build's own object. The image is produced only if both pass |
| `rootfs` | The stripped install tree, every shared library its binaries and plugins load (resolved by `ldd`, glibc and the ROCm runtime included) in one directory the loader searches by default, a static busybox, and `curl`, `jq` and the CA bundle. ROCm's `libdrm_amdgpu` has the card name table compiled in, so no `amdgpu.ids` is needed. Every ELF file is then resolved from inside a chroot of the tree, so the build fails on a library the image lacks |
| `runtime` | `FROM scratch` and the rootfs. This is the image that ships |

The largest files are the LLVM and Clang libraries comgr loads (about 130 MB and 90 MB), without
which the HIP runtime cannot initialise a device. The shipped image has no distribution: no package
manager and no shell beyond busybox (`docker exec -it <name> sh`). JPEG XL input is refused by name,
since its decoder needs libjxl.

A build container has no card to enumerate, so the GPU architecture must be named; a bare-metal
`./build.sh` detects it. `libr4d` builds for gfx12 targets only; any other target in the list gets
no RDNA4 kernels.

The ROCm version in the image must be one the host's kernel driver supports. It is a build argument
(`ROCM_VERSION`, default `10.1.0-full`; from ROCm 10 on, the development image is published only
with the `-full` suffix).

The GPU suites (`mem_test`, `r4d_selftest`, `kbench`, `kbench_moe`) need the card to themselves.
They run from the `build` stage:

```sh
docker/build.sh --target build
docker run --rm --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined \
    radiance-build ctest --test-dir /build -L gpu --output-on-failure
```

## Run it

The entrypoint is `radiance`. The other installed tools (`rad-convert`, `rad-info`, `rad-kbench`,
`rad-tune`, `rad-schemas`, `rad-tokcheck`) are on `PATH`, and the recipes are in
`/opt/radiance/share/radiance/recipes/`.

```sh
docker run --rm -v /srv/models/rad:/models:ro --entrypoint rad-info radiance /models/m.rad

docker run -d --name radiance --network host --init --ulimit memlock=-1 \
    --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined \
    --group-add "$(getent group render | cut -d: -f3)" \
    --group-add "$(getent group video | cut -d: -f3)" \
    -v /srv/models/rad:/models:ro \
    radiance --model /models/m.rad --tp 2 --host :: --port 8000
```

The options:

- `--device /dev/kfd --device /dev/dri`, with the `render` and `video` groups, give the container
  the GPUs.
- `--security-opt seccomp=unconfined`: Docker's default seccomp profile refuses io_uring and the
  NUMA binding calls the ROCm runtime makes for host memory. Qwen3.8-Flash-Next's n-gram embedding
  table is read from the container through io_uring, so without this the model cannot serve.
- `--init`: an init process runs as PID 1 and forwards SIGTERM to the engine, which finishes its
  step and releases the cards.
- `--ulimit memlock=-1`: where no device kernel gathers the n-gram table (a run on the host
  backend), `--ngram-placement ram` locks a copy of it in RAM (47.7 GiB for Qwen3.8-Flash-Next, far
  past Docker's default memlock limit). Without the limit raised, `auto` reads the table from the
  container and `ram` refuses the start. On the cards the copy is pinned memory the cards read,
  which the limit does not count. `deploy/compose/common.yaml` sets it.
- `--network host`: the server binds the host's addresses directly, IPv4 and IPv6, with no NAT in
  front of it.

The image sets `GPU_MAX_ALLOC_PERCENT=100` and `GPU_MAX_HEAP_SIZE=100`, because the HIP runtime's
defaults cap one allocation, and the heap, below what the placement plan hands out.

A plugin of your own (a kernel library for another card, an architecture, a quantiser) is mounted
and put first on the search path; nothing in the image changes:

```sh
docker run ... -v /srv/radiance-plugins:/plugins:ro \
    -e RADIANCE_HOME=/plugins:/opt/radiance/share/radiance \
    radiance --model /models/m.rad
```

`/plugins` holds any of `kernels/`, `architectures/` and `quantizers/`, built against the same
radiance version (`docs/PLUGIN.md` §13). The engine in the image starts on any AMD GPU family: the
image's `GPU_TARGETS` are what its bundled kernel library, libr4d, is compiled for, not a limit on
the cards the engine runs on.

## Compose

`deploy/compose/` has one file per model and a `common.yaml` that each of them extends:

| file | container | what it is |
|---|---|---|
| `flashnext.yaml` | `qwen3.8-next-flash-fp8-iq4r-moe.rad` | Qwen3.8-Flash-Next, 4-bit codebook experts and an int8 trunk; MTP depth 3; finished conversations kept in host memory (`--prefix-cache-mib`) |
| `flashnext-tp1.yaml` | `qwen3.8-next-flash-fp8-iq4r-moe.rad` | The same on one card: a 42 GiB pinned host pool for the experts that do not fit, 4096-token prefill steps, 4 sequences |
| `flashnext-bf16.yaml` | `qwen3.8-flash-next-bf16.rad` | Qwen3.8-Flash-Next in bf16, streamed from the disk tier. A reference, not a production server |

All three Flash-Next files serve `qwen-fixed-chat-template.jinja` (froggeric/Qwen-Fixed-Chat-Templates v22.5, Apache-2.0) through `--override-chat-template`, in place of the template the container carries. Drop the flag and the `volumes` entry to go back to the container's own.
| `qwen3.6-35b-a3b.yaml` | `qwen3.6-35b-a3b-fp8.rad` | Qwen3.6-35B-A3B, FP8 |
| `ornith-1.5-35b-a3b.yaml` | `ornith-1.5-35b-a3b-fp8.rad` | Ornith-1.5-35B-A3B, FP8, MTP, vision |
| `qwen3.8-27b-fp8.yaml` | `qwen3.8-27b-fp8.rad` | Qwen3.8-27B FP8 with its DFlash2 drafter and vision tower |
| `qwen3.8-27b-mxfp4.yaml` | `qwen3.8-27b-mxfp4.rad` | Qwen3.8-27B with AMD's Quark AWQ-MXFP4 trunk, its DFlash2 drafter and vision tower |
| `qwen3.8-27b-paro.yaml` | `qwen3.8-27b-paro-int4.rad` | Qwen3.8-27B ParoQuant int4 (learned rotations) with its DFlash2 drafter and vision tower, on one card; the int5 and MXFP4 containers serve from it too |
| `qwen3.8-27b-bf16.yaml` | `qwen3.8-27b-bf16.rad` | Qwen3.8-27B in bf16 |
| `gemma4-31b-w4a8.yaml` | `gemma4-31b-w4a8.rad` | Gemma 4 31B, int4 trunk with int8 activations, its assistant head as the MTP drafter, vision tower |
| `gemma4-31b-heretic-ara-w4a8.yaml` | `gemma4-31b-heretic-ara-w4a8.rad` | The same for Gemma 4 31B heretic-ara (refusals removed) |
| `translategemma-27b.yaml` | `translategemma-27b-mxfp4-gptq.rad` | TranslateGemma 27B MXFP4 with its chat template (`gemma3.jinja`) |
| `glm53-flash.yaml` | `glm53-flash-mxfp4.rad` | GLM-5.3-Flash, MXFP4 experts and block-FP8 attention, on eight cards |
| `minicpm5-2b.yaml` | `minicpm5-2b-fp8.rad` | MiniCPM5-2B FP8 with its DSpark drafter |
| `minicpm5-2b-bf16.yaml` | `minicpm5-2b-bf16.rad` | MiniCPM5-2B in bf16 with its DSpark drafter |
| `embeddinggemma2.yaml` | `embeddinggemma2-bf16.rad` | EmbeddingGemma 2 in bf16: text, image, video and audio embeddings on one card (`RADIANCE_GPU`) |
| `qwen-image-2.1.yaml` | `qwen-image-2.1-fp8.rad` | Qwen-Image 2.1 as block fp8 on two cards: OpenAI's images API, generation and edits |
| `qwen-image-2.1-turbo.yaml` | `qwen-image-2.1-turbo-fp8.rad` | Qwen-Image 2.1 Turbo, the same in 8 steps |
| `krea-2-turbo.yaml`, `krea-2-raw.yaml` | `krea-2-turbo-fp8.rad`, `krea-2-raw-fp8.rad` | Krea 2 Turbo and Raw as block fp8: text to image on one card (`RADIANCE_GPU`) |
| `anima.yaml` | `anima-aesthetic-fp8.rad` | Anima Aesthetic as block fp8: text to image on one card (`RADIANCE_GPU`); the Base and Turbo containers serve the same way |
| `flux-3-klein.yaml` | `flux-3-klein-fp8.rad` | FLUX.3 [klein] as block fp8: generation and edits on one card (`RADIANCE_GPU`); the base container serves the same way |
| `sdxl.yaml` | `illustrious-xl-v2.0-bf16.rad` | Stable Diffusion XL in bf16 (here Illustrious-XL v2.0; any single-file SDXL checkpoint converts the same way): text to image on one card (`RADIANCE_GPU`) |
| `qwen3-asr-1.7b.yaml` | `Qwen3-ASR-1.7B-hf/` (the checkpoint) | Qwen3-ASR 1.7B: `/v1/audio/transcriptions` and live `/v1/realtime`, on one card (`RADIANCE_GPU`) |
| `minimax-h3.yaml` | `minimax-h3-fp8.rad` | MiniMax-H3 as block fp8: video with its stereo soundtrack on `/v1/videos`, two cards and about 39 GiB of pinned host memory |
| `fasth3.yaml` | `fasth3-8-step-v2-fp8.rad` | FastVideo's FastH3 8-Step V2, the same in eight steps |
| `ltx-2.5.yaml` | `ltx-2.5-distilled-fp8.rad` | LTX-2.5 distilled as block fp8: video with its stereo soundtrack on `/v1/videos`, one card (`RADIANCE_GPU`), the text encoder in pinned host memory; the base container serves the same way |
| `minimax-music3.yaml` | `minimax-music3-bf16.rad` | MiniMax-Music3 in bf16: songs from lyrics and a description on `/v1/audio/speech` |
| `breeze-tts2.yaml` | `breeze-tts2-bf16.rad` | Breeze-TTS-2 in bf16: speech on `/v1/audio/speech` and live over `/v1/audio/speech/stream`, on one card (`RADIANCE_GPU`) |

The container names are the files the README's model table downloads or converts.

Copy the directory to the serving host and fill in `.env` from `.env.example`. It sets the image,
the models directory (mounted read-only at `/models`), the API key, the port, the user and GPU
groups the server runs as, and the card a one-card model uses. The models directory must be on a
local filesystem that does direct I/O, such as ext4, XFS or btrfs: Qwen3.8-Flash-Next reads its
n-gram table from the container that way. On btrfs with compression on, extents stored compressed
are read through the page cache instead, which works but loses the benefit of direct reads; a
container copied into a directory set `chattr +C` is stored uncompressed. Then:

```sh
docker compose -f flashnext.yaml up -d      # start; restarts with the Docker daemon
docker compose -f flashnext.yaml logs -f    # follow the log
docker compose -f flashnext.yaml down       # stop
```

Every model listens on the same port and most take both cards (`flashnext-tp1.yaml`,
`qwen3.8-27b-paro.yaml` and the files that read `RADIANCE_GPU` take one, `glm53-flash.yaml`
eight), so run one at a time: `down` one before `up` on another. The server runs as the host user given in `.env`. `/health`, `/ping` and
the dashboard page answer without the API key, and the health check fetches `/health` with
busybox's `wget`.
