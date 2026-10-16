# Remote worker on an NVIDIA DGX Spark

Maestro's remote GPU worker can run on a DGX Spark you own instead of a rented
cloud GPU. The Spark joins dstack as an **SSH fleet**, and Maestro starts its
worker there as an ordinary dstack run, with the `spark` variant of the worker
image. Nothing changes in how Maestro routes jobs, caches checkpoints in R2 or
records provenance. The hourly price is $0, so jobs show `$0.00/h` and an
estimated cost of $0.

| | |
|---|---|
| Hardware | GB10 Grace Blackwell superchip, 20 Arm cores, 128 GB unified CPU/GPU memory |
| Platform | `linux/arm64` (aarch64), GPU compute capability 12.1 (`sm_121`) |
| Image | `ghcr.io/quentinriviere/maestro-worker:spark` (`docker/remote-worker-spark/Dockerfile`) |
| Software | Python 3.10, torch 2.10.0 + CUDA 13.0 (`cu130` index), SageAttention 2.2 (sm_120 kernels), decord built from source |

## 1. Prepare the Spark

DGX OS (Ubuntu 24.04 based) ships Docker and the NVIDIA Container Toolkit.
Check both, then give dstack what it needs (see dstack's
[SSH fleet requirements](https://dstack.ai/docs/concepts/fleets/)):

```bash
# GPU visible from a container (driver 580+, CUDA 13)
docker run --rm --gpus all nvidia/cuda:13.0.3-base-ubuntu22.04 nvidia-smi

# A user with passwordless sudo, reachable with an SSH key
sudo adduser --disabled-password maestro
echo "maestro ALL=(ALL) NOPASSWD:ALL" | sudo tee /etc/sudoers.d/maestro
sudo -u maestro mkdir -p ~maestro/.ssh
cat id_ed25519.pub | sudo -u maestro tee -a ~maestro/.ssh/authorized_keys

# dstack tunnels the worker's port through SSH
grep -i '^AllowTcpForwarding' /etc/ssh/sshd_config   # must be "yes" (the default)
```

The **dstack server** (not only your laptop) must reach the Spark over SSH: on
the same LAN, or through Tailscale. The firewall should allow SSH and block
other inbound traffic. `nvidia-smi` on the Spark reports the GPU memory as
`N/A`: the GB10 has no dedicated VRAM. That is expected (see below).

## 2. Create the fleet

```bash
cp docs/examples/spark-fleet.dstack.yml spark-fleet.dstack.yml   # edit user, key, host
dstack apply -f spark-fleet.dstack.yml
dstack fleet
# FLEET  INSTANCE  GPU     PRICE  STATUS
# spark  0         GB10:1  $0     idle
```

dstack installs its shim on the host. The fleet stays up until you
`dstack fleet delete spark`. Maestro never applies, changes or deletes it.

## 3. Point Maestro at it

In **Settings → Remote GPU** (the Advanced section holds the image, backends,
cache warmer backends and host data path):

| Setting | Value | Why |
|---|---|---|
| Dedicated Maestro fleet (`manage_fleet`) | off | Maestro would otherwise apply its own fleet spec under that name |
| Fleet (`fleet`) | `spark` | the SSH fleet above |
| Backends (`backends`) | `remote` | dstack's backend for SSH fleets. `runpod` would exclude the Spark |
| GPU (`gpu`) | `GB10` | a name, not a VRAM floor: dstack reads 0 GB of VRAM on the GB10. A GB10 name also makes dstack expect an Arm host, where `48GB..` defaults to x86 |
| RAM / vCPUs / Disk | `64GB..` / `8..` / `100GB..` | the Spark offers about 119 GB, 20 cores, 1-4 TB |
| Container image (`image`) | `ghcr.io/quentinriviere/maestro-worker:spark` | the arm64 image |
| Host data path (`host_data_path`) | `/var/lib/maestro-worker` | weights, HF cache and compiled kernels stay on the Spark between runs (dstack creates the directory) |
| Cache warmer backends (`cache_warm_backends`) | e.g. `runpod` | the R2 cache warmer still needs a cloud CPU pod. It never uses `remote` |
| Pinned memory (`pinned_memory`) | off | CPU and GPU share the same memory: pinning only duplicates it |
| Keep models in VRAM (`keep_models_in_vram`) | on | moving a model "back to RAM" copies it within the same memory |
| Parallel pods (`worker_count`) | `1` | one GPU |
| `slow_pull_seconds` | `0` | the first pull of the 15-20 GB image on a home link is slow. Replacing the host makes no sense with only one host |
| Max price / Max duration | unchanged / e.g. `12h` | the instance price is $0. Max duration is still the safety net |

`slow_pull_seconds` has no field in the UI. Set it through the API, or for
any field use `MAESTRO_REMOTE_<FIELD>` environment variables on the controller:

```bash
curl -X PUT http://localhost:7860/api/v1/remote-worker/settings \
  -H 'Content-Type: application/json' \
  -d '{"fleet": "spark", "manage_fleet": false, "backends": ["remote"], "gpu": "GB10",
       "memory": "64GB..", "cpu": "8..", "disk": "100GB..",
       "image": "ghcr.io/quentinriviere/maestro-worker:spark",
       "host_data_path": "/var/lib/maestro-worker", "cache_warm_backends": ["runpod"],
       "pinned_memory": false, "keep_models_in_vram": true, "worker_count": 1,
       "slow_pull_seconds": 0, "max_duration": "12h"}'
```

To switch back to cloud GPUs, restore `backends: ["runpod"]`, `gpu: "48GB.."`,
the `latest` image and `manage_fleet: true` with `fleet: maestro`. The host
data volume is optional, so it is skipped on backends without instance volumes.

Everything else works as on a cloud pod. The R2 checkpoint cache and the
boot-time prefetch use `s5cmd` (arm64 build). Kernel caches are keyed by GPU
architecture and, on Arm hosts, by CPU architecture (`sm_121-…-aarch64`), so
they never mix with amd64 archives. Weights in R2 are architecture-independent
and shared with the cloud pods.

## 4. Build and publish the image

The `spark` job of `.github/workflows/remote-worker-image.yml` builds it
natively on GitHub's `ubuntu-24.04-arm` runner (free for public repositories)
and pushes `:spark` and `:spark-deps-<hash>`. It runs on the same triggers as
the amd64 image, or by hand (*Actions → Remote worker image → Run workflow*).
Under QEMU on an amd64 runner, compiling SageAttention and decord would take
hours.

To build by hand, on any arm64 machine (the Spark itself is the fastest, an
Apple Silicon Mac works too):

```bash
docker buildx build --platform linux/arm64 \
  -f docker/remote-worker-spark/Dockerfile \
  -t ghcr.io/quentinriviere/maestro-worker:spark --push .
```

The Dockerfile's build context is the repository root. The pod still clones
the configured branch at boot. When `app/requirements.txt` changed since the
image was built, the boot script installs the difference, which also works on
arm64 because the requirements carry `platform_machine` markers.

## What differs from the amd64 image

| Package | Spark (linux/arm64) |
|---|---|
| torch / torchvision / torchaudio | 2.10.0 / 0.25.0 / 2.10.0 from `cu130`. The aarch64 wheel ships sm_120 SASS + PTX, which run on sm_121 |
| torchcodec 0.10.0 | from the PyTorch `cu130` index (PyPI has no aarch64 wheel) |
| SageAttention 2.2 | compiled for 12.0 with CUDA 13.0. v2.2.0 has no 12.1 target, and sm_120 cubins load on sm_121. Enabled through `MAESTRO_SAGE2_EXTRA_ARCHS=12.0,12.1` and gated by a kernel probe at startup |
| FlashAttention 2 | **not installed**: no aarch64 wheel for torch 2.10. Maestro drops `flash` and the models that use it fall back to SDPA |
| onnxruntime-gpu | **onnxruntime (CPU) 1.22.1**: no cp310 aarch64 GPU wheel. rembg, DWPose and the face detectors run on the Grace CPU |
| decord 0.6.0 | built from source against FFmpeg 4.4 (CPU decoding) |
| taichi | **not installed** (no aarch64 wheel). Only the SCAIL taichi pose renderer needs it |
| Triton 3.6 | uses CUDA 13's `ptxas` (`TRITON_PTXAS_PATH`), which knows sm_121 |
| s5cmd | arm64 build |

`attention_mode: auto` picks the fastest mode that works: `sage2` when the
probe passes, else `sage` (Triton), else `sdpa`. The worker log prints
`SageAttention 2 kernel probe failed` when it falls back.

## Known limitations

- **Memory bandwidth.** The GB10's LPDDR5x (~273 GB/s) is several times
  slower than the HBM/GDDR of A100/H100/L40S cards. Video diffusion is much
  slower per step than on a cloud GPU. The Spark suits image models, long
  unattended queues, and models too large for a 48 GB card.
- **Unified memory.** NVML reports no dedicated memory, so the worker's live
  stats show no GPU memory gauge, and dstack lists the GB10 with 0 GB VRAM.
  CPU-side copies of weights (mmgp loads to RAM before moving to the GPU) use
  the same 128 GB. Very large models can run out of memory, and the Spark is
  known to freeze instead of raising a CUDA out-of-memory error. Keep one
  worker and one job per worker.
- **Not validated on GB10 hardware yet:** SageAttention's sm_120 kernels on
  sm_121, Triton and `torch.compile` kernels, the INT8 Quanto kernels, and the
  comfy-kitchen CUDA backend. Each one falls back (probe, guarded import or
  SDPA), but its speed is unmeasured.
- **Cache warmer:** it stays on a cloud CPU pod (amd64 `cpu` image). It fills
  the same R2 cache the Spark reads.
- **First boot:** the image weighs 15-20 GB. Docker keeps it on the Spark, so
  only the first run, or a run after an image update, pulls it.

## Blackwell / NVFP4 opportunities

The GB10's 5th-generation tensor cores run FP4 (NVFP4) natively. With 128 GB
shared between weights and activations, 4-bit weights matter more here than on
a cloud card:

- Maestro's `nvfp4` quantization type uses the comfy-kitchen CUDA backend,
  which ships aarch64 wheels. NVFP4 checkpoints may run with native FP4 GEMMs
  on the Spark. Untested.
- SageAttention 3 (FP4 attention for Blackwell, `sage3` in Maestro) is not
  built in this image. It is the next attention kernel to try for sm_120/121.
- A native `sm_121` build (CUDA 13 targets it directly) of SageAttention and
  of torch itself avoids relying on sm_120 binary compatibility. Community
  builds report noticeably faster end-to-end runs on GB10.
