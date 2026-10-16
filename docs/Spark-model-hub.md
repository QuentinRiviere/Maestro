# Spark model hub

A small web app for the DGX Spark that lists a curated set of GGUF models,
searches Hugging Face, downloads models with hash verification into the
models folder, and maps them into [llama-swap](https://github.com/mostlygeek/llama-swap)'s
configuration. It runs next to llama-swap in a Docker Compose stack, with or
without Maestro.

```
browser ── :8090 model-hub ── writes ──▶ $MODELS_DIR/<owner>/<repo>/<file>   (verified GGUF)
                    │                    config/models.yml                   (what is fetched / served)
                    │                    config/llama-swap.yaml              (rendered)
                    │ reads              config/llama-swap.base.yaml         (hand-written, never changed)
                    └ GET /running ─▶ :8080 llama-swap ── watches config/llama-swap.yaml (read-only mount)
```

Status: unit tests and tiny real downloads from the Hub pass on a Mac (see
[What was verified](#what-was-verified)). **Nothing has run on the Spark
yet**, and the llama-swap image was not started: use the
[test checklist](#test-on-the-spark) the first time.

## Files

| Path | Role |
|---|---|
| `docker/spark-stack/model-hub/` | the app (`model_hub/`), its `Dockerfile` (python:3.12-slim, multi-arch), `catalog.yml`, pinned `requirements.txt` |
| `model_hub/fetcher.py` | verified downloads, shared by the web app and the CLI |
| `model_hub/render.py` | renders `llama-swap.yaml`, validates it against llama-swap's schema (`model_hub/vendor/`, MIT) |
| `model_hub/cli.py` | `fetch` (download every entry of `models.yml`) and `render` |
| `docker/spark-stack/llama-swap-standalone/` | stack without Maestro: `compose.yaml`, `.env.example`, `config/*.example.*` |
| `docker/spark-stack/compose.model-hub.yaml` | adds the hub to the Maestro stack (`docker/spark-stack/compose.yaml`) |

## Setup (standalone stack)

```bash
# On the Spark
git clone https://github.com/QuentinRiviere/Maestro.git ~/Maestro   # or: cd ~/Maestro && git pull
cd ~/Maestro/docker/spark-stack/llama-swap-standalone
cp .env.example .env
$EDITOR .env      # SPARK_BIND_ADDR, MODELS_DIR, UID_GID, LLAMA_SWAP_API_KEY, MODEL_HUB_TOKEN (HF_TOKEN optional)
mkdir -p /home/qriv/models      # your MODELS_DIR, owned by UID_GID
docker compose up -d --build
```

- `MODELS_DIR` and `./config` must be writable by `UID_GID` (`id -u` / `id -g`).
- On its first start the hub copies `config/models.example.yml` to
  `config/models.yml` and `config/llama-swap.base.example.yaml` to
  `config/llama-swap.base.yaml`. Git ignores those copies and the rendered
  `config/llama-swap.yaml`, so `git pull` never conflicts with them.
- llama-swap waits until the hub is healthy (it renders the config at start).
- UI: `http://<SPARK_BIND_ADDR>:8090`, log in with `MODEL_HUB_TOKEN`.
  llama-swap: `http://<SPARK_BIND_ADDR>:8080` with `LLAMA_SWAP_API_KEY`.

Update: `git pull && docker compose up -d --build`.

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_HUB_TOKEN` | required | login token (16+ characters; the hub refuses to start without it) |
| `HF_TOKEN` | empty | optional read-only Hugging Face token, only given to the hub and `fetch`, only sent to `huggingface.co` |
| `MODEL_HUB_MEMORY_GB` / `MODEL_HUB_RESERVED_GB` | 128 / 10 | memory budget for the fit badges |
| `MODEL_HUB_DEFAULT_GROUP` | `on-demand` (`gpu-heavy` with Maestro) | llama-swap group of new mappings |
| `MODEL_HUB_TRUSTED_AUTHORS` | ggml-org, unsloth, bartowski, Qwen, huihui-ai, google, openai, mistralai, microsoft, nvidia, lmstudio-community, meta-llama | publishers Explore searches by default |
| `MODEL_HUB_COOKIE_SECURE` | 0 | set to 1 behind HTTPS (e.g. `tailscale serve`) |

## Usage

- **Catalog**: `catalog.yml`, hand-curated (11 entries in five categories:
  general, abliterated, vision, code, small), each pinned to a commit with the
  recommended quant, its size, an estimated memory use and a fit badge.
  *Open / download* shows the repo with the catalog's serve settings filled in.
- **Explore**: Hub search of GGUF repos (sort by downloads, trending, likes,
  update). Only trusted publishers by default; *All publishers* shows a warning.
- **Repo view**: the commit to pin (downloads always use the commit, never a
  branch), GGUF files grouped by quant with split files
  (`-00001-of-0000N.gguf`) merged, vision projectors (`mmproj*.gguf`), MTP
  detection (tags, repo name, `draft-mtp` in the model card), and per quant
  an estimated memory use for a chosen context against the budget: total −
  reserved − resident models (preloaded or in a `persistent` group). Badge:
  *fits* (≤ 85 % of what is left), *tight* (≤ 100 %), *too big*. The KV-cache
  part is a rough rule (≈ 6 KB per token per GB of weights, f16 cache); real
  values vary about 2× with the architecture.
- **Serve settings** (before downloading, or later in **Installed**): id,
  display name, model file, mmproj, `--ctx-size`, `--parallel`, MTP
  (`--spec-type draft-mtp --spec-draft-n-max N`), TTL, group, preload,
  aliases, extra `llama-server` arguments. *Preview* shows the
  `llama-swap.yaml` diff.
- **Downloads**: one job at a time, bytes, MB/s, ETA, cancel (the partial
  file stays in `$MODELS_DIR/.staging`), resume (HTTP Range).
- **Installed**: files, disk use, commit, hash status (*Re-verify* re-hashes
  in the background), mapping status, *Delete* (type the repo id: removes the
  mapping, re-renders, then deletes the files), and llama-swap's running models
  (`GET /running`, read-only).
- **llama-swap config**: the rendered file, pending changes, *Re-render*
  (after editing `llama-swap.base.yaml`).

### models.yml and rendering

`config/models.yml` is the single list of models: `repo`, `revision`,
`include` / `exclude` globs (the `fetch` CLI reads only these) and an optional
`serve:` section (keys in `model_hub/registry.py`). The hub adds an entry when
a download finishes (exact file names, pinned commit).

`config/llama-swap.yaml` = `llama-swap.base.yaml` + one model per enabled
`serve:` entry whose files are installed:

```yaml
qwen36-35b-abliterated:
  name: Qwen3.6 35B-A3B abliterated (MTP)
  description: huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated-MTP-GGUF@acd5abe0c5eb (managed by model hub)
  cmd: |
    ${llama-server}
    --model ${models}/huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated-MTP-GGUF/Huihui-Qwen3.6-35B-A3B-abliterated-ggml-model-Q4_K.gguf
    --mmproj ${models}/huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated-MTP-GGUF/mmproj-model-f16.gguf
    --ctx-size 65536 --parallel 1
    --spec-type draft-mtp --spec-draft-n-max 6
  ttl: 0
  aliases: [small-llm]
```

plus the id in its group's `members` (the group is created as `swap: true,
exclusive: false` if the base lacks it) and in `hooks.on_startup.preload`
when `preload: true`. The base must define the `llama-server` macro (and
`models`, else `/models` is used). Managed ids and aliases may not reuse a
base model, alias or peer model name. The result must pass llama-swap's JSON
schema, otherwise nothing is written; the previous file is kept as
`llama-swap.yaml.bak`, and the write is atomic (temp file + rename).

Model cards of the MTP GGUFs (May 2026) say MTP did not support `--mmproj`
or `--parallel` > 1 yet; the hub warns about that combination but renders it.

### Hot reload

Verified in llama-swap's source (commit `a2796fb`, v265): `-watch-config`
polls the config file's mtime and size every 2 s (`internal/watcher/watcher.go`,
`os.Stat`, so it works in containers) and reloads on change; `SIGHUP` reloads
too (`llama-swap.go`). The unified image maps `LLAMA_SWAP_WATCH_CONFIG` to
`-watch-config` (`docker/unified/run.sh`, default `true`). A reload builds a
new server, swaps it in, then **shuts the old one down with a 30 s timeout,
stopping every running model** (`server.Shutdown`), and the new server
re-runs the startup preloads. So any mapping change briefly unloads the
resident model too.

The compose files mount the **config directory** read-only into llama-swap,
not the single file: the hub replaces `llama-swap.yaml` by rename, and a
single-file bind mount keeps showing the old inode.

## With the Maestro stack

`docker/spark-stack/compose.model-hub.yaml` adds the hub to the Maestro stack
and points llama-swap at the rendered file. The Maestro lease model, the peer
and the groups stay hand-written in the base:

```bash
cd ~/Maestro/docker/spark-stack
mkdir -p config && cp llama-swap.yaml config/llama-swap.base.yaml
# optional: drop the placeholder small-llm / big-llm models from the base copy,
# then map real models in the hub with aliases small-llm / big-llm.
echo "MODEL_HUB_TOKEN=$(openssl rand -hex 24)" >> .env
docker compose -f compose.yaml -f compose.model-hub.yaml up -d --build
```

- New mappings default to the `gpu-heavy` group (they swap with the Maestro
  worker's lease); `resident` stays for the always-on model.
- Config watching is **off** in this stack: a reload stops the
  `maestro-worker` lease with a 30 s grace period, which can cut a render's
  release short (see [Remote-worker-static-and-llama-swap.md](Remote-worker-static-and-llama-swap.md#stop-timeouts-set-them-above-your-longest-render)).
  Apply a mapping when the worker is idle: unload `maestro-worker` first
  (`POST /api/models/unload/maestro-worker` waits for the render), then
  `docker compose -f compose.yaml -f compose.model-hub.yaml kill -s HUP llama-swap`.
- The test `test_maestro_lease_and_peer_survive` renders the repo's Maestro
  `llama-swap.yaml` with a managed model and checks the lease model, peer,
  API keys and preloads come out unchanged and the result validates.

## CLI

```bash
docker compose run --rm fetch --dry-run     # what models.yml would download
docker compose run --rm fetch               # download + verify every entry
docker compose run --rm fetch --verify      # re-hash files already in place
docker compose run --rm fetch render        # re-render config/llama-swap.yaml
docker compose run --rm fetch render --check
```

Same checks as the web app; exit code 1 on any error.

## Security model

- **No Docker socket**, no host networking. The hub can only write
  `$MODELS_DIR` and `./config`; llama-swap mounts both read-only.
- Containers: non-root (`UID_GID`), read-only root filesystem, tmpfs `/tmp`,
  `cap_drop: ALL`, `no-new-privileges`, pid limits. Ports published on
  `SPARK_BIND_ADDR` only (127.0.0.1 when unset).
- **Auth**: `MODEL_HUB_TOKEN` required. The login compares it in constant
  time and sets an HttpOnly, SameSite=Strict session cookie (12 h, in memory:
  a restart logs out). Every POST/DELETE with the cookie needs the session's
  `X-CSRF-Token`; scripts can send `Authorization: Bearer <MODEL_HUB_TOKEN>`
  instead. Failed logins: 5 per client and 30 overall per 5 minutes. No CORS
  headers; CSP `default-src 'self'` (no inline script, no CDN), `frame-ancestors 'none'`.
- **Downloads**: only `https://huggingface.co` and `*.hf.co` /
  `*.huggingface.co` (redirects elsewhere are refused); `HF_TOKEN` only sent
  to `huggingface.co` and stripped on redirect. Revisions are resolved to a
  commit and files fetched at that commit. Only `.gguf` through the UI
  (`fetch` also takes `.safetensors`, `.json`, ...); pickle and code formats
  (`.bin`, `.pt`, `.ckpt`, `.pkl`, `.py`, `.so`...) are refused. Every file is
  checked against the Hub's SHA-256 (LFS) or git blob SHA-1 before an atomic
  move into place; a mismatch discards it. Disk space is checked first (1 GB
  margin). `manifest.json` and `models.sha256` (`sha256sum -c`) record them.
- **Paths**: repo ids, revisions and file names are validated (no `..`, no
  absolute paths, no hidden files, resolved paths must stay in the models
  directory, symlinks included).
- **Rendered config**: serve values are validated; extra arguments may not
  contain spaces, quotes or `$` (no `${env.*}` expansion of llama-swap's
  secrets into a command line); the result must pass llama-swap's schema.
- `LLAMA_SWAP_API_KEY` is given to the hub only to read `/running`.

## What was verified

On a Mac (arm64), Python 3.12, 2026-10-16:

- `tests/test_spark_model_hub.py` (34 tests): rendering (standalone base,
  Maestro base with lease and peer preserved, legacy `groups`, collisions,
  schema errors, unsafe serve values, `.bak` + atomic write), split-file
  grouping and quant names, MTP detection, path safety, auth (401 without
  login, cookie flags, CSRF 403, bearer, login rate limit, refusal to start
  without a token), memory estimates and badges, the CLI.
- Network tests (skipped with `MODEL_HUB_OFFLINE=1`): real download of
  `ggml-org/models` `tinyllamas/stories260K.gguf` (1.2 MB, SHA-256) and
  `README.md` (git SHA-1), tamper detection, forged hash refused, Range
  resume; a download through the API with a mapping, rendered config
  validated, then deleted; trusted-only search; and every `catalog.yml` entry
  resolved through `GET /api/models/<repo>/revision/<commit>?blobs=true`
  (commit, file names, total size within 20 MB).
- `docker compose config` of both stacks (merged Maestro + hub override).
- llama-swap behaviour (watch, SIGHUP, reload stops models, `/running`
  payload, `-validate`, image env variables) read from its source at
  `a2796fb`, not run.

**Untested**: the hub image build and the containers (no Docker build was run),
the UI in a browser, llama-swap loading a rendered config, hot reload in
practice (including whether a preload right after a reload collides with the
old process on the same port), MTP with `--mmproj` / `--parallel`, the memory
estimates against real usage on the GB10, gated repos with `HF_TOKEN`, very
large downloads (tens of GB) and resuming them across a container restart.

## Test on the Spark

Run these in order from `~/Maestro/docker/spark-stack/llama-swap-standalone`
with `.env` filled in. Set `K=<LLAMA_SWAP_API_KEY>`, `T=<MODEL_HUB_TOKEN>`,
`H=<SPARK_BIND_ADDR>` and `MODELS_DIR=<MODELS_DIR>` in your shell first.

1. **Build and start**
   ```bash
   docker compose up -d --build
   docker compose ps
   ```
   Expect `model-hub` *healthy* and `llama-swap` *running* (llama-swap starts
   only after the hub is healthy). `ls config/` shows `models.yml`,
   `llama-swap.base.yaml`, `llama-swap.yaml`.
   `docker compose logs model-hub` shows `created models.yml from its example`
   on the first start, then `Uvicorn running on http://0.0.0.0:8090`.
2. **llama-swap reads the rendered config**
   ```bash
   docker compose logs llama-swap | grep -iE "watching|error|config"
   docker compose exec llama-swap llama-swap -config /etc/llama-swap/config/llama-swap.yaml -validate
   ```
   Expect `watching configuration for changes (poll-based, 2s interval)` and
   `config is valid: 0 model(s), 0 peer(s)` (nothing downloaded yet).
3. **Login**: `curl -s http://$H:8090/api/status` → `401`. Open
   `http://$H:8090`, log in with `$T`. The header shows free disk and the
   memory line (`128 GB − 10 reserved − 0.0 resident = 118.0 GB`).
4. **Search**: *Explore* → `qwen3`, *Most downloads* → a list of trusted
   repos with downloads, likes and dates. Tick *All publishers*: the warning
   appears and untrusted repos show a red badge.
5. **Download a small model**: *Catalog* → *Qwen3 1.7B (small test model)* →
   *Open / download*. Check the commit `daeb8e2d528a…` and the *fits* badge,
   click *Preview llama-swap.yaml diff* (adds `qwen3-1.7b`), then
   *Download (verified)*. *Downloads* shows progress, MB/s and ETA, then
   `done` with `verified 1 file(s); mapped as qwen3-1.7b (llama-swap.yaml updated)`.
   ```bash
   cat "$MODELS_DIR/models.sha256"
   (cd "$MODELS_DIR" && sha256sum -c models.sha256)    # ...: OK
   ```
6. **Rendered config**
   ```bash
   grep -A10 'qwen3-1.7b:' config/llama-swap.yaml
   docker compose exec llama-swap llama-swap -config /etc/llama-swap/config/llama-swap.yaml -validate
   ```
   Expect the `cmd` with `--model ${models}/ggml-org/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q4_K_M.gguf`
   and `config is valid: 1 model(s), 0 peer(s)`.
7. **Hot reload**
   ```bash
   docker compose logs llama-swap | grep -iE "reload"
   curl -s -H "Authorization: Bearer $K" http://$H:8080/v1/models
   ```
   Expect `reloading configuration` then `configuration reloaded` within
   ~2 s of the download finishing, and `qwen3-1.7b` in the model list.
8. **One completion**
   ```bash
   curl -s -H "Authorization: Bearer $K" -H 'Content-Type: application/json' \
     http://$H:8080/v1/chat/completions \
     -d '{"model": "qwen3-1.7b", "messages": [{"role": "user", "content": "Say hi in 3 words."}], "max_tokens": 64}'
   curl -s -H "Authorization: Bearer $K" http://$H:8080/running
   ```
   Expect a JSON answer with `choices[0].message.content` (the first request
   loads the model, a few seconds), then `qwen3-1.7b` with state `ready`. The
   hub's *Installed* tab lists it under *Running in llama-swap*.
9. **Re-verify**: *Installed* → *Re-verify hashes* → badge `hash ok`.
10. **Delete**: *Installed* → *Delete* → type `ggml-org/Qwen3-1.7B-GGUF`.
    Expect `freed 1.28 GB`, the directory gone from `$MODELS_DIR`, a new
    reload in the llama-swap log, and `qwen3-1.7b` absent from `/v1/models`.
11. **Optional, the real model**: download *Qwen3.6 35B-A3B abliterated
    (MTP, vision)* (22.6 GB). It is preloaded in the `resident` group: after
    the reload, `docker compose logs llama-swap` should show llama-server
    loading it. Look for errors about `--spec-type`, `--mmproj` or `-np`
    (MTP with mmproj): if it fails, set `mmproj` to none or MTP off in
    *Installed* → *Edit mapping*.

If a step fails, send back:

```bash
docker compose ps
docker compose logs --tail 200 model-hub llama-swap
cat config/llama-swap.yaml config/models.yml
ls -la config "$MODELS_DIR" "$MODELS_DIR/.staging"
docker compose exec llama-swap llama-swap -config /etc/llama-swap/config/llama-swap.yaml -validate
```

plus the failing command's output, and for UI problems the browser console
errors and the failing request (DevTools → Network: URL, status, response).
