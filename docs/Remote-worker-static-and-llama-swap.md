# Static remote worker and llama-swap

A dstack run takes the whole machine it runs on. On a DGX Spark that also
serves LLMs (llama.cpp), TTS or other models, the Maestro worker instead runs
as a **permanent process** next to them, and
[llama-swap](https://github.com/mostlygeek/llama-swap) is the single inference
gateway that decides who holds the GPU memory. Maestro's controller talks to
that worker in **static** mode.

```
Mac (controller) ─┐                          private compose network "inference"
clients ──────────┴▶ Spark :8080 llama-swap ─ peer /upstream/maestro-api/ ─▶ maestro-worker:7860
                     (only published port)  │                                (always running)
                                            ├ cmd ▶ maestro_worker_lease.py ── acquire/release ─┘
                                            ├ cmd ▶ llama-server small-llm   (always on)
                                            └ cmd ▶ llama-server big-llm     (swaps with the lease)
```

Static mode itself does not need llama-swap: `static_url` can point straight
at a worker (`http://host:7860/`, token = its `MAESTRO_WORKER_TOKEN`). The
Spark stack below puts the worker behind llama-swap so only one port is
published.

Status: the controller mode, the worker control API and the lease shim are
covered by unit tests. The compose stack and the llama-swap configuration
have **not been run on a Spark yet** (see [What was verified](#what-was-verified)).

## 1. Controller: static mode

**Settings → Remote GPU → Worker: Static worker**, or the API / environment
(`MAESTRO_REMOTE_<FIELD>` overrides any field, as for dstack):

| Setting | Example | Meaning |
|---|---|---|
| `provider` | `static` | `dstack` (default, unchanged behaviour) or `static` |
| `static_url` | `http://spark.tailnet:8080/upstream/maestro-api/` | the worker's base URL: the llama-swap peer route of the Spark stack, or `http://host:7860/` for a worker reached directly |
| `static_token` | `LLAMA_SWAP_API_KEY` | the bearer token of `static_url`: llama-swap's API key behind llama-swap (the peer swaps in the worker's own token), else the worker's `MAESTRO_WORKER_TOKEN`. Stored in `app/settings`, returned masked |
| `static_wake_url` | `http://spark.tailnet:8080/upstream/maestro-worker/readyz` | optional, see [Waking the worker](#waking-the-worker) |
| `route_generations` | `true` | send generations to the worker |

```bash
curl -X PUT http://localhost:7860/api/v1/remote-worker/settings \
  -H 'Content-Type: application/json' \
  -d '{"provider": "static", "static_url": "http://spark.tailnet:8080/upstream/maestro-api/",
       "static_token": "<LLAMA_SWAP_API_KEY>",
       "static_wake_url": "http://spark.tailnet:8080/upstream/maestro-worker/readyz",
       "route_generations": true}'
```

What changes in static mode:

- **No dstack call at all.** The status comes from `GET /api/v1/worker/state`
  on the worker (cached 2 s): `ready`, `busy` (rendering, still takes jobs: they
  queue on the worker), `paused`, `unreachable`, or `unknown` when the worker
  refuses the token.
- **Start, Stop and the idle stop are no-ops.** The controller never stops the
  worker process (it may be rendering, and llama-swap owns its lease), and
  never resumes a paused worker: resuming it could load a model next to the one
  llama-swap just loaded. Only the wake URL (if set) asks llama-swap for it.
- **One pod**, `jobs_per_worker` jobs held at once, price `$0/h`.
- **Jobs wait instead of failing.** While the worker is `paused`, jobs stay in
  the controller queue with the message *Remote worker paused (its GPU is lent
  to another service), waiting…*, without a timeout. While it is `unreachable`,
  they wait like for a starting dstack pod and fail after the same 45 minutes.
  A refused token fails them at once.
- **Handed-back jobs are resubmitted.** A submission the worker refuses because
  it is paused (HTTP 423), or a queued job it hands back (`cancelled` with
  `handed_back: true`), goes back to waiting and is submitted again (at most 50
  times). It is never marked failed. A job that already started on the worker
  is never handed back.
- **File transfer:** through R2 when R2 is configured on both sides (the worker
  needs the `R2_*` / `MAESTRO_R2_*` variables itself), else uploaded to the
  worker (`/api/v1/worker/files`), the same route dstack pods use through their
  proxy (here through the llama-swap peer; large uploads through it are
  untested, so prefer R2).

### Waking the worker

When jobs wait for a paused worker, the controller requests `static_wake_url`
(in the background, at most once a minute). Point it at llama-swap's
passthrough for the lease model: `/upstream/maestro-worker/readyz`. llama-swap
then loads `maestro-worker`, which swaps out `big-llm` once its in-flight
requests are done, and the worker resumes. The request carries `static_token`
as bearer token, which behind llama-swap is its API key; a URL with its own
credentials (`http://any:<key>@host:8080/...`, llama-swap reads a Basic
password as API key) is sent as is, and the credentials are redacted from the
controller's logs. Leave the URL empty to let jobs wait until someone else
loads `maestro-worker`.

## 2. Worker: authentication and control API

Port 7860 has no authentication by default (on dstack, the service proxy
authenticates every request). A worker reachable any other way **must** set:

```
MAESTRO_WORKER_TOKEN=<secret>
```

Every request then needs `Authorization: Bearer <secret>` (constant-time
compare), the UI and `/docs` included, **except** `/healthz` (liveness, always
200) and `/readyz` (200, or 503 while paused). Neither returns anything
sensitive. Without the variable nothing changes.

Control routes (worker process only, `--worker`):

| Route | Effect |
|---|---|
| `GET /api/v1/worker/state` | `{paused, running_job, queued, loaded_models, memory, idle_unload_seconds}` |
| `POST /api/v1/worker/pause` | start no new job; the running one finishes. Idempotent. New submissions get 423 |
| `POST /api/v1/worker/resume` | start jobs again. Idempotent |
| `POST /api/v1/worker/unload-queue` | every job not started yet ends `cancelled` + `handed_back`; the controller resubmits it. The running job is untouched |
| `POST /api/v1/worker/unload-models` | release the generation model, post-processors and the in-process prompt LLM, `gc` + `torch.cuda.empty_cache()`. Waits for the running job (never interrupts it); `?wait=false` answers 409 instead |
| `POST /api/v1/worker/release` | pause + unload-queue + unload-models (`?wait=false`: 409 if a job runs, after pausing and handing back) |
| `POST /api/v1/worker/acquire` | resume |

Worker environment:

| Variable | Default | Meaning |
|---|---|---|
| `MAESTRO_WORKER_TOKEN` | unset | bearer token, see above |
| `MAESTRO_WORKER_START_PAUSED` | off | start paused, so only an `acquire` lets jobs load models |
| `MAESTRO_WORKER_IDLE_UNLOAD_SECONDS` | `0` (off) | unload the models after that long without a queued or running job, even when resumed |
| `SERVER_NAME` | `127.0.0.1` | bind address (use the Tailscale / LAN address, not `0.0.0.0`) |

`/api/v1/system/release-model` (the UI's *Release model*) shares the same
release code but still refuses while anything is queued or running.

## 3. llama-swap integration

### How the lease works

The Maestro worker is a long-running container that llama-swap does **not**
start or kill: killing it mid-render would lose the job. llama-swap instead
runs `docker/spark-stack/maestro_worker_lease.py` (Python, stdlib only) as the
`cmd` of a model named `maestro-worker`:

1. llama-swap starts the shim. It calls `POST /api/v1/worker/acquire`
   (retrying while the worker is not up), then blocks. llama-swap's health
   check (`checkEndpoint: /readyz`, through `proxy:` = the worker URL) passes
   once the worker is resumed.
2. Every 15 s the shim checks the worker is still resumed and re-acquires it
   otherwise (e.g. the worker container restarted with
   `MAESTRO_WORKER_START_PAUSED=1`). So pause the worker through llama-swap
   (unload `maestro-worker`), not with `/pause`, while the lease is held.
3. A request for `big-llm` (same swap group) makes llama-swap stop
   `maestro-worker`: it sends SIGTERM to the shim, which calls
   `POST /api/v1/worker/release` and exits only when it returned, i.e. once
   the running render is done and every model is unloaded. Then llama-swap
   starts `big-llm`.
4. A request for `maestro-worker` (the controller's wake URL, a preload, or
   `/upstream/maestro-worker/...`) swaps `big-llm` out and the shim acquires
   the worker again.

The controller reaches the worker through a llama-swap **peer** (`peers.maestro`,
model `maestro-api`), not through `/upstream/maestro-worker/`:

- a request to `/upstream/maestro-worker/...` loads the lease model, so every
  status probe would swap `big-llm` out, and while the lease is stopping it
  would be refused or start another swap; a peer route only proxies (*peer
  proxying never swaps*, `internal/server/api.go`), so the controller keeps
  following a render through a release;
- llama-swap checks the caller's API key, then the peer replaces
  `Authorization` with `Bearer <peer apiKey>` (`internal/router/peer.go`), so
  the worker token never leaves the stack and the controller only knows the
  llama-swap key;
- the bare peer model name is used (`maestro-api`, not `maestro/maestro-api`):
  with a qualified name the peer router rewrites the request's model, which
  would touch the request body.

`/upstream/maestro-worker/readyz` is still the right wake URL: loading the
lease is exactly what it is for.

### Stop timeouts: set them above your longest render

From llama-swap's source (`internal/router/base.go`, `internal/process/process_command.go`, v265):

- A **swap** stops the evicted models with `healthCheckTimeout` as graceful
  timeout, then SIGKILLs their process group. `healthCheckTimeout` is also how
  long a model may take to become ready (min 15 s, default 120 s).
- A **manual / API / TTL unload** uses `unloadTimeout` (global, default 10 s,
  overridable per model).
- Without `cmdStop`, the graceful stop is SIGTERM to the process group.

So in `llama-swap.yaml`, both `healthCheckTimeout` and the `maestro-worker`
model's `unloadTimeout` are 3600 s. Meanwhile the request that triggered the
swap waits. If a render outlasts them, llama-swap kills the shim. That is
safe for the job: the worker received the release before the kill and carries
it out on its own (paused, queue handed back, models unloaded once the render
ends), and stays alive and paused; the next start of the lease resumes it. But
`big-llm` then starts loading **while the render still holds its memory**,
which can freeze the GB10. Raise the timeouts rather than relying on that.

Never give `maestro-worker` a `ttl`: the controller talks to the worker
directly, so llama-swap sees no traffic and would unload it as idle.

### Files

- [`docker/spark-stack/llama-swap.yaml`](../docker/spark-stack/llama-swap.yaml):
  `small-llm` in a `persistent` group (never unloaded by others),
  `maestro-worker` and `big-llm` in a `swap: true, exclusive: false` group (one
  at a time, other groups untouched), both preloaded at startup through
  `hooks.on_startup.preload`. Uses the `routing.router.settings.groups` form;
  the legacy top-level `groups:` key also exists but cannot be mixed with it.
- [`docker/spark-stack/compose.yaml`](../docker/spark-stack/compose.yaml):
  `ghcr.io/mostlygeek/llama-swap:unified-cuda13` (multi-arch, CUDA 13,
  `llama-server` and `python3` included, so the shim runs inside it) and
  `ghcr.io/quentinriviere/maestro-worker:spark` on a private bridge network.
  Only llama-swap's port is published, on `SPARK_BIND_ADDR` (the Tailscale
  address; `127.0.0.1` when unset, never `0.0.0.0` by default). `llama-server`
  runs inside the llama-swap container on its loopback; the worker publishes
  nothing (it listens on `0.0.0.0` inside its container, on the private network
  only, and requires its token). The worker boots like a dstack run (clone
  `MAESTRO_REPO_REF`, `remote_worker_boot.sh`), starts paused, and keeps its
  data in `MAESTRO_DATA_DIR`.
- [`docker/spark-stack/maestro_worker_lease.py`](../docker/spark-stack/maestro_worker_lease.py):
  the shim (`--url`, `--check-interval`, `--release-timeout`, `--retry`;
  `MAESTRO_WORKER_URL`, `MAESTRO_WORKER_TOKEN`). Exit codes: 0 released, 1
  release timed out (`--release-timeout`), 2 token refused.

```bash
cd docker/spark-stack
cat > .env <<'EOF'
SPARK_BIND_ADDR=100.64.0.2
MAESTRO_WORKER_TOKEN=<openssl rand -hex 32>
LLAMA_SWAP_API_KEY=sk-<openssl rand -hex 24>
MODELS_DIR=/srv/models
MAESTRO_DATA_DIR=/var/lib/maestro-worker
EOF
docker compose up -d
curl -H "Authorization: Bearer $LLAMA_SWAP_API_KEY" http://100.64.0.2:8080/upstream/maestro-api/api/v1/worker/state
curl -H "Authorization: Bearer $LLAMA_SWAP_API_KEY" http://100.64.0.2:8080/running
```

To download GGUF models and generate the llama-swap models from a web UI
(the lease model and the peer stay hand-written in a base file), add the
model hub: [Spark-model-hub.md](Spark-model-hub.md#with-the-maestro-stack).

### llama-swap image on the Spark

No custom image is needed: `ghcr.io/mostlygeek/llama-swap:unified-cuda13` is a
manifest list with `linux/amd64` and `linux/arm64` entries (checked on GHCR on
2026-10-16), and llama-swap's `docker/unified/README.md` lists its arm64 build
as CUDA 13.3.1 with `CMAKE_CUDA_ARCHITECTURES` `90;100;120;121`, i.e. native
`sm_121` code for the GB10, plus `llama-server` and `python3`. Pin a dated tag
or digest once the stack is validated.

### Other containerised services

TTS servers, ComfyUI and the like follow the Maestro pattern: their own
compose service on the `inference` network, and a small HTTP lease shim as
their llama-swap `cmd` (free memory on SIGTERM, reload on start), with
`checkEndpoint` on their readiness route. llama-swap never runs `docker run`.
Mounting `/var/run/docker.sock` into llama-swap so `cmd` / `cmdStop` can drive
containers (the pattern of llama-swap's `docker-llama` example) is possible
but **opt-in only**: access to the Docker socket is root on the host, so any
config or request path that reaches it can take over the Spark.

### Memory budget (128 GB unified)

| Share | Holder |
|---|---|
| ~10 GB | DGX OS, Docker, llama-swap, the idle Maestro process |
| ~22 GB | `small-llm`: a ~30B MoE at Q4 (~18 GB) + KV cache, resident |
| ~70 GB | **either** the Maestro worker's models **or** `big-llm` |
| ~26 GB | margin: the GB10 freezes rather than raising out-of-memory errors |

Keep one worker, `jobs_per_worker` 1, `MAESTRO_PINNED_MEMORY=0` and
`MAESTRO_KEEP_MODELS_IN_VRAM=1` (see [Remote-worker-DGX-Spark.md](Remote-worker-DGX-Spark.md)).

### Operations

```bash
# Hand the GPU to big-llm now (waits for the running render):
curl -X POST -H "Authorization: Bearer $LLAMA_SWAP_API_KEY" \
  http://spark:8080/api/models/unload/maestro-worker
# Give it back:
curl -H "Authorization: Bearer $LLAMA_SWAP_API_KEY" \
  http://spark:8080/upstream/maestro-worker/readyz
# Restart the worker container without killing a render:
curl -X POST -H "Authorization: Bearer $LLAMA_SWAP_API_KEY" \
  http://spark:8080/upstream/maestro-api/api/v1/worker/release
docker compose restart maestro-worker   # comes back paused; the lease resumes it
```

## What was verified

Checked against llama-swap's repository at commit `a2796fb` (release v265,
2026-10-14): `docs/config.example.yaml`, `config-schema.json`, `README.md` and
the Go sources named above. `llama-swap.yaml` validates against
`config-schema.json`; it was not loaded by a llama-swap binary.

| Item | Source |
|---|---|
| model `cmd`, `cmdStop`, `proxy`, `checkEndpoint` (path on `proxy`, 200 = ready), `ttl` (0 = never), `unloadTimeout`, `unlisted`, `env` | config.example.yaml, schema |
| `cmd` inherits llama-swap's environment plus `env` | process_command.go (`cmd.Env = append(cmd.Environ(), ...)`) |
| groups `swap`, `exclusive`, `persistent`, `members`; under `routing.router.settings.groups` or legacy top-level `groups`, not both | config.example.yaml, schema, internal/config/load.go |
| `hooks.on_startup.preload` | config.example.yaml, schema |
| `macros`, `${env.VAR}` (config fails if unset), `${PORT}`, `startPort` | config.example.yaml |
| `healthCheckTimeout` is the swap stop timeout; `unloadTimeout` for manual/API/TTL unloads; SIGTERM then SIGKILL of the process group | internal/router/base.go, internal/process/process_command.go |
| `/upstream/<model>/...` passthrough (loads the model), `/running`, `POST /api/models/unload/<model>`, `/health` | README.md, internal/server/server.go |
| `apiKeys`: Bearer, Basic (password) or `x-api-key`; also guards `/upstream` | internal/server/auth.go, internal/swaputil/http.go |
| `peers` (`proxy`, `apiKey` injected as `Authorization: Bearer` and `x-api-key`, `models`, `timeouts.responseHeader: 0` = none); `/upstream/<peer model>/...` proxied without swapping | config.example.yaml, internal/router/peer.go, internal/server/api.go |
| `upstream.ignorePaths` refuses (409) a matching request for an unloaded local model instead of swapping | internal/server/api.go |
| `unified-cuda13` image: amd64 + arm64, GB10 listed, `llama-server` and `python3` in `/usr/local/bin` / apt, `LLAMA_SWAP_CONFIG` / `LLAMA_SWAP_LISTEN` / `LLAMA_SWAP_WATCH_CONFIG` | README.md, docker/unified/runtime.Dockerfile |

Not verified: the stack on Spark hardware; the peer route end to end (path
rewriting, large uploads and streamed downloads through it, request capture
buffers); that the swap waits for the evicted
model's in-flight requests before stopping it (stated in base.go comments,
not exercised); `stop_grace_period` interplay when stopping the llama-swap
container; the example GGUF file names and llama-server flags; `torch.cuda.mem_get_info`
figures on the GB10's unified memory.
