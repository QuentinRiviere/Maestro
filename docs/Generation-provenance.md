# Generation provenance

Every new generation records what produced it. A compact chip shows it on
gallery cards, completed queue tiles and Director shot cards, for example:

```
MiniMax H3 · INT8 · PDD 8-step · FlashVSR x1.5 · A100 80GB PCIe · 1 m 26 s · $0.043
```

Hover the chip for every recorded fact. Click it, or open the gallery details
panel or a Director shot, to see the full list.

## What is recorded

| Group | Facts |
| --- | --- |
| Model | Model type, display name, architecture, transformer / video VAE / text encoder files |
| Runtime | Transformer quantization read from the loaded file name (plus the configured one when the name says nothing), text encoder variant, VAE, attention backend actually used (SDPA, Sage, Sol, SLA…), compile on/off, inference backend, memory profile, per-job precision overrides |
| LoRAs | Name and multiplier of every active LoRA |
| Acceleration | Step count the engine ran, Turbo preset (only when Turbo is on), fused or baked Turbo checkpoint, step caching, sparse attention |
| Postprocessing | Spatial upsampling (FlashVSR, DLSS, Lanczos), frame interpolation, film grain |
| GPU | Device name and VRAM of the machine that ran the model; for the remote worker also the dstack backend, region, instance type, spot and $/h |
| Timing | Wall time from the generation slot to the last output, active generation time, and a breakdown into download, model load, text encode, denoise, VAE decode and postprocess when the progress stream revealed those phases |
| Cost | Remote jobs only: worker time × the pod's hourly price, as an estimate |

Facts Maestro could not observe are left out rather than guessed. Older outputs
have no record and show no chip.

## Where it is stored

* `provenance` on the job, returned by `GET /api/v1/status/{job_id}`,
  `GET /api/v1/jobs/{job_id}` and `GET /api/v1/jobs`.
* `provenance` in each output's `.meta.json` sidecar, returned by the output
  metadata endpoint, so it survives restarts. A sidecar's timing names that
  output's own generation time.
* `image_provenance` and `video_provenance` on each Director clip in the saved
  run state, read from the start image and clip sidecars.

Tool outputs (upscale, film grain, media flow) record the tool, its processing,
the GPU and the elapsed time.

## Remote worker

The worker builds the record itself, where the model runs: its GPU, files,
attention and phase timings are measured, not inferred. The controller copies
it back with the outputs and their `.meta.json` sidecars, then adds what only
dstack knows: backend, region, instance type, the pod's price, the time spent
waiting for the worker and the estimated cost. The local sidecar keeps the
controller's job id and file name.

## Limits

* The phase breakdown depends on the status messages a model emits. Models
  that report no "Encoding Prompt" or "VAE Decoding" phase show only the phases
  they report; the model load time comes from the engine itself.
* The quantization label comes from the checkpoint file name. A file without a
  quantization token shows the configured quantization instead.
* The cost is an estimate from the hourly price; dstack's billed run cost also
  covers boot and idle time and is tracked separately in the Director cost
  report.
* H3 sequence shots split from one render carry provenance only when their
  split files have a sidecar with a record.
