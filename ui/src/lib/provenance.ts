/**
 * Generation provenance (backend: app/services/generation_provenance.py):
 * which model, runtime, LoRAs, acceleration, postprocessing and GPU produced
 * an output, and how long it took. Every field is optional — the backend
 * omits what it could not observe, and the UI shows only what is there.
 */

export interface ProvenanceLora {
  name: string
  multiplier?: number | string
}

export interface GenerationProvenance {
  version?: number
  captured_at?: number
  model?: {
    type?: string
    name?: string
    architecture?: string
    files?: { transformer?: string; video_vae?: string; text_encoder?: string }
  }
  runtime?: {
    inference_backend?: string
    transformer_quantization?: string
    configured_quantization?: string
    transformer_dtype?: string
    text_encoder?: string
    text_encoder_quantization?: string
    video_vae?: string
    video_vae_precision?: string
    vae_precision?: string
    attention?: string
    compile?: boolean
    memory_profile?: number | string
    model_reloaded?: boolean
    precision_overrides?: Record<string, string>
  }
  loras?: ProvenanceLora[]
  acceleration?: {
    steps?: number
    turbo?: boolean
    turbo_preset?: string
    turbo_preset_label?: string
    fused_turbo?: boolean
    baked_turbo?: boolean
    step_cache?: string
    step_cache_multiplier?: number
    step_cache_start_percent?: number
    sparse_attention?: string
  }
  postprocessing?: {
    spatial_upsampling?: string
    temporal_upsampling?: string
    film_grain_intensity?: number
    film_grain_saturation?: number
  }
  hardware?: {
    location?: 'local' | 'remote' | string
    gpu?: string
    gpu_count?: number
    vram_gb?: number
    compute_capability?: string
    backend?: string
    region?: string
    instance_type?: string
    spot?: boolean
    price_per_hour?: number
    pod?: string
  }
  timing?: {
    total_seconds?: number
    generation_seconds?: number
    worker_seconds?: number
    remote_wait_seconds?: number
    remote_work_seconds?: number
    phases?: Partial<Record<ProvenancePhase, number>>
  }
  cost?: { estimated_usd?: number; price_per_hour?: number; basis?: string }
}

export type ProvenancePhase = 'download' | 'model_load' | 'text_encode' | 'denoise' | 'vae_decode' | 'postprocess'

export const PHASE_LABELS: Record<ProvenancePhase, string> = {
  download: 'Download',
  model_load: 'Model load',
  text_encode: 'Text encode',
  denoise: 'Denoise',
  vae_decode: 'VAE decode',
  postprocess: 'Postprocess',
}

const ATTENTION_LABELS: Record<string, string> = {
  sdpa: 'SDPA', sage: 'Sage', sage2: 'Sage 2', sage3: 'Sage 3', flash: 'Flash', xformers: 'xFormers',
  sol: 'Sol', sla: 'SLA', radial: 'Radial',
}

const STEP_CACHE_LABELS: Record<string, string> = {
  first_block: 'First Block Cache', tea: 'TeaCache', mag: 'MagCache',
}

function isNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

/** "86 s", "2 m 08 s", "1 h 04 m": compact enough for a chip. */
export function formatSeconds(seconds: number | undefined | null): string {
  if (!isNumber(seconds)) return ''
  const total = Math.max(0, Math.round(seconds))
  if (total < 60) return `${total} s`
  const hours = Math.floor(total / 3600)
  const minutes = Math.floor((total % 3600) / 60)
  const rest = total % 60
  if (hours) return `${hours} h ${String(minutes).padStart(2, '0')} m`
  return `${minutes} m ${String(rest).padStart(2, '0')} s`
}

/** "$0.043", "<$0.001". */
export function formatUsd(value: number | undefined | null): string {
  if (!isNumber(value)) return ''
  if (value > 0 && value < 0.001) return '<$0.001'
  return `$${value < 1 ? value.toFixed(3) : value.toFixed(2)}`
}

export function attentionLabel(value: string | undefined): string {
  if (!value) return ''
  return ATTENTION_LABELS[value.toLowerCase()] ?? value
}

const SPATIAL_LABELS: Record<string, string> = {
  'dlss5*': 'DLSS 5', lanczos: 'Lanczos', flashvsr: 'FlashVSR', flashvsr2pass: 'FlashVSR 2-pass',
}

/** "FlashVSR x1.5" from "flashvsr1.5" (see services/media_info.spatial_details). */
export function upsamplingLabel(value: string | undefined): string {
  if (!value) return ''
  const match = /^(dlss5\*|lanczos|flashvsr2pass|flashvsr)([0-9.]+)$/.exec(value)
  if (match) return `${SPATIAL_LABELS[match[1]]} x${match[2]}`
  if (value === 'vae1' || value === 'vae2') return 'VAE upscaling'
  return value
}

/** "RIFE x2" from "rife2", "DLSS FG x3" from "dlssg*3". */
export function interpolationLabel(value: string | undefined): string {
  if (!value) return ''
  const match = /^(rife|dlssg\*)([2-6])$/.exec(value)
  if (match) return `${match[1] === 'rife' ? 'RIFE' : 'DLSS FG'} x${match[2]}`
  return value
}

/** Short GPU name: "NVIDIA A100-SXM4-80GB" -> "A100-SXM4-80GB". */
export function gpuLabel(hardware: GenerationProvenance['hardware']): string {
  if (!hardware?.gpu) return ''
  const name = hardware.gpu.replace(/^NVIDIA\s+(GeForce\s+)?/i, '').trim()
  return hardware.gpu_count && hardware.gpu_count > 1 ? `${hardware.gpu_count}x ${name}` : name
}

/** Turbo/distillation in a few words, or '' when the job ran at full steps. */
export function accelerationLabel(acceleration: GenerationProvenance['acceleration']): string {
  if (!acceleration) return ''
  const steps = isNumber(acceleration.steps) ? `${acceleration.steps}-step` : ''
  if (acceleration.turbo) {
    const preset = acceleration.turbo_preset_label || acceleration.turbo_preset || 'Turbo'
    return steps && !preset.toLowerCase().includes('step') ? `${preset} ${steps}` : preset
  }
  if (acceleration.fused_turbo) return `Fused Turbo${steps ? ` ${steps}` : ''}`
  if (acceleration.baked_turbo) return `Baked Turbo${steps ? ` ${steps}` : ''}`
  return steps ? `${acceleration.steps} steps` : ''
}

export function totalSeconds(p: GenerationProvenance | null | undefined): number | undefined {
  const timing = p?.timing
  if (!timing) return undefined
  return timing.total_seconds ?? timing.worker_seconds ?? timing.generation_seconds
}

/** Chip parts, most identifying first: model · quantization · acceleration · GPU · time · cost. */
export function provenanceSummaryParts(p: GenerationProvenance | null | undefined): string[] {
  if (!p) return []
  const parts = [
    p.model?.name || p.model?.type || '',
    p.runtime?.transformer_quantization || '',
    accelerationLabel(p.acceleration),
    upsamplingLabel(p.postprocessing?.spatial_upsampling),
    gpuLabel(p.hardware),
    formatSeconds(totalSeconds(p)),
    formatUsd(p.cost?.estimated_usd),
  ]
  return parts.filter(Boolean)
}

export function provenanceSummary(p: GenerationProvenance | null | undefined): string {
  return provenanceSummaryParts(p).join(' · ')
}

export interface ProvenanceRow {
  label: string
  value: string
}

function lorasText(loras: ProvenanceLora[] | undefined): string {
  return (loras || [])
    .map(lora => lora.multiplier !== undefined ? `${lora.name} (${lora.multiplier}x)` : lora.name)
    .join(', ')
}

/** Every known fact as label/value rows, for the hover title and the expanded view. */
export function provenanceRows(p: GenerationProvenance | null | undefined): ProvenanceRow[] {
  if (!p) return []
  const rows: ProvenanceRow[] = []
  const add = (label: string, value: string | undefined | null | false) => {
    if (value) rows.push({ label, value })
  }
  const model = p.model || {}
  add('Model', model.name && model.type && model.type !== model.name ? `${model.name} (${model.type})` : model.name || model.type)
  add('Checkpoint', model.files?.transformer)
  const runtime = p.runtime || {}
  add('Transformer', [
    runtime.transformer_quantization,
    runtime.transformer_dtype,
    !runtime.transformer_quantization && runtime.configured_quantization
      ? `configured ${runtime.configured_quantization}` : '',
  ].filter(Boolean).join(' · '))
  add('Text encoder', [runtime.text_encoder, runtime.text_encoder_quantization].filter(Boolean).join(' · '))
  add('VAE', [runtime.video_vae, runtime.video_vae_precision, runtime.vae_precision].filter(Boolean).join(' · '))
  add('Attention', attentionLabel(runtime.attention))
  if (runtime.compile !== undefined) add('Compile', runtime.compile ? 'on' : 'off')
  add('Engine', [runtime.inference_backend, runtime.memory_profile !== undefined ? `profile ${runtime.memory_profile}` : '']
    .filter(Boolean).join(' · '))
  const overrides = Object.entries(runtime.precision_overrides || {})
  add('Overrides', overrides.map(([key, value]) => `${key.replace(/_/g, ' ')}=${value}`).join(', '))
  add('LoRAs', lorasText(p.loras))
  const acceleration = p.acceleration || {}
  add('Acceleration', [
    accelerationLabel(acceleration),
    acceleration.step_cache
      ? `${STEP_CACHE_LABELS[acceleration.step_cache] ?? acceleration.step_cache}${
        isNumber(acceleration.step_cache_multiplier) ? ` ${acceleration.step_cache_multiplier}` : ''}`
      : '',
    acceleration.sparse_attention ? `${attentionLabel(acceleration.sparse_attention)} sparse attention` : '',
  ].filter(Boolean).join(' · '))
  const post = p.postprocessing || {}
  add('Postprocess', [
    upsamplingLabel(post.spatial_upsampling),
    interpolationLabel(post.temporal_upsampling),
    isNumber(post.film_grain_intensity) ? `film grain ${post.film_grain_intensity}` : '',
  ].filter(Boolean).join(' · '))
  const hardware = p.hardware || {}
  add('GPU', [
    gpuLabel(hardware),
    isNumber(hardware.vram_gb) ? `${hardware.vram_gb} GB` : '',
    hardware.location === 'remote' ? 'remote' : hardware.location === 'local' ? 'local' : '',
  ].filter(Boolean).join(' · '))
  add('Host', [
    hardware.backend, hardware.region, hardware.instance_type, hardware.spot ? 'spot' : '',
    isNumber(hardware.price_per_hour) ? `$${hardware.price_per_hour.toFixed(2)}/h` : '',
  ].filter(Boolean).join(' · '))
  const timing = p.timing || {}
  add('Render time', formatSeconds(totalSeconds(p)))
  if (isNumber(timing.generation_seconds) && timing.generation_seconds !== totalSeconds(p)) {
    add('Generation', formatSeconds(timing.generation_seconds))
  }
  if (isNumber(timing.remote_wait_seconds)) add('Worker wait', formatSeconds(timing.remote_wait_seconds))
  if (isNumber(timing.remote_work_seconds)) add('Worker time', formatSeconds(timing.remote_work_seconds))
  add('Breakdown', (Object.keys(PHASE_LABELS) as ProvenancePhase[])
    .filter(phase => isNumber(timing.phases?.[phase]))
    .map(phase => `${PHASE_LABELS[phase]} ${formatSeconds(timing.phases![phase])}`)
    .join(' · '))
  if (p.cost && isNumber(p.cost.estimated_usd)) add('Est. cost', `${formatUsd(p.cost.estimated_usd)} (GPU time × hourly price)`)
  return rows
}

/** Multi-line hover text. */
export function provenanceTitle(p: GenerationProvenance | null | undefined): string {
  return provenanceRows(p).map(row => `${row.label}: ${row.value}`).join('\n')
}

export function hasProvenance(p: GenerationProvenance | null | undefined): p is GenerationProvenance {
  return provenanceSummaryParts(p).length > 0
}
