import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  accelerationLabel, formatSeconds, formatUsd, hasProvenance, interpolationLabel, provenanceRows,
  provenanceSummary, provenanceTitle, upsamplingLabel, type GenerationProvenance,
} from '../src/lib/provenance.ts'

const REMOTE_H3: GenerationProvenance = {
  version: 1,
  model: {
    type: 'minimax_h3_fl2va_full',
    name: 'MiniMax H3',
    files: { transformer: 'h3_full_int8_convrot.safetensors' },
  },
  runtime: {
    transformer_quantization: 'INT8',
    text_encoder: 'nvfp4_awq',
    attention: 'sage2',
    compile: false,
    precision_overrides: { transformer_quantization: 'int8' },
  },
  loras: [{ name: 'style.safetensors', multiplier: 0.8 }],
  acceleration: { steps: 8, turbo: true, turbo_preset: 'alibaba-pai-fl2va-pdd-8step', turbo_preset_label: 'PDD 8-step' },
  postprocessing: { spatial_upsampling: 'flashvsr1.5', temporal_upsampling: 'rife2' },
  hardware: {
    location: 'remote', gpu: 'NVIDIA A100 80GB PCIe', vram_gb: 79.2,
    backend: 'runpod', region: 'EU-RO-1', price_per_hour: 1.8,
  },
  timing: {
    total_seconds: 210, worker_seconds: 86, remote_wait_seconds: 120, remote_work_seconds: 86,
    phases: { model_load: 41, denoise: 30, vae_decode: 6 },
  },
  cost: { estimated_usd: 0.043, price_per_hour: 1.8 },
}

test('summary chip names model, quantization, acceleration, upscale, GPU, time and cost', () => {
  assert.equal(
    provenanceSummary(REMOTE_H3),
    'MiniMax H3 · INT8 · PDD 8-step · FlashVSR x1.5 · A100 80GB PCIe · 3 m 30 s · $0.043',
  )
})

test('details list every known fact and the phase breakdown', () => {
  const rows = Object.fromEntries(provenanceRows(REMOTE_H3).map(row => [row.label, row.value]))
  assert.equal(rows.Model, 'MiniMax H3 (minimax_h3_fl2va_full)')
  assert.equal(rows.Checkpoint, 'h3_full_int8_convrot.safetensors')
  assert.equal(rows.Attention, 'Sage 2')
  assert.equal(rows.Compile, 'off')
  assert.equal(rows.LoRAs, 'style.safetensors (0.8x)')
  assert.equal(rows.Postprocess, 'FlashVSR x1.5 · RIFE x2')
  assert.equal(rows.GPU, 'A100 80GB PCIe · 79.2 GB · remote')
  assert.equal(rows.Host, 'runpod · EU-RO-1 · $1.80/h')
  assert.equal(rows['Worker wait'], '2 m 00 s')
  assert.equal(rows.Breakdown, 'Model load 41 s · Denoise 30 s · VAE decode 6 s')
  assert.match(rows['Est. cost'], /^\$0\.043/)
  assert.match(provenanceTitle(REMOTE_H3), /^Model: MiniMax H3/)
})

test('missing fields are simply left out', () => {
  const minimal: GenerationProvenance = { model: { type: 'flux' }, hardware: { location: 'local' } }
  assert.equal(provenanceSummary(minimal), 'flux')
  assert.deepEqual(provenanceRows(minimal), [{ label: 'Model', value: 'flux' }, { label: 'GPU', value: 'local' }])
  assert.equal(hasProvenance(null), false)
  assert.equal(hasProvenance({}), false)
  assert.equal(hasProvenance(minimal), true)
})

test('labels and formatting', () => {
  assert.equal(accelerationLabel({ steps: 4, fused_turbo: true }), 'Fused Turbo 4-step')
  assert.equal(accelerationLabel({ steps: 30 }), '30 steps')
  assert.equal(accelerationLabel({ steps: 8, turbo: true, turbo_preset: 'lightx2v' }), 'lightx2v 8-step')
  assert.equal(upsamplingLabel('dlss5*2'), 'DLSS 5 x2')
  assert.equal(upsamplingLabel('custom'), 'custom')
  assert.equal(interpolationLabel('dlssg*3'), 'DLSS FG x3')
  assert.equal(formatSeconds(86.4), '1 m 26 s')
  assert.equal(formatSeconds(42), '42 s')
  assert.equal(formatSeconds(3720), '1 h 02 m')
  assert.equal(formatUsd(0.0004), '<$0.001')
  assert.equal(formatUsd(2.5), '$2.50')
})
