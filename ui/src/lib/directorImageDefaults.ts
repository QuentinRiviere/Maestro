/**
 * The Director's default image model and the sampler it runs with.
 *
 * Mirrors DIRECTOR_DEFAULT_IMAGE_MODEL / DIRECTOR_IMAGE_SAMPLERS in
 * app/services/director_pipeline.py, which applies the same sampler to a
 * project the backend starts. Qwen-Image 2.1 base runs with the Viggle Turbo
 * v0.2.1 profile (6 steps, CFG 1; the model handler adds its LoRA) and the KV
 * cache. An explicit model or sampler choice always wins.
 */

export const DIRECTOR_DEFAULT_IMAGE_MODEL = 'qwen_image_21_7B'

interface DirectorImageSampler {
  sample_solver: string
  num_inference_steps: number
  guidance_scale: number
  custom_settings: Record<string, string>
}

export const DIRECTOR_IMAGE_SAMPLERS: Readonly<Record<string, DirectorImageSampler>> = {
  qwen_image_21_7B: {
    sample_solver: 'viggle_v021',
    num_inference_steps: 6,
    guidance_scale: 1,
    custom_settings: { qwen21_kv_cache: 'Enabled' },
  },
}

/** ``params`` with the Director's sampler for ``modelType`` when no sampler was chosen. */
export function directorImageParams(
  modelType: string,
  params: Record<string, unknown>,
): Record<string, unknown> {
  const preset = DIRECTOR_IMAGE_SAMPLERS[modelType]
  if (!preset || String(params.sample_solver ?? '').trim()) return { ...params }
  const custom = params.custom_settings && typeof params.custom_settings === 'object'
    ? params.custom_settings as Record<string, unknown>
    : {}
  return {
    ...params,
    sample_solver: preset.sample_solver,
    num_inference_steps: preset.num_inference_steps,
    guidance_scale: preset.guidance_scale,
    custom_settings: { ...preset.custom_settings, ...custom },
  }
}
