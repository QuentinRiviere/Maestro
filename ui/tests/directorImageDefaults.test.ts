import { test } from 'node:test'
import assert from 'node:assert/strict'
import { DIRECTOR_DEFAULT_IMAGE_MODEL, directorImageParams } from '../src/lib/directorImageDefaults.ts'

test('the Director defaults to Qwen-Image 2.1 base with Viggle Turbo v0.2.1 and the KV cache', () => {
  assert.equal(DIRECTOR_DEFAULT_IMAGE_MODEL, 'qwen_image_21_7B')
  assert.deepEqual(directorImageParams('qwen_image_21_7B', { num_inference_steps: 40, guidance_scale: 4, resolution: '1280x720' }), {
    num_inference_steps: 6,
    guidance_scale: 1,
    resolution: '1280x720',
    sample_solver: 'viggle_v021',
    custom_settings: { qwen21_kv_cache: 'Enabled' },
  })
})

test('an explicit sampler, cache choice or other model is kept', () => {
  const euler = { num_inference_steps: 40, guidance_scale: 4, sample_solver: 'default' }
  assert.deepEqual(directorImageParams('qwen_image_21_7B', euler), euler)
  assert.deepEqual(
    directorImageParams('qwen_image_21_7B', { custom_settings: { qwen21_kv_cache: 'Disabled' } }).custom_settings,
    { qwen21_kv_cache: 'Disabled' },
  )
  const klein = { num_inference_steps: 4, guidance_scale: 1 }
  assert.deepEqual(directorImageParams('flux2_klein_9b', klein), klein)
})
