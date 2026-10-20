import { test } from 'node:test'
import assert from 'node:assert/strict'
import { clipFrames, clipSeconds, formatSeconds, MAX_CLIP_SECONDS, normaliseWeights, videoFrames, videoWindows } from '../src/lib/persona.ts'

test('weights normalise to one, fall back to even', () => {
  assert.deepEqual(normaliseWeights(2, [3, 1]), [0.75, 0.25])
  assert.deepEqual(normaliseWeights(2, ['60', '40']), [0.6, 0.4])
  assert.deepEqual(normaliseWeights(2, [0, -1]), [0.5, 0.5])
  assert.deepEqual(normaliseWeights(3, [1]), [0.3333, 0.3333, 0.3333])
  assert.deepEqual(normaliseWeights(0, []), [])
})

test('clip frames follow the LTX window', () => {
  assert.equal(clipFrames(1), 25)
  assert.equal(clipFrames(6), 145)
  assert.equal(clipFrames(12), 289)
  assert.equal(clipFrames(40), 345)
  assert.equal(clipFrames(0), 17)
  assert.equal(clipSeconds(289), 12)
  assert.ok(MAX_CLIP_SECONDS > 14 && MAX_CLIP_SECONDS < 14.5)
  assert.equal(videoFrames(40), 961)
  assert.equal(videoFrames(500), videoFrames(120))
  assert.equal(videoWindows(289), 1)
  assert.equal(videoWindows(961), 4)
})

test('positions format as minutes and seconds', () => {
  assert.equal(formatSeconds(0), '0:00')
  assert.equal(formatSeconds(75.9), '1:15')
})
