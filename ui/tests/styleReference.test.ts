import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  formatSeconds, formatShotMix, isAnalysing, pacingBars, parseSceneTags, rulesFromText, rulesToText,
} from '../src/lib/styleReference.ts'

test('scene tags are normalised like the server keeps them', () => {
  assert.deepEqual(parseSceneTags('Argument, Slow Romance;tension\nargument,, '), ['argument', 'slow_romance', 'tension'])
  assert.deepEqual(parseSceneTags(''), [])
})

test('rules round-trip through text, list markers dropped', () => {
  assert.deepEqual(rulesFromText('1. Hold the reaction.\n- Open on an insert.\n\n* Push in on a realisation.'),
    ['Hold the reaction.', 'Open on an insert.', 'Push in on a realisation.'])
  assert.equal(rulesToText(['A', 'B']), 'A\nB')
  assert.equal(rulesToText(null), '')
})

test('shot mix lists the largest shares first', () => {
  assert.equal(formatShotMix({ wide: 10, close_up: 55.4, medium: 34.6 }, 2), 'Close-up 55% · Medium 35%')
  assert.equal(formatShotMix(null), '')
})

test('status, durations and pacing bars', () => {
  assert.ok(isAnalysing('queued') && isAnalysing('analysing') && !isAnalysing('done'))
  assert.equal(formatSeconds(2.345), '2.3 s')
  assert.equal(formatSeconds(75), '1:15')
  assert.equal(formatSeconds(null), '–')
  assert.deepEqual(pacingBars([{ asl_sec: 2 }, { asl_sec: 4 }, { asl_sec: 0 }]), [1, 0.5])
  assert.deepEqual(pacingBars(undefined), [])
})
