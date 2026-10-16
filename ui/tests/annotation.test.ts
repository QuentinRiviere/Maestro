import { test } from 'node:test'
import assert from 'node:assert/strict'
import { hasAnnotation, parseTags } from '../src/lib/annotation.ts'

test('parseTags trims, collapses spaces and drops case-insensitive duplicates', () => {
  assert.deepEqual(parseTags(' keep,  hand   morph , Keep,,redo '), ['keep', 'hand morph', 'redo'])
  assert.deepEqual(parseTags('  , ,'), [])
})

test('hasAnnotation needs a note or a tag', () => {
  assert.equal(hasAnnotation(null), false)
  assert.equal(hasAnnotation({ note: '  ', tags: [] }), false)
  assert.equal(hasAnnotation({ note: '', tags: ['redo'] }), true)
  assert.equal(hasAnnotation({ note: 'cup handle appears', tags: [] }), true)
})
