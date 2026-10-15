import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  compareRuns, formatRelativeTime, mergeRunPages, runActivityTime, runDisplayTime, runListItemFromState,
  runPageSearch, runShortLabel, runTimeTooltip, toTimestamp,
} from '../src/lib/directorRunList.ts'
import type { PipelineListItem, SavedPipelineState } from '../src/types/index.ts'

function run(id: string, fields: Partial<PipelineListItem> = {}): PipelineListItem {
  return {
    id, status: 'completed', pipeline_type: 'music_video', created_at: null, clip_count: 0, output_count: 0,
    scene_description: '', workspace: 'default', ...fields,
  }
}

test('timestamps are coerced to positive seconds', () => {
  assert.equal(toTimestamp(1000), 1000)
  assert.equal(toTimestamp('1500.5'), 1500.5)
  assert.equal(toTimestamp(1_700_000_000_000), 1_700_000_000)
  assert.equal(toTimestamp('1970-01-01T00:16:40Z'), 1000)
  for (const bad of [null, undefined, '', 'soon', 0, -1, Number.NaN, {}]) assert.equal(toTimestamp(bad), null)
})

test('runs sort by latest activity, newest first, ties by id', () => {
  const clones = [
    run('b', { created_at: 1000, updated_at: 1000 }),
    run('source', { created_at: 1000, completed_at: 1100 }),
    run('a', { created_at: 1000, updated_at: 1000 }),
    run('clone', { created_at: 1000, updated_at: 5000 }),
    run('broken', { created_at: null }),
  ]
  assert.deepEqual([...clones].sort(compareRuns).map(r => r.id), ['clone', 'source', 'a', 'b', 'broken'])
  assert.equal(runActivityTime(run('x', { created_at: 10, completed_at: 30, updated_at: 20 })), 30)
})

test('merging pages dedupes by id, keeps the fresher row and re-sorts', () => {
  const first = [run('r3', { updated_at: 300 }), run('r2', { updated_at: 200, status: 'running' })]
  const second = [run('r2', { updated_at: 400, status: 'completed' }), run('r1', { updated_at: 100 })]
  const merged = mergeRunPages(first, second)
  assert.deepEqual(merged.map(r => r.id), ['r2', 'r3', 'r1'])
  assert.equal(merged[0].status, 'completed')
  assert.deepEqual(mergeRunPages([], []), [])
})

test('page query strings omit empty values', () => {
  assert.equal(runPageSearch({}), '')
  assert.equal(runPageSearch({ limit: 20, cursor: 'abc', q: '  sea  ', status: 'running,paused' }),
    '?limit=20&cursor=abc&q=sea&status=running%2Cpaused')
  assert.equal(runPageSearch({ limit: 20, cursor: null, q: '   ' }), '?limit=20')
})

test('display time prefers finish, then a later update, then start', () => {
  assert.deepEqual(runDisplayTime({ created_at: 1000, completed_at: 2000, updated_at: 2001 }), { ts: 2000, kind: 'finished' })
  assert.deepEqual(runDisplayTime({ created_at: 1000, completed_at: 2000, updated_at: 9000 }), { ts: 9000, kind: 'updated' })
  assert.deepEqual(runDisplayTime({ created_at: 1000, updated_at: 1030 }), { ts: 1000, kind: 'started' })
  assert.deepEqual(runDisplayTime({ created_at: 1000, updated_at: 5000 }), { ts: 5000, kind: 'updated' })
  assert.deepEqual(runDisplayTime({ created_at: null, updated_at: 5000 }), { ts: 5000, kind: 'updated' })
  assert.equal(runDisplayTime({ created_at: null }), null)
  const tooltip = runTimeTooltip({ created_at: 1000, completed_at: 2000, updated_at: 3000 })
  assert.match(tooltip, /^Started .+\nFinished .+\nUpdated .+$/)
})

test('relative times', () => {
  const now = 1_000_000 * 1000
  assert.equal(formatRelativeTime(null, now), '—')
  assert.equal(formatRelativeTime(1_000_000 - 10, now), 'just now')
  assert.equal(formatRelativeTime(1_000_000 - 5 * 60, now), '5 min ago')
  assert.equal(formatRelativeTime(1_000_000 - 3 * 3600, now), '3 h ago')
  assert.equal(formatRelativeTime(1_000_000 - 30 * 3600, now), 'yesterday')
  assert.equal(formatRelativeTime(1_000_000 - 4 * 86400, now), '4 d ago')
  assert.doesNotMatch(formatRelativeTime(1_000_000 - 40 * 86400, now), /ago/)
})

test('short labels and list items built from a loaded run', () => {
  assert.equal(runShortLabel(run('p1', { scene_description: '  A  lighthouse\nkeeper ' })), 'A lighthouse keeper')
  assert.equal(runShortLabel(run('p1')), 'music video · p1')
  assert.equal(runShortLabel(run('p1', { scene_description: 'x'.repeat(100) }), 10), `${'x'.repeat(9)}…`)

  const state = {
    pipeline_id: 'deep/link', status: 'completed', pipeline_type: 'short_film', created_at: 1000, completed_at: 2000,
    scene_description: 'Deep linked', workspace: 'ws', clips: [{}, {}], output_files: ['a.mp4'], repair: null,
  } as unknown as SavedPipelineState
  const item = runListItemFromState(state)
  assert.equal(item.id, 'deep/link')
  assert.equal(item.clip_count, 2)
  assert.equal(item.thumbnail_url, '/api/v1/director/pipelines/deep%2Flink/thumbnail')
  const known = run('deep/link', { updated_at: 3000, status: 'running' })
  assert.equal(runListItemFromState(state, known).updated_at, 3000)
  assert.equal(runListItemFromState(state, known).status, 'completed')
})
