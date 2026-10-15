import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  anchorIds, anchorNodeId, annotationsByAnchor, annotationsForShot, badgesByAnchor, describeTarget,
  filterAnnotations, formatNoteTimecode, hostNodeId, nodeTakesTimecode, parseTimecode, sceneOfShot,
  storyboardCellId, targetForNode, type AnnotationTarget, type RunAnnotation,
} from '../src/lib/directorAnnotations.ts'
import type { DagNode } from '../src/lib/directorDag.ts'
import type { RunShot, SceneGroup } from '../src/lib/directorRun.ts'

function node(id: string, kind: DagNode['kind'], extra: Partial<DagNode> = {}): DagNode {
  return { id, kind, x: 0, y: 0, width: 10, height: 10, status: 'done', label: id, ...extra }
}

function scene(key: string, title: string, shots: number[]): SceneGroup {
  return {
    key, number: Number(key.replace('scene-', '')), title, location: '', group: '',
    shots: shots.map(index => ({ index }) as RunShot),
  }
}

let seq = 0
function note(target: AnnotationTarget, extra: Partial<RunAnnotation> = {}): RunAnnotation {
  seq += 1
  return {
    id: `n${seq}`, target, category: 'other', severity: 2, text: 'x', timecode_sec: null,
    created_at: seq, updated_at: seq, resolved: false, ...extra,
  }
}

const SCENES = [scene('scene-1', 'Scene 1 · Harbor', [0, 1]), scene('scene-2', 'Scene 2 · Attic', [2])]

test('targetForNode maps each canvas node to what a note on it is about', () => {
  assert.deepEqual(targetForNode(node('clip-3', 'clip', { shot: 3 })), { kind: 'shot', ref: '3', node: 'clip-3' })
  assert.deepEqual(targetForNode(node('image-0', 'image', { shot: 0 })), { kind: 'shot', ref: '0', node: 'image-0' })
  assert.deepEqual(targetForNode(node('scene-2', 'scene', { shot: 2 })), { kind: 'scene', ref: 'scene-2', node: 'scene-2' })
  assert.deepEqual(
    targetForNode(node('frame-scene-1', 'frame', { toggleKey: 'scene-1' })),
    { kind: 'scene', ref: 'scene-1', node: 'frame-scene-1' },
  )
  assert.deepEqual(
    targetForNode(node('summary-scene-1', 'summary', { toggleKey: 'scene-1', shot: 0 })),
    { kind: 'scene', ref: 'scene-1', node: 'summary-scene-1' },
  )
  assert.deepEqual(targetForNode(node('column-3', 'column')), { kind: 'stage', ref: 'start_images', node: 'column-3' })
  assert.deepEqual(targetForNode(node('final', 'final')), { kind: 'stage', ref: 'final_edit', node: 'final' })
  assert.deepEqual(targetForNode(node('brief', 'brief')), { kind: 'stage', ref: 'inputs', node: 'brief' })
  assert.deepEqual(targetForNode(node('storyboard', 'storyboard')), { kind: 'stage', ref: 'storyboard', node: 'storyboard' })
  assert.deepEqual(targetForNode(node('storyboard', 'storyboard'), 4), { kind: 'shot', ref: '4', node: 'storyboard:4' })
  assert.deepEqual(
    targetForNode(node('render-seq_1', 'render', { members: [1, 2], shot: 1 })),
    { kind: 'node', ref: 'render-seq_1', node: 'render-seq_1' },
  )
  assert.deepEqual(
    targetForNode(node('render-c2', 'render', { members: [2], shot: 2 })),
    { kind: 'shot', ref: '2', node: 'render-c2' },
  )
  assert.deepEqual(targetForNode(node('character-0', 'character')), { kind: 'node', ref: 'character-0', node: 'character-0' })
  assert.equal(nodeTakesTimecode(node('clip-0', 'clip')), true)
  assert.equal(nodeTakesTimecode(node('image-0', 'image')), false)
})

test('anchorNodeId prefers the written-on node, then falls back to whatever shows the target', () => {
  const nodes = [
    node('clip-0', 'clip', { shot: 0 }), node('image-0', 'image', { shot: 0 }), node('summary-scene-2', 'summary'),
    node('scene-1', 'scene'), node('column-2', 'column'), node('final', 'final'),
    node('storyboard', 'storyboard', { cells: [0, 1, 2].map(shot => ({ shot, media: null, startSec: 0, status: 'done' as const })) }),
  ]
  const visible = anchorIds(nodes)
  const scenes = sceneOfShot(SCENES)
  assert.ok(visible.has(storyboardCellId(2)))
  assert.equal(anchorNodeId({ kind: 'shot', ref: '0', node: 'image-0' }, visible, scenes), 'image-0')
  // Written on a prompt node that is now hidden (prompts column collapsed): the clip shows it.
  assert.equal(anchorNodeId({ kind: 'shot', ref: '0', node: 'prompt-0' }, visible, scenes), 'clip-0')
  // A shot inside a collapsed scene lands on the scene summary.
  assert.equal(anchorNodeId({ kind: 'shot', ref: '2', node: 'clip-2' }, visible, scenes), 'summary-scene-2')
  assert.equal(anchorNodeId({ kind: 'shot', ref: '1', node: 'storyboard:1' }, visible, scenes), 'storyboard:1')
  assert.equal(anchorNodeId({ kind: 'scene', ref: 'scene-1' }, visible, scenes), 'scene-1')
  assert.equal(anchorNodeId({ kind: 'scene', ref: 'scene-2' }, visible, scenes), 'summary-scene-2')
  assert.equal(anchorNodeId({ kind: 'stage', ref: 'prompts' }, visible, scenes), 'column-2')
  assert.equal(anchorNodeId({ kind: 'stage', ref: 'final_edit' }, visible, scenes), 'final')
  assert.equal(anchorNodeId({ kind: 'run', ref: '' }, visible, scenes), null)
  assert.equal(anchorNodeId({ kind: 'node', ref: 'render-seq_9' }, visible, scenes), null)
  assert.equal(hostNodeId('storyboard:1'), 'storyboard')
  assert.equal(hostNodeId('clip-0'), 'clip-0')
})

test('badges count open notes per anchor and take the color of the most severe open one', () => {
  const nodes = [node('clip-0', 'clip', { shot: 0 }), node('clip-1', 'clip', { shot: 1 }), node('scene-1', 'scene')]
  const annotations = [
    note({ kind: 'shot', ref: '0', node: 'clip-0' }, { severity: 1 }),
    note({ kind: 'shot', ref: '0' }, { severity: 3 }),
    note({ kind: 'shot', ref: '0', node: 'clip-0' }, { severity: 3, resolved: true }),
    note({ kind: 'shot', ref: '1', node: 'clip-1' }, { resolved: true }),
    note({ kind: 'run', ref: '' }),
  ]
  const groups = annotationsByAnchor(annotations, nodes, SCENES)
  assert.deepEqual([...groups.keys()].sort(), ['clip-0', 'clip-1'])
  const badges = badgesByAnchor(groups)
  assert.deepEqual(badges.get('clip-0'), { open: 2, resolved: 1, maxSeverity: 3 })
  assert.deepEqual(badges.get('clip-1'), { open: 0, resolved: 1, maxSeverity: 0 })
})

test('filtering keeps open notes first, most severe first, and hides resolved ones by default', () => {
  const a = note({ kind: 'run', ref: '' }, { severity: 1, category: 'image' })
  const b = note({ kind: 'run', ref: '' }, { severity: 3, category: 'prompt' })
  const c = note({ kind: 'run', ref: '' }, { severity: 3, category: 'image', resolved: true })
  const list = [a, b, c]
  assert.deepEqual(filterAnnotations(list, { category: 'all', showResolved: false }).map(n => n.id), [b.id, a.id])
  assert.deepEqual(filterAnnotations(list, { category: 'all', showResolved: true }).map(n => n.id), [b.id, a.id, c.id])
  assert.deepEqual(filterAnnotations(list, { category: 'image', showResolved: true }).map(n => n.id), [a.id, c.id])
  const shotNote = note({ kind: 'shot', ref: '2', node: 'storyboard:2' })
  assert.deepEqual(annotationsForShot([...list, shotNote], 2).map(n => n.id), [shotNote.id])
})

test('describeTarget names the target for lists and exports', () => {
  assert.equal(describeTarget({ kind: 'run', ref: '' }), 'Whole run')
  assert.equal(describeTarget({ kind: 'shot', ref: '2', node: 'image-2' }), 'Shot 3 · start image')
  assert.equal(describeTarget({ kind: 'shot', ref: '2', node: 'storyboard:2' }), 'Shot 3 · storyboard')
  assert.equal(describeTarget({ kind: 'shot', ref: '2' }), 'Shot 3')
  assert.equal(describeTarget({ kind: 'scene', ref: 'scene-2' }, SCENES), 'Scene 2 · Attic')
  assert.equal(describeTarget({ kind: 'scene', ref: 'scene-7' }), 'Scene 7')
  assert.equal(describeTarget({ kind: 'stage', ref: 'final_edit' }), 'Stage · Final edit')
})

test('timecodes parse from seconds or m:ss and format back', () => {
  assert.equal(parseTimecode(''), null)
  assert.equal(parseTimecode('12.5'), 12.5)
  assert.equal(parseTimecode('1:05'), 65)
  assert.equal(parseTimecode('1:05.25'), 65.25)
  assert.equal(parseTimecode('1:75'), undefined)
  assert.equal(parseTimecode('soon'), undefined)
  assert.equal(formatNoteTimecode(65), '1:05')
  assert.equal(formatNoteTimecode(65.25), '1:05.3')
  assert.equal(formatNoteTimecode(59.96), '1:00')
  assert.equal(formatNoteTimecode(2.5), '0:02.5')
})
