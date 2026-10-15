import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  buildRunShots, dialogueLines, finalEditFile, formatTimecode, groupScenes, renderUnits, shortAction, shotTimestamps,
} from '../src/lib/directorRun.ts'
import {
  buildDirectorDag, COL_WIDTH, COLLAPSED_COL_WIDTH, columnX, stageKey, STORYBOARD_KEY, STORYBOARD_WIDTH,
  storyboardGrid, SUMMARY_ROW_HEIGHT,
} from '../src/lib/directorDag.ts'
import { loadCollapsed, saveCollapsed } from '../src/lib/directorCanvasPrefs.ts'
import { formatDirectorHash, parseDirectorHash } from '../src/lib/directorRoute.ts'

function clip(index: number, extra: Record<string, unknown> = {}) {
  return {
    index,
    planned_clip: null,
    image_prompt: `image ${index}`,
    video_prompt: `Shot ${index} walks in. Then more.`,
    keyframe_prompts: [],
    window_prompts: [],
    window_count: 1,
    image_prompt_pre_polish: null,
    video_prompt_pre_polish: null,
    window_prompts_pre_polish: null,
    keyframe_prompts_pre_polish: null,
    start_image_filename: null,
    keyframe_filenames: [],
    video_filename: null,
    tag: null,
    image_gen_time_sec: null,
    video_gen_time_sec: null,
    ...extra,
  }
}

function saved(clips: ReturnType<typeof clip>[], extra: Record<string, unknown> = {}) {
  return {
    version: 1, pipeline_id: 'p1', created_at: 1, completed_at: null, status: 'completed',
    pipeline_type: 'short_film_story', scene_description: 'A heist', reference_image_path: null,
    auto_mode: true, seamless: false, image_model: 'img', video_model: 'vid', llm_log: null,
    clips, output_files: [], total_time_sec: null, ...extra,
  } as never
}

const sequence = (members: number[], index: number) => ({
  id: `seq_${String(members[0] + 1).padStart(2, '0')}`, index, size: members.length, members,
  duration_sec: 2.5, offset_sec: 0,
})

test('shortAction keeps the first sentence and trims long text', () => {
  assert.equal(shortAction('He runs. She waits.'), 'He runs.')
  assert.equal(shortAction('x'.repeat(200), 20).length, 20)
  assert.equal(shortAction(''), '')
})

test('dialogueLines resolves speaker names from the registry', () => {
  const lines = dialogueLines({
    _director_dialogue_beats: [{ speaker_id: 'S1', spoken_text: 'Hello.' }, { spoken_text: '' }],
    _director_speaker_registry: { S1: { name: 'Mara' } },
  })
  assert.deepEqual(lines, [{ speaker: 'Mara', text: 'Hello.' }])
})

test('buildRunShots reads scene metadata, judge verdicts and statuses', () => {
  const state = saved([
    clip(0, { start_image_filename: 'a.png', video_filename: 'a.mp4', _director_continuity_group: 'kitchen',
      _director_location: 'Kitchen', _director_scene_type: 'dialogue', _director_camera_framing: 'close-up',
      _director_duration_sec: 3.2 }),
    clip(1, { start_image_filename: 'b.png', _director_continuity_group: 'kitchen', _director_location: 'Kitchen' }),
  ])
  const shots = buildRunShots(state, {
    status: 'completed', recovered_from_disk: true,
    storyboard_judge: { shots: [{ index: 0, verdict: 'pass' }, { index: 1, verdict: 'flagged', issues: ['extra hand'] }] },
  })
  assert.equal(shots.length, 2)
  assert.equal(shots[0].sceneType, 'dialogue')
  assert.equal(shots[0].framing, 'close-up')
  assert.equal(shots[0].durationSec, 3.2)
  assert.equal(shots[0].imageStatus, 'done')
  assert.equal(shots[0].videoStatus, 'done')
  assert.equal(shots[0].action, 'Shot 0 walks in.')
  assert.equal(shots[1].imageStatus, 'flagged')
  assert.equal(shots[1].videoStatus, 'failed')
  assert.deepEqual(shots[1].judge?.issues, ['extra hand'])
})

test('live plans override the saved file and mark the active shot', () => {
  const state = saved([clip(0), clip(1)], { status: 'running' })
  const shots = buildRunShots(state, {
    status: 'running', phase: 'generating_images',
    clip_plans: [{ image_prompt: 'live 0', video_prompt: 'v' }, { image_prompt: 'live 1', video_prompt: 'v' },
      { image_prompt: 'live 2', video_prompt: 'v' }],
    clip_images: ['first.png', '', ''],
  })
  assert.equal(shots.length, 3)
  assert.equal(shots[0].imagePrompt, 'live 0')
  assert.equal(shots[0].startImage, 'first.png')
  assert.equal(shots[1].imageStatus, 'running')
  assert.equal(shots[2].imageStatus, 'pending')
})

test('a clip published before the batch ends shows as a provisional render', () => {
  const state = saved([
    clip(0, { video_filename: 'shot_1.mp4', video_provisional: true }),
    clip(1),
    clip(2),
  ], { status: 'running' })
  const shots = buildRunShots(state, {
    status: 'running', phase: 'generating_video', progress: { current_clip: 2 },
    clip_plans: [{}, {}, {}],
  })
  assert.equal(shots[0].video, 'shot_1.mp4')
  assert.equal(shots[0].videoProvisional, true)
  assert.equal(shots[1].videoProvisional, false)
  assert.deepEqual(shots.map(s => s.videoStatus), ['done', 'running', 'pending'])
})

test('a running H3 sequence marks every member shot as rendering', () => {
  const state = saved([
    clip(0, { _director_sequence: sequence([0, 1], 0) }),
    clip(1, { _director_sequence: sequence([0, 1], 1) }),
    clip(2),
  ], { status: 'running' })
  const shots = buildRunShots(state, {
    status: 'running', phase: 'generating_video', progress: { current_clip: 1 },
    clip_plans: [{ _director_sequence: sequence([0, 1], 0) }, { _director_sequence: sequence([0, 1], 1) }, {}],
  })
  assert.deepEqual(shots.map(s => s.videoStatus), ['running', 'running', 'pending'])
  assert.deepEqual(renderUnits(shots), [
    { id: 'seq_01', members: [0, 1], isSequence: true },
    { id: 'shot_3', members: [2], isSequence: false },
  ])
})

test('groupScenes splits on continuity group, then location, in story order', () => {
  const shots = buildRunShots(saved([
    clip(0, { _director_continuity_group: 'g1', _director_location: 'Roof' }),
    clip(1, { _director_continuity_group: 'g1' }),
    clip(2, { _director_location: 'Street' }),
    clip(3, { _director_location: 'street' }),
    clip(4, { _director_continuity_group: 'g1' }),
  ]))
  const scenes = groupScenes(shots)
  assert.deepEqual(scenes.map(s => s.shots.map(x => x.index)), [[0, 1], [2, 3], [4]])
  assert.equal(scenes[0].title, 'Scene 1 · Roof')
  assert.equal(scenes[2].title, 'Scene 3 · g1')
})

test('finalEditFile picks the joined multiclip output', () => {
  assert.equal(finalEditFile(['a.mp4', 'b.mp4', 'run_multiclip.mp4']), 'run_multiclip.mp4')
  assert.equal(finalEditFile(['a.mp4']), null)
})

test('buildDirectorDag lays stages out in columns with scene frames and sequence spans', () => {
  const shots = buildRunShots(saved([
    clip(0, { start_image_filename: 'a.png', video_filename: 'a.mp4', _director_continuity_group: 'g1',
      _director_location: 'Kitchen', _director_sequence: sequence([0, 1], 0),
      _director_subjects_on_screen: [{ name: 'Mara' }] }),
    clip(1, { start_image_filename: 'b.png', video_filename: 'b.mp4', _director_continuity_group: 'g1',
      _director_location: 'Kitchen', _director_sequence: sequence([0, 1], 1) }),
    clip(2, { start_image_filename: 'c.png', _director_continuity_group: 'g2', _director_location: 'Street' }),
  ], { output_files: ['a.mp4', 'b.mp4', 'x_multiclip.mp4'] }))
  const scenes = groupScenes(shots)
  const { nodes, edges } = buildDirectorDag({
    shots, scenes, units: renderUnits(shots),
    characters: [{ name: 'Mara', image: 'mara.png' }, { name: 'Nobody' }],
    locations: [{ name: 'kitchen' }],
    brief: 'A heist', finalFile: 'x_multiclip.mp4', runStatus: 'completed', phase: 'completed', costUsd: 1.5,
  })
  const byId = new Map(nodes.map(n => [n.id, n]))
  assert.equal(byId.get('prompt-0')!.x, columnX(2))
  assert.equal(byId.get('image-0')!.x, columnX(3))
  assert.equal(byId.get('clip-2')!.x, columnX(5))
  // The sequence render spans both member rows.
  const render = byId.get('render-seq_01')!
  assert.equal(render.y, byId.get('clip-0')!.y)
  assert.equal(render.y + render.height, byId.get('clip-1')!.y + byId.get('clip-1')!.height)
  assert.deepEqual(render.members, [0, 1])
  // Two scene frames; the second starts below the first.
  const frames = nodes.filter(n => n.kind === 'frame')
  assert.equal(frames.length, 2)
  assert.ok(frames[1].y >= frames[0].y + frames[0].height)
  // Data flow edges.
  const has = (s: string, t: string) => edges.some(e => e.source === s && e.target === t)
  assert.ok(has('character-0', 'scene-1'))
  assert.ok(!has('character-1', 'scene-1'))
  assert.ok(has('location-0', 'scene-1'))
  assert.ok(has('image-1', 'render-seq_01'))
  assert.ok(has('render-seq_01', 'clip-1'))
  assert.ok(has('clip-2', 'final'))
  assert.equal(byId.get('final')!.status, 'done')
  assert.equal(byId.get('clip-2')!.status, 'failed')
  assert.equal(byId.get('cost')!.detail, '$1.50')
  // Deterministic.
  const again = buildDirectorDag({
    shots, scenes, units: renderUnits(shots), characters: [{ name: 'Mara', image: 'mara.png' }, { name: 'Nobody' }],
    locations: [{ name: 'kitchen' }], brief: 'A heist', finalFile: 'x_multiclip.mp4', runStatus: 'completed',
    phase: 'completed', costUsd: 1.5,
  })
  assert.deepEqual(again.nodes, nodes)
})

function threeSceneRun() {
  const shots = buildRunShots(saved([
    clip(0, { start_image_filename: 'a.png', video_filename: 'a.mp4', _director_continuity_group: 'g1',
      _director_location: 'Kitchen', _director_sequence: sequence([0, 1], 0), _director_duration_sec: 2 }),
    clip(1, { start_image_filename: 'b.png', _director_continuity_group: 'g1', _director_location: 'Kitchen',
      _director_sequence: sequence([0, 1], 1), _director_duration_sec: 3 }),
    clip(2, { start_image_filename: 'c.png', _director_continuity_group: 'g2', _director_location: 'Street' }),
    clip(3, { _director_continuity_group: 'g3', _director_location: 'Roof' }),
  ]))
  const scenes = groupScenes(shots)
  const dag = (collapsed?: Set<string>) => buildDirectorDag({
    shots, scenes, units: renderUnits(shots), characters: [{ name: 'Mara' }], locations: [{ name: 'Kitchen' }],
    brief: 'b', finalFile: null, runStatus: 'completed', phase: 'completed', costUsd: null, collapsed,
  })
  return { shots, scenes, dag }
}

test('a collapsed scene folds into one summary row and the next scenes move up', () => {
  const { dag } = threeSceneRun()
  const open = dag()
  const closed = dag(new Set(['scene-1']))
  const byId = (nodes: ReturnType<typeof dag>['nodes']) => new Map(nodes.map(n => [n.id, n]))
  const a = byId(open.nodes)
  const b = byId(closed.nodes)
  // Scene 1's per-shot nodes are gone, replaced by a summary across the shot columns.
  for (const id of ['scene-1', 'prompt-0', 'image-1', 'render-seq_01', 'clip-0', 'clip-1']) assert.ok(!b.has(id), id)
  const summary = b.get('summary-scene-1')!
  assert.equal(summary.kind, 'summary')
  assert.equal(summary.x, columnX(1))
  assert.equal(summary.x + summary.width, columnX(5) + COL_WIDTH)
  assert.match(summary.detail!, /^2 shots · 5s · 1\/2 rendered · 1 missing$/)
  assert.deepEqual(summary.thumbs, ['a.png', 'b.png'])
  assert.equal(b.get('frame-scene-1')!.collapsed, true)
  assert.equal(b.get('frame-scene-1')!.toggleKey, 'scene-1')
  assert.equal(b.get('frame-scene-1')!.height, SUMMARY_ROW_HEIGHT + 30 + 14)
  // Following scenes reflow upward by the height saved; their rows stay intact.
  const saved = a.get('frame-scene-1')!.height - b.get('frame-scene-1')!.height
  assert.ok(saved > 0)
  assert.equal(b.get('frame-scene-2')!.y, a.get('frame-scene-2')!.y - saved)
  assert.equal(b.get('clip-3')!.y, a.get('clip-3')!.y - saved)
  // Edges reroute through the summary.
  const has = (s: string, t: string) => closed.edges.some(e => e.source === s && e.target === t)
  assert.ok(has('brief', 'summary-scene-1'))
  assert.ok(has('location-0', 'summary-scene-1'))
  assert.ok(has('summary-scene-1', 'final'))
  assert.ok(has('clip-2', 'final'))
  assert.ok(!has('clip-0', 'final'))
  // Every edge points at an existing node.
  for (const e of closed.edges) assert.ok(b.has(e.source) && b.has(e.target), e.id)
})

test('collapsed stage columns shrink and edges skip over them', () => {
  const { dag } = threeSceneRun()
  const { nodes, edges } = dag(new Set([stageKey('prompts'), stageKey('renders')]))
  const byId = new Map(nodes.map(n => [n.id, n]))
  assert.ok(!nodes.some(n => n.kind === 'prompt' || n.kind === 'render'))
  assert.equal(byId.get('column-2')!.collapsed, true)
  assert.equal(byId.get('column-2')!.width, COLLAPSED_COL_WIDTH)
  assert.equal(byId.get('column-3')!.collapsed, undefined)
  // Columns after a collapsed stage move left.
  assert.ok(byId.get('image-0')!.x < columnX(3))
  assert.ok(byId.get('clip-0')!.x < byId.get('final')!.x)
  const has = (s: string, t: string) => edges.some(e => e.source === s && e.target === t)
  assert.ok(has('scene-1', 'image-0'))
  assert.ok(has('image-0', 'clip-0'))
  assert.ok(has('scene-3', 'image-3'))
  for (const e of edges) assert.ok(byId.has(e.source) && byId.has(e.target), e.id)
  // Edge ids stay unique.
  assert.equal(new Set(edges.map(e => e.id)).size, edges.length)
})

test('a sequence split across a collapsed scene keeps only its visible rows', () => {
  const shots = buildRunShots(saved([
    clip(0, { _director_continuity_group: 'g1', _director_sequence: sequence([0, 1], 0) }),
    clip(1, { _director_continuity_group: 'g2', _director_sequence: sequence([0, 1], 1) }),
  ]))
  const scenes = groupScenes(shots)
  const { nodes, edges } = buildDirectorDag({
    shots, scenes, units: renderUnits(shots), characters: [], locations: [], brief: 'b', finalFile: null,
    runStatus: 'completed', phase: 'completed', costUsd: null, collapsed: new Set(['scene-1']),
  })
  const render = nodes.find(n => n.id === 'render-seq_01')!
  const clip1 = nodes.find(n => n.id === 'clip-1')!
  assert.equal(render.y, clip1.y)
  assert.equal(render.height, clip1.height)
  assert.ok(edges.some(e => e.source === 'render-seq_01' && e.target === 'clip-1'))
  assert.ok(!edges.some(e => e.target === 'clip-0'))
})

test('the storyboard node lists every start frame with its cut timestamp', () => {
  const { shots, dag } = threeSceneRun()
  const open = dag()
  const board = open.nodes.find(n => n.id === 'storyboard')!
  assert.equal(board.kind, 'storyboard')
  assert.equal(board.width, STORYBOARD_WIDTH)
  assert.equal(board.x, columnX(6))
  const cost = open.nodes.find(n => n.id === 'cost')!
  assert.ok(board.y >= cost.y + cost.height)
  assert.equal(board.gridColumns, 6)
  assert.equal(board.toggleKey, STORYBOARD_KEY)
  assert.equal(board.detail, '3/4 frames')
  // Shots 0 and 1 last 2 s and 3 s (planned); shot 2 has no duration.
  assert.deepEqual(board.cells!.map(c => [c.shot, c.media, c.startSec]), [
    [0, 'a.png', 0], [1, 'b.png', 2], [2, 'c.png', 5], [3, null, 5],
  ])
  assert.equal(board.height, storyboardGrid('9 / 16', shots.length).height)
  // No edges are drawn to the storyboard; it previews the cut.
  assert.ok(!open.edges.some(e => e.source === 'storyboard' || e.target === 'storyboard'))
  // Collapsing it keeps only the header; it is still there.
  const closed = dag(new Set([STORYBOARD_KEY])).nodes.find(n => n.id === 'storyboard')!
  assert.equal(closed.collapsed, true)
  assert.ok(closed.height < board.height)
})

test('storyboardGrid picks columns from the aspect ratio at a fixed width', () => {
  const portrait = storyboardGrid('9 / 16', 13)
  assert.equal(portrait.columns, 6)
  assert.equal(portrait.rows, 3)
  assert.ok(Math.abs(portrait.cellHeight - portrait.cellWidth * 16 / 9) < 1e-9)
  assert.equal(portrait.cellWidth * portrait.columns, STORYBOARD_WIDTH - 4)
  assert.equal(storyboardGrid('1:1', 1).columns, 5)
  assert.equal(storyboardGrid('16 / 9', 1).columns, 3)
  assert.equal(storyboardGrid('bogus', 1).columns, 6)
  assert.equal(storyboardGrid('16 / 9', 0).rows, 0)
  assert.equal(storyboardGrid('9 / 16', 13, true).height, storyboardGrid('9 / 16', 0).height)
})

test('canvas collapse prefs round-trip through storage and survive broken storage', () => {
  const data = new Map<string, string>()
  const storage = {
    getItem: (k: string) => data.get(k) ?? null,
    setItem: (k: string, v: string) => { data.set(k, v) },
    removeItem: (k: string) => { data.delete(k) },
  }
  saveCollapsed('run-1', new Set(['scene-2', 'stage:prompts']), storage)
  assert.deepEqual([...loadCollapsed('run-1', storage)].sort(), ['scene-2', 'stage:prompts'])
  assert.equal(loadCollapsed('run-2', storage).size, 0)
  saveCollapsed('run-1', new Set(), storage)
  assert.equal(data.size, 0)
  data.set('maestro.director.canvas.collapsed.run-3', '{not json')
  assert.equal(loadCollapsed('run-3', storage).size, 0)
  const broken = {
    getItem: () => { throw new Error('blocked') },
    setItem: () => { throw new Error('full') },
    removeItem: () => { throw new Error('blocked') },
  }
  assert.equal(loadCollapsed('run-1', broken).size, 0)
  assert.doesNotThrow(() => saveCollapsed('run-1', new Set(['x']), broken))
  assert.equal(loadCollapsed('run-1', null).size, 0)
})

test('buildDirectorDag shows a planning node before the storyboard exists', () => {
  const { nodes, edges } = buildDirectorDag({
    shots: [], scenes: [], units: [], characters: [{ name: 'Mara' }], locations: [], brief: 'b',
    finalFile: null, runStatus: 'running', phase: 'planning', costUsd: null,
  })
  assert.equal(nodes.find(n => n.id === 'planning')!.status, 'running')
  assert.ok(edges.some(e => e.source === 'character-0' && e.target === 'planning' && e.animated))
})

test('shotTimestamps accumulates planned durations and prefers measured clip lengths', () => {
  const shots = [{ durationSec: 3.2 }, { durationSec: null }, { durationSec: 5 }, { durationSec: 61 }, { durationSec: 2 }]
  assert.deepEqual(shotTimestamps(shots), [0, 3.2, 3.2, 8.2, 69.2])
  // A measured length replaces the plan; bogus measurements fall back to it.
  assert.deepEqual(shotTimestamps(shots, [4, 1.5, Number.NaN, -1]), [0, 4, 5.5, 10.5, 71.5])
  assert.deepEqual(shotTimestamps([]), [])
})

test('formatTimecode renders MM:SS', () => {
  assert.equal(formatTimecode(0), '00:00')
  assert.equal(formatTimecode(3.2), '00:03')
  assert.equal(formatTimecode(69.99), '01:09')
  assert.equal(formatTimecode(3725), '62:05')
  assert.equal(formatTimecode(Number.NaN), '00:00')
  assert.equal(formatTimecode(-4), '00:00')
})

test('director hash routes round-trip', () => {
  assert.deepEqual(parseDirectorHash('#/director'), { pid: null, view: 'runs' })
  assert.deepEqual(parseDirectorHash('#/director/abc'), { pid: 'abc', view: 'storyboard' })
  assert.deepEqual(parseDirectorHash('#/director/abc/canvas'), { pid: 'abc', view: 'canvas' })
  assert.deepEqual(parseDirectorHash('#/director/abc/bogus'), { pid: 'abc', view: 'storyboard' })
  assert.equal(parseDirectorHash('#/studio'), null)
  assert.equal(parseDirectorHash(''), null)
  assert.equal(formatDirectorHash({ pid: 'a b', view: 'details' }), '#/director/a%20b/details')
  assert.equal(formatDirectorHash({ pid: 'abc', view: 'storyboard' }), '#/director/abc')
  assert.equal(formatDirectorHash({ pid: null, view: 'canvas' }), '#/director')
  assert.deepEqual(parseDirectorHash(formatDirectorHash({ pid: 'a/b', view: 'canvas' })), { pid: 'a/b', view: 'canvas' })
})

test('live clip number falls back to the progress message', async () => {
  const { liveClipNumber } = await import('../src/lib/directorRun.ts')
  assert.equal(liveClipNumber({ current_clip: 3, message: 'Clip 5/10' }), 3)
  assert.equal(liveClipNumber({ message: 'Clip 4/10: Prompt 4/10 - Denoising | 18.0s' }), 4)
  assert.equal(liveClipNumber({ message: 'Review images' }), 0)
  assert.equal(liveClipNumber(null), 0)
})

test('keep range accepts only a valid [start, end] pair', async () => {
  const { keepRange } = await import('../src/lib/directorRun.ts')
  assert.deepEqual(keepRange([0, 2.5]), [0, 2.5])
  assert.equal(keepRange([2, 1]), null)
  assert.equal(keepRange(null), null)
  assert.equal(keepRange([0]), null)
})
