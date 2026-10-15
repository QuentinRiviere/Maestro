import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  buildRunShots, dialogueLines, finalEditFile, groupScenes, renderUnits, shortAction,
} from '../src/lib/directorRun.ts'
import { buildDirectorDag, columnX } from '../src/lib/directorDag.ts'
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

test('buildDirectorDag shows a planning node before the storyboard exists', () => {
  const { nodes, edges } = buildDirectorDag({
    shots: [], scenes: [], units: [], characters: [{ name: 'Mara' }], locations: [], brief: 'b',
    finalFile: null, runStatus: 'running', phase: 'planning', costUsd: null,
  })
  assert.equal(nodes.find(n => n.id === 'planning')!.status, 'running')
  assert.ok(edges.some(e => e.source === 'character-0' && e.target === 'planning' && e.animated))
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
