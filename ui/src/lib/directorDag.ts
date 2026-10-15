/**
 * Deterministic left-to-right layout of a Director run as a flow graph.
 *
 * Columns follow the pipeline stages; every shot owns one row, and the rows
 * of one scene sit inside a scene frame. H3 sequences render several shots
 * in one generation, so their render node spans the member rows. The output
 * is plain data (positions in pixels); the canvas component maps it to
 * React Flow nodes. Pure and unit-tested.
 */
import type { RenderUnit, RunShot, SceneGroup, StageStatus } from './directorRun.ts'

export type DagNodeKind =
  | 'column' | 'frame' | 'brief' | 'character' | 'location' | 'scene' | 'planning'
  | 'prompt' | 'image' | 'render' | 'clip' | 'final' | 'cost'

export interface DagNode {
  id: string
  kind: DagNodeKind
  x: number
  y: number
  width: number
  height: number
  status: StageStatus
  label: string
  /** Shot index for per-shot nodes, so a click can open that shot. */
  shot?: number
  /** Member shots of a render unit. */
  members?: number[]
  /** Media to show as a thumbnail (output file name or absolute URL). */
  media?: string | null
  /** Short secondary text. */
  detail?: string
  /** Index into the cast list for character/location nodes. */
  castIndex?: number
}

export interface DagEdge {
  id: string
  source: string
  target: string
  animated: boolean
}

export interface CastRef {
  name: string
  image?: string | null
}

export interface DagInput {
  shots: RunShot[]
  scenes: SceneGroup[]
  units: RenderUnit[]
  characters: CastRef[]
  locations: CastRef[]
  brief: string
  finalFile: string | null
  runStatus: string
  phase: string
  costUsd: number | null
}

export const DAG_COLUMNS = [
  'Inputs', 'Planning', 'Prompts', 'Start images', 'Video renders', 'Shot clips', 'Final edit',
] as const

export const COL_WIDTH = 200
export const COL_GAP = 72
export const ROW_HEIGHT = 128
const NODE_GAP = 14
const FRAME_HEADER = 30
const FRAME_PAD = 14
const SCENE_GAP = 28
const INPUT_HEIGHT = 64

export function columnX(column: number): number {
  return column * (COL_WIDTH + COL_GAP)
}

function aggregate(statuses: StageStatus[]): StageStatus {
  if (statuses.length === 0) return 'pending'
  if (statuses.some(s => s === 'running')) return 'running'
  if (statuses.every(s => s === 'done' || s === 'skipped')) return 'done'
  if (statuses.some(s => s === 'failed')) return 'failed'
  if (statuses.some(s => s === 'flagged')) return 'flagged'
  return 'pending'
}

function mentions(haystack: string, name: string): boolean {
  const needle = name.trim().toLowerCase()
  return needle.length > 1 && haystack.toLowerCase().includes(needle)
}

function sameLocation(a: string, b: string): boolean {
  const x = a.trim().toLowerCase()
  const y = b.trim().toLowerCase()
  return !!x && !!y && (x === y || x.includes(y) || y.includes(x))
}

export function buildDirectorDag(input: DagInput): { nodes: DagNode[]; edges: DagEdge[] } {
  const nodes: DagNode[] = []
  const edges: DagEdge[] = []
  const statusOf = new Map<string, StageStatus>()
  const add = (node: DagNode) => { nodes.push(node); statusOf.set(node.id, node.status) }
  const link = (source: string, target: string) => {
    edges.push({ id: `${source}->${target}`, source, target, animated: statusOf.get(target) === 'running' })
  }
  const failedRun = ['failed', 'crashed', 'cancelled'].includes(input.runStatus)

  DAG_COLUMNS.forEach((title, column) => add({
    id: `column-${column}`, kind: 'column', x: columnX(column), y: -56, width: COL_WIDTH, height: 28,
    status: 'pending', label: title,
  }))

  // Shot rows, scene by scene.
  const rowY = new Map<number, number>()
  const sceneSpans: Array<{ scene: SceneGroup; top: number; height: number }> = []
  let cursor = 0
  for (const scene of input.scenes) {
    const top = cursor
    scene.shots.forEach((shot, i) => rowY.set(shot.index, top + FRAME_HEADER + i * ROW_HEIGHT))
    const height = FRAME_HEADER + scene.shots.length * ROW_HEIGHT + FRAME_PAD
    sceneSpans.push({ scene, top, height })
    cursor = top + height + SCENE_GAP
  }
  const contentHeight = Math.max(cursor - SCENE_GAP, ROW_HEIGHT)

  // Scene frames sit behind the per-shot columns (Planning to Shot clips).
  for (const { scene, top, height } of sceneSpans) {
    const left = columnX(1) - FRAME_PAD
    add({
      id: `frame-${scene.key}`, kind: 'frame', x: left, y: top,
      width: columnX(5) + COL_WIDTH + FRAME_PAD - left, height,
      status: aggregate(scene.shots.map(s => s.videoStatus)),
      label: scene.title,
      detail: scene.group && scene.location ? scene.group : undefined,
    })
  }

  // Inputs: the brief, then the cast.
  let inputY = 0
  add({
    id: 'brief', kind: 'brief', x: columnX(0), y: inputY, width: COL_WIDTH, height: INPUT_HEIGHT + 16,
    status: 'done', label: 'Brief', detail: input.brief,
  })
  inputY += INPUT_HEIGHT + 16 + NODE_GAP
  input.characters.forEach((character, index) => {
    add({
      id: `character-${index}`, kind: 'character', x: columnX(0), y: inputY, width: COL_WIDTH, height: INPUT_HEIGHT,
      status: character.image ? 'done' : 'pending', label: character.name || `Character ${index + 1}`,
      media: character.image || null, castIndex: index, detail: 'Character',
    })
    inputY += INPUT_HEIGHT + NODE_GAP
  })
  input.locations.forEach((location, index) => {
    add({
      id: `location-${index}`, kind: 'location', x: columnX(0), y: inputY, width: COL_WIDTH, height: INPUT_HEIGHT,
      status: location.image ? 'done' : 'pending', label: location.name || `Location ${index + 1}`,
      media: location.image || null, castIndex: index, detail: 'Location',
    })
    inputY += INPUT_HEIGHT + NODE_GAP
  })

  if (input.shots.length === 0) {
    const planning = input.phase === 'planning' || input.phase === 'casting' || input.phase === 'resuming'
    add({
      id: 'planning', kind: 'planning', x: columnX(1), y: 0, width: COL_WIDTH, height: 96,
      status: planning ? 'running' : failedRun ? 'failed' : 'pending',
      label: planning ? 'Planning the storyboard…' : 'No storyboard yet',
    })
    link('brief', 'planning')
    input.characters.forEach((_, i) => link(`character-${i}`, 'planning'))
    input.locations.forEach((_, i) => link(`location-${i}`, 'planning'))
    return { nodes, edges }
  }

  // Planning: one node per scene, centred on its rows.
  for (const { scene, top, height } of sceneSpans) {
    const nodeHeight = Math.min(96, height - FRAME_HEADER - FRAME_PAD)
    add({
      id: scene.key, kind: 'scene', x: columnX(1),
      y: top + FRAME_HEADER + (height - FRAME_HEADER - FRAME_PAD - nodeHeight) / 2,
      width: COL_WIDTH, height: nodeHeight, status: 'done', label: scene.title,
      detail: `${scene.shots.length} shot${scene.shots.length === 1 ? '' : 's'}`,
      shot: scene.shots[0]?.index,
    })
    link('brief', scene.key)
    const sceneText = scene.shots
      .map(s => [s.subjects.join(' '), s.imagePrompt, s.dialogue.map(d => d.speaker).join(' ')].join(' '))
      .join(' ')
    input.characters.forEach((character, i) => {
      if (mentions(sceneText, character.name)) link(`character-${i}`, scene.key)
    })
    input.locations.forEach((location, i) => {
      if (scene.shots.some(s => sameLocation(s.location, location.name))) link(`location-${i}`, scene.key)
    })
  }

  const sceneOf = new Map<number, string>()
  input.scenes.forEach(scene => scene.shots.forEach(shot => sceneOf.set(shot.index, scene.key)))
  const nodeHeight = ROW_HEIGHT - NODE_GAP

  for (const shot of input.shots) {
    const y = rowY.get(shot.index) ?? 0
    const label = `Shot ${shot.index + 1}`
    add({
      id: `prompt-${shot.index}`, kind: 'prompt', x: columnX(2), y, width: COL_WIDTH, height: nodeHeight,
      status: shot.videoPrompt || shot.imagePrompt ? 'done' : 'pending',
      label, detail: [shot.sceneType, shot.framing].filter(Boolean).join(' · ') || shot.action, shot: shot.index,
    })
    link(sceneOf.get(shot.index) || 'brief', `prompt-${shot.index}`)
    if (shot.imageStatus !== 'skipped') {
      add({
        id: `image-${shot.index}`, kind: 'image', x: columnX(3), y, width: COL_WIDTH, height: nodeHeight,
        status: shot.imageStatus, label, media: shot.startImage, shot: shot.index,
        detail: shot.judge?.verdict && shot.judge.verdict !== 'skipped' ? `judge: ${shot.judge.verdict}` : undefined,
      })
      link(`prompt-${shot.index}`, `image-${shot.index}`)
    }
  }

  for (const unit of input.units) {
    const first = unit.members[0]
    const ys = unit.members.map(m => rowY.get(m) ?? 0)
    const top = Math.min(...ys)
    const bottom = Math.max(...ys) + nodeHeight
    const id = `render-${unit.id}`
    const members = unit.members.map(m => input.shots[m]).filter(Boolean)
    add({
      id, kind: 'render', x: columnX(4), y: top, width: COL_WIDTH, height: bottom - top,
      status: aggregate(members.map(s => s.videoStatus)),
      label: unit.isSequence ? `Sequence ${unit.id.replace(/^seq_/, '')}` : `Shot ${first + 1}`,
      detail: unit.isSequence ? `H3 render of shots ${unit.members.map(m => m + 1).join(', ')}` : 'Single render',
      members: unit.members, shot: first,
    })
    for (const shot of members) {
      link(shot.imageStatus === 'skipped' ? `prompt-${shot.index}` : `image-${shot.index}`, id)
    }
  }

  for (const shot of input.shots) {
    const y = rowY.get(shot.index) ?? 0
    add({
      id: `clip-${shot.index}`, kind: 'clip', x: columnX(5), y, width: COL_WIDTH, height: nodeHeight,
      status: shot.videoStatus, label: `Shot ${shot.index + 1}`, media: shot.video,
      detail: shot.sequence ? `split from ${shot.sequence.id}` : undefined, shot: shot.index,
    })
  }
  for (const unit of input.units) {
    for (const member of unit.members) {
      if (input.shots[member]) link(`render-${unit.id}`, `clip-${member}`)
    }
  }

  const finalHeight = 150
  const finalY = Math.max(0, contentHeight / 2 - finalHeight / 2 - 50)
  const finalStatus: StageStatus = input.finalFile
    ? 'done'
    : input.phase === 'post_processing' ? 'running' : failedRun ? 'failed' : 'pending'
  add({
    id: 'final', kind: 'final', x: columnX(6), y: finalY, width: COL_WIDTH, height: finalHeight,
    status: finalStatus, label: 'Final edit', media: input.finalFile,
    detail: `${input.shots.filter(s => s.videoStatus === 'done').length}/${input.shots.length} shots`,
  })
  for (const shot of input.shots) link(`clip-${shot.index}`, 'final')
  add({
    id: 'cost', kind: 'cost', x: columnX(6), y: finalY + finalHeight + NODE_GAP * 2, width: COL_WIDTH, height: 56,
    status: input.costUsd != null ? 'done' : 'pending', label: 'Cost',
    detail: input.costUsd != null ? `$${input.costUsd.toFixed(2)}` : 'not priced yet',
  })
  return { nodes, edges }
}
