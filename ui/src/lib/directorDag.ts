/**
 * Deterministic left-to-right layout of a Director run as a flow graph.
 *
 * Columns follow the pipeline stages; every shot owns one row, and the rows
 * of one scene sit inside a scene frame. H3 sequences render several shots
 * in one generation, so their render node spans the member rows. The output
 * is plain data (positions in pixels); the canvas component maps it to
 * React Flow nodes. Pure and unit-tested.
 *
 * Sections can be collapsed: a collapsed scene folds its rows into one
 * summary row, and a collapsed stage column (prompts, video renders) shrinks
 * to a thin strip while edges skip over it. Collapse keys live in one set so
 * the canvas can keep them across live re-layouts.
 */
import {
  formatSeconds, shotTimestamps, type RenderUnit, type RunShot, type SceneGroup, type StageStatus,
} from './directorRun.ts'
import type { StoryboardDrafts } from '../types'

export type DagNodeKind =
  | 'column' | 'frame' | 'brief' | 'character' | 'location' | 'scene' | 'planning'
  | 'prompt' | 'image' | 'render' | 'clip' | 'final' | 'cost' | 'summary' | 'storyboard'

/** One start frame of the storyboard grid. */
export interface StoryboardCell {
  shot: number
  media: string | null
  /** Start of the shot in the cut, in seconds. */
  startSec: number
  status: StageStatus
}

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
  /** Collapse key this node toggles (scene frames and summaries, stage columns). */
  toggleKey?: string
  /** Whether the section this node heads is collapsed. */
  collapsed?: boolean
  /** Start frames shown as a strip on a collapsed scene summary. */
  thumbs?: string[]
  /** Storyboard grid cells, in shot order. */
  cells?: StoryboardCell[]
  /** Storyboard grid column count. */
  gridColumns?: number
  /** Best-of-N draft candidates of a start-image node. */
  drafts?: StoryboardDrafts | null
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
  /** Collapsed sections: scene keys, and `stageKey(...)` for stage columns. */
  collapsed?: ReadonlySet<string>
  /** CSS aspect ratio of the frames ("9 / 16"); sizes the storyboard grid. */
  aspect?: string
  /** Measured clip lengths by shot index, when known; planned durations otherwise. */
  clipDurations?: ReadonlyArray<number | null | undefined>
}

/** Collapse key of the storyboard node. */
export const STORYBOARD_KEY = 'storyboard'
export const STORYBOARD_WIDTH = 360
const STORYBOARD_HEADER = 30
const STORYBOARD_BORDER = 2

/** Parse a "W / H" or "W:H" aspect ratio; 9:16 when unreadable. */
export function parseAspect(aspect: string | null | undefined): { w: number; h: number } {
  const match = String(aspect || '').match(/^\s*(\d+(?:\.\d+)?)\s*[/:]\s*(\d+(?:\.\d+)?)\s*$/)
  const w = match ? Number(match[1]) : 9
  const h = match ? Number(match[2]) : 16
  return w > 0 && h > 0 ? { w, h } : { w: 9, h: 16 }
}

/**
 * Storyboard grid geometry for a fixed node width: 6 columns of portrait
 * frames, 5 of square-ish ones, 3 of landscape ones; zero gap.
 */
export function storyboardGrid(aspect: string | null | undefined, count: number, collapsed = false) {
  const { w, h } = parseAspect(aspect)
  const ratio = w / h
  const columns = ratio < 0.8 ? 6 : ratio <= 1.25 ? 5 : 3
  const cellWidth = (STORYBOARD_WIDTH - 2 * STORYBOARD_BORDER) / columns
  const cellHeight = cellWidth * h / w
  const rows = Math.ceil(Math.max(0, count) / columns)
  const body = collapsed ? 0 : Math.ceil(rows * cellHeight)
  return { columns, cellWidth, cellHeight, rows, height: 2 * STORYBOARD_BORDER + STORYBOARD_HEADER + body }
}

export const DAG_COLUMNS = [
  'Inputs', 'Planning', 'Prompts', 'Start images', 'Video renders', 'Shot clips', 'Final edit',
] as const

export type CollapsibleStage = 'prompts' | 'renders'

/** Stage columns that can be collapsed, by column index. */
export const COLLAPSIBLE_STAGES: Readonly<Partial<Record<number, CollapsibleStage>>> = { 2: 'prompts', 4: 'renders' }

export function stageKey(stage: CollapsibleStage): string {
  return `stage:${stage}`
}

export const COL_WIDTH = 200
export const COL_GAP = 72
export const ROW_HEIGHT = 128
export const COLLAPSED_COL_WIDTH = 24
const COLLAPSED_COL_GAP = 36
export const SUMMARY_ROW_HEIGHT = 92
const MAX_SUMMARY_THUMBS = 8
const NODE_GAP = 14
const FRAME_HEADER = 30
const FRAME_PAD = 14
const SCENE_GAP = 28
const INPUT_HEIGHT = 64

/** X of a column in the default layout (no collapsed stage). */
export function columnX(column: number): number {
  return column * (COL_WIDTH + COL_GAP)
}

/** Column x positions and widths, with collapsed stages shrunk to a strip. */
export function columnLayout(collapsed: ReadonlySet<string>): { xs: number[]; widths: number[] } {
  const xs: number[] = []
  const widths: number[] = []
  let x = 0
  DAG_COLUMNS.forEach((_, column) => {
    const stage = COLLAPSIBLE_STAGES[column]
    const folded = !!stage && collapsed.has(stageKey(stage))
    xs.push(x)
    widths.push(folded ? COLLAPSED_COL_WIDTH : COL_WIDTH)
    x += folded ? COLLAPSED_COL_WIDTH + COLLAPSED_COL_GAP : COL_WIDTH + COL_GAP
  })
  return { xs, widths }
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

function plural(count: number, word: string): string {
  return `${count} ${word}${count === 1 ? '' : 's'}`
}

export function buildDirectorDag(input: DagInput): { nodes: DagNode[]; edges: DagEdge[] } {
  const nodes: DagNode[] = []
  const edges: DagEdge[] = []
  const statusOf = new Map<string, StageStatus>()
  const add = (node: DagNode) => { nodes.push(node); statusOf.set(node.id, node.status) }
  const linked = new Set<string>()
  const link = (source: string, target: string) => {
    const id = `${source}->${target}`
    if (linked.has(id)) return
    linked.add(id)
    edges.push({ id, source, target, animated: statusOf.get(target) === 'running' })
  }
  const failedRun = ['failed', 'crashed', 'cancelled'].includes(input.runStatus)
  const collapsed = input.collapsed ?? new Set<string>()
  const { xs, widths } = columnLayout(collapsed)
  const promptsHidden = collapsed.has(stageKey('prompts'))
  const rendersHidden = collapsed.has(stageKey('renders'))

  DAG_COLUMNS.forEach((title, column) => {
    const stage = COLLAPSIBLE_STAGES[column]
    add({
      id: `column-${column}`, kind: 'column', x: xs[column], y: -56, width: widths[column], height: 28,
      status: 'pending', label: title,
      ...(stage ? { toggleKey: stageKey(stage), collapsed: collapsed.has(stageKey(stage)) } : {}),
    })
  })

  // Shot rows, scene by scene. A collapsed scene keeps a single summary row.
  const rowY = new Map<number, number>()
  const sceneSpans: Array<{ scene: SceneGroup; top: number; height: number; folded: boolean }> = []
  let cursor = 0
  for (const scene of input.scenes) {
    const top = cursor
    const folded = collapsed.has(scene.key)
    if (!folded) scene.shots.forEach((shot, i) => rowY.set(shot.index, top + FRAME_HEADER + i * ROW_HEIGHT))
    const height = FRAME_HEADER + (folded ? SUMMARY_ROW_HEIGHT : scene.shots.length * ROW_HEIGHT) + FRAME_PAD
    sceneSpans.push({ scene, top, height, folded })
    cursor = top + height + SCENE_GAP
  }
  const contentHeight = Math.max(cursor - SCENE_GAP, ROW_HEIGHT)

  // Scene frames sit behind the per-shot columns (Planning to Shot clips).
  const frameLeft = xs[1] - FRAME_PAD
  const frameRight = xs[5] + widths[5] + FRAME_PAD
  for (const { scene, top, height, folded } of sceneSpans) {
    add({
      id: `frame-${scene.key}`, kind: 'frame', x: frameLeft, y: top, width: frameRight - frameLeft, height,
      status: aggregate(scene.shots.map(s => s.videoStatus)),
      label: scene.title,
      detail: scene.group && scene.location ? scene.group : undefined,
      toggleKey: scene.key, collapsed: folded,
    })
  }

  // Inputs: the brief, then the cast.
  let inputY = 0
  add({
    id: 'brief', kind: 'brief', x: xs[0], y: inputY, width: COL_WIDTH, height: INPUT_HEIGHT + 16,
    status: 'done', label: 'Brief', detail: input.brief,
  })
  inputY += INPUT_HEIGHT + 16 + NODE_GAP
  input.characters.forEach((character, index) => {
    add({
      id: `character-${index}`, kind: 'character', x: xs[0], y: inputY, width: COL_WIDTH, height: INPUT_HEIGHT,
      status: character.image ? 'done' : 'pending', label: character.name || `Character ${index + 1}`,
      media: character.image || null, castIndex: index, detail: 'Character',
    })
    inputY += INPUT_HEIGHT + NODE_GAP
  })
  input.locations.forEach((location, index) => {
    add({
      id: `location-${index}`, kind: 'location', x: xs[0], y: inputY, width: COL_WIDTH, height: INPUT_HEIGHT,
      status: location.image ? 'done' : 'pending', label: location.name || `Location ${index + 1}`,
      media: location.image || null, castIndex: index, detail: 'Location',
    })
    inputY += INPUT_HEIGHT + NODE_GAP
  })

  if (input.shots.length === 0) {
    const planning = input.phase === 'planning' || input.phase === 'casting' || input.phase === 'resuming'
    add({
      id: 'planning', kind: 'planning', x: xs[1], y: 0, width: COL_WIDTH, height: 96,
      status: planning ? 'running' : failedRun ? 'failed' : 'pending',
      label: planning ? 'Planning the storyboard…' : 'No storyboard yet',
    })
    link('brief', 'planning')
    input.characters.forEach((_, i) => link(`character-${i}`, 'planning'))
    input.locations.forEach((_, i) => link(`location-${i}`, 'planning'))
    return { nodes, edges }
  }

  // Planning: one node per scene, centred on its rows. A collapsed scene is
  // one summary node across every per-shot column instead.
  const sceneNodeOf = new Map<number, string>()
  const folded = new Set<number>()
  for (const span of sceneSpans) {
    const { scene, top, height } = span
    let id = scene.key
    if (span.folded) {
      id = `summary-${scene.key}`
      const total = scene.shots.reduce((sum, shot) => sum + (shot.durationSec || 0), 0)
      const done = scene.shots.filter(s => s.videoStatus === 'done').length
      const failed = scene.shots.filter(s => s.videoStatus === 'failed').length
      const running = scene.shots.filter(s => s.videoStatus === 'running' || s.imageStatus === 'running').length
      add({
        id, kind: 'summary', x: xs[1], y: top + FRAME_HEADER,
        width: xs[5] + widths[5] - xs[1], height: SUMMARY_ROW_HEIGHT - NODE_GAP,
        status: aggregate(scene.shots.map(s => s.videoStatus)),
        label: scene.title,
        detail: [
          plural(scene.shots.length, 'shot'),
          total > 0 ? formatSeconds(total) : null,
          `${done}/${scene.shots.length} rendered`,
          running ? `${running} running` : null,
          failed ? `${failed} missing` : null,
        ].filter(Boolean).join(' · '),
        thumbs: scene.shots.map(s => s.startImage).filter((m): m is string => !!m).slice(0, MAX_SUMMARY_THUMBS),
        shot: scene.shots[0]?.index, toggleKey: scene.key, collapsed: true,
      })
    } else {
      const nodeHeight = Math.min(96, height - FRAME_HEADER - FRAME_PAD)
      add({
        id, kind: 'scene', x: xs[1],
        y: top + FRAME_HEADER + (height - FRAME_HEADER - FRAME_PAD - nodeHeight) / 2,
        width: COL_WIDTH, height: nodeHeight, status: 'done', label: scene.title,
        detail: plural(scene.shots.length, 'shot'),
        shot: scene.shots[0]?.index,
      })
    }
    for (const shot of scene.shots) {
      sceneNodeOf.set(shot.index, id)
      if (span.folded) folded.add(shot.index)
    }
    link('brief', id)
    const sceneText = scene.shots
      .map(s => [s.subjects.join(' '), s.imagePrompt, s.dialogue.map(d => d.speaker).join(' ')].join(' '))
      .join(' ')
    input.characters.forEach((character, i) => {
      if (mentions(sceneText, character.name)) link(`character-${i}`, id)
    })
    input.locations.forEach((location, i) => {
      if (scene.shots.some(s => sameLocation(s.location, location.name))) link(`location-${i}`, id)
    })
  }

  const nodeHeight = ROW_HEIGHT - NODE_GAP
  const visibleShots = input.shots.filter(shot => !folded.has(shot.index))
  const sceneSource = (index: number) => sceneNodeOf.get(index) || 'brief'
  const promptSource = (index: number) => promptsHidden ? sceneSource(index) : `prompt-${index}`
  // The node feeding a shot's video render, skipping collapsed stages.
  const renderInput = (shot: RunShot) => shot.imageStatus !== 'skipped' ? `image-${shot.index}` : promptSource(shot.index)

  for (const shot of visibleShots) {
    const y = rowY.get(shot.index) ?? 0
    const label = `Shot ${shot.index + 1}`
    if (!promptsHidden) {
      add({
        id: `prompt-${shot.index}`, kind: 'prompt', x: xs[2], y, width: COL_WIDTH, height: nodeHeight,
        status: shot.videoPrompt || shot.imagePrompt ? 'done' : 'pending',
        label, detail: [shot.sceneType, shot.framing].filter(Boolean).join(' · ') || shot.action, shot: shot.index,
      })
      link(sceneSource(shot.index), `prompt-${shot.index}`)
    }
    if (shot.imageStatus !== 'skipped') {
      add({
        id: `image-${shot.index}`, kind: 'image', x: xs[3], y, width: COL_WIDTH, height: nodeHeight,
        status: shot.imageStatus, label, media: shot.startImage, shot: shot.index,
        detail: shot.judge?.verdict && shot.judge.verdict !== 'skipped' ? `judge: ${shot.judge.verdict}` : undefined,
        drafts: shot.drafts,
      })
      link(promptSource(shot.index), `image-${shot.index}`)
    }
  }

  // Video renders: a unit spans its visible member rows; a unit whose
  // members all sit in collapsed scenes folds into their summary.
  const clipSource = new Map<number, string>()
  if (!rendersHidden) {
    for (const unit of input.units) {
      const visible = unit.members.filter(m => input.shots[m] && !folded.has(m))
      if (visible.length === 0) continue
      const ys = visible.map(m => rowY.get(m) ?? 0)
      const top = Math.min(...ys)
      const bottom = Math.max(...ys) + nodeHeight
      const id = `render-${unit.id}`
      const members = unit.members.map(m => input.shots[m]).filter(Boolean)
      add({
        id, kind: 'render', x: xs[4], y: top, width: COL_WIDTH, height: bottom - top,
        status: aggregate(members.map(s => s.videoStatus)),
        label: unit.isSequence ? `Sequence ${unit.id.replace(/^seq_/, '')}` : `Shot ${unit.members[0] + 1}`,
        detail: unit.isSequence ? `H3 render of shots ${unit.members.map(m => m + 1).join(', ')}` : 'Single render',
        members: unit.members, shot: unit.members[0],
      })
      for (const member of visible) {
        link(renderInput(input.shots[member]), id)
        clipSource.set(member, id)
      }
    }
  }

  for (const shot of visibleShots) {
    const y = rowY.get(shot.index) ?? 0
    add({
      id: `clip-${shot.index}`, kind: 'clip', x: xs[5], y, width: COL_WIDTH, height: nodeHeight,
      status: shot.videoStatus, label: `Shot ${shot.index + 1}`, media: shot.video,
      detail: shot.sequence ? `split from ${shot.sequence.id}` : undefined, shot: shot.index,
    })
    link(clipSource.get(shot.index) ?? renderInput(shot), `clip-${shot.index}`)
  }

  const finalHeight = 150
  const finalY = Math.max(0, contentHeight / 2 - finalHeight / 2 - 50)
  const finalStatus: StageStatus = input.finalFile
    ? 'done'
    : input.phase === 'post_processing' ? 'running' : failedRun ? 'failed' : 'pending'
  add({
    id: 'final', kind: 'final', x: xs[6], y: finalY, width: COL_WIDTH, height: finalHeight,
    status: finalStatus, label: 'Final edit', media: input.finalFile,
    detail: `${input.shots.filter(s => s.videoStatus === 'done').length}/${input.shots.length} shots`,
  })
  for (const shot of input.shots) {
    link(folded.has(shot.index) ? sceneSource(shot.index) : `clip-${shot.index}`, 'final')
  }
  const costY = finalY + finalHeight + NODE_GAP * 2
  const costHeight = 56
  add({
    id: 'cost', kind: 'cost', x: xs[6], y: costY, width: COL_WIDTH, height: costHeight,
    status: input.costUsd != null ? 'done' : 'pending', label: 'Cost',
    detail: input.costUsd != null ? `$${input.costUsd.toFixed(2)}` : 'not priced yet',
  })

  // Storyboard: every start frame of the cut as one dense grid, under the
  // final edit it previews.
  const starts = shotTimestamps(input.shots, input.clipDurations)
  const boardCollapsed = collapsed.has(STORYBOARD_KEY)
  const grid = storyboardGrid(input.aspect, input.shots.length, boardCollapsed)
  const withImage = input.shots.filter(s => !!s.startImage).length
  add({
    id: 'storyboard', kind: 'storyboard', x: xs[6], y: costY + costHeight + NODE_GAP * 2,
    width: STORYBOARD_WIDTH, height: grid.height,
    status: aggregate(input.shots.map(s => s.imageStatus)),
    label: 'Storyboard', detail: `${withImage}/${input.shots.length} frames`,
    gridColumns: grid.columns, toggleKey: STORYBOARD_KEY, collapsed: boardCollapsed,
    cells: input.shots.map((shot, i) => ({
      shot: shot.index, media: shot.startImage, startSec: starts[i], status: shot.imageStatus,
    })),
  })
  return { nodes, edges }
}
