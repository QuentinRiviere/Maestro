/**
 * Post-analysis notes on a Director run: targets, canvas anchoring, badges.
 *
 * A note targets the run, a scene, a shot, a pipeline stage or a canvas node
 * (`{kind, ref}`), and may remember the exact canvas node it was written on
 * (`target.node`). On the canvas each note is anchored to the node that
 * currently shows its target; when that node is folded away (collapsed scene
 * or stage column) the note falls back to the closest visible node, so a
 * badge never disappears. Pure and unit-tested.
 */
import type { DagNode } from './directorDag.ts'
import type { SceneGroup } from './directorRun.ts'

export const ANNOTATION_CATEGORIES = [
  'storyboard', 'image', 'video_motion', 'identity', 'continuity', 'audio_dialogue', 'prompt', 'edit_timing', 'other',
] as const

export type AnnotationCategory = typeof ANNOTATION_CATEGORIES[number]

export const CATEGORY_LABEL: Record<AnnotationCategory, string> = {
  storyboard: 'Storyboard',
  image: 'Image',
  video_motion: 'Video / motion',
  identity: 'Identity',
  continuity: 'Continuity',
  audio_dialogue: 'Audio / dialogue',
  prompt: 'Prompt',
  edit_timing: 'Edit / timing',
  other: 'Other',
}

export type AnnotationSeverity = 1 | 2 | 3

export const SEVERITY_LABEL: Record<AnnotationSeverity, string> = { 1: 'Minor', 2: 'Notable', 3: 'Blocking' }

export const MAX_ANNOTATION_TEXT = 4000

export type AnnotationTargetKind = 'run' | 'scene' | 'shot' | 'stage' | 'node'

export interface AnnotationTarget {
  kind: AnnotationTargetKind
  /** Shot index, scene key, stage slug or node id; empty for the run. */
  ref: string
  /** Canvas node the note was written on, when it was written on the canvas. */
  node?: string
}

export interface RunAnnotation {
  id: string
  target: AnnotationTarget
  category: AnnotationCategory
  severity: AnnotationSeverity
  text: string
  timecode_sec: number | null
  created_at: number
  updated_at: number
  resolved: boolean
}

export interface AnnotationDraft {
  target: AnnotationTarget
  category: AnnotationCategory
  severity: AnnotationSeverity
  text: string
  timecode_sec: number | null
}

export type AnnotationPatch = Partial<Omit<AnnotationDraft, 'target'>> & { resolved?: boolean }

/** Stage slugs, indexed like `DAG_COLUMNS`. */
export const STAGE_REFS = [
  'inputs', 'planning', 'prompts', 'start_images', 'video_renders', 'shot_clips', 'final_edit',
] as const

const STAGE_LABEL: Record<string, string> = {
  inputs: 'Inputs', planning: 'Planning', prompts: 'Prompts', start_images: 'Start images',
  video_renders: 'Video renders', shot_clips: 'Shot clips', final_edit: 'Final edit', storyboard: 'Storyboard',
}

/** Pseudo node id of one storyboard grid cell. */
export function storyboardCellId(shot: number): string {
  return `storyboard:${shot}`
}

/** Real canvas node that hosts an anchor id (storyboard cells live in the storyboard node). */
export function hostNodeId(anchorId: string): string {
  return anchorId.startsWith('storyboard:') ? 'storyboard' : anchorId
}

const shotTarget = (shot: number, node: string): AnnotationTarget => ({ kind: 'shot', ref: String(shot), node })

/** What a note written on this canvas node is about. `cellShot` picks a storyboard cell. */
export function targetForNode(node: DagNode, cellShot?: number): AnnotationTarget {
  switch (node.kind) {
    case 'column': {
      const column = Number(node.id.replace(/^column-/, ''))
      return { kind: 'stage', ref: STAGE_REFS[column] ?? node.id, node: node.id }
    }
    case 'frame':
    case 'summary':
      return { kind: 'scene', ref: node.toggleKey || node.id, node: node.id }
    case 'scene':
      return { kind: 'scene', ref: node.id, node: node.id }
    case 'brief':
      return { kind: 'stage', ref: 'inputs', node: node.id }
    case 'planning':
      return { kind: 'stage', ref: 'planning', node: node.id }
    case 'final':
      return { kind: 'stage', ref: 'final_edit', node: node.id }
    case 'storyboard':
      return cellShot != null
        ? shotTarget(cellShot, storyboardCellId(cellShot))
        : { kind: 'stage', ref: 'storyboard', node: node.id }
    case 'prompt':
    case 'image':
    case 'clip':
      return node.shot != null ? shotTarget(node.shot, node.id) : { kind: 'node', ref: node.id, node: node.id }
    case 'render':
      return node.members && node.members.length === 1
        ? shotTarget(node.members[0], node.id)
        : { kind: 'node', ref: node.id, node: node.id }
    default:
      return { kind: 'node', ref: node.id, node: node.id }
  }
}

/** Whether a note on this node may carry a timecode inside a clip. */
export function nodeTakesTimecode(node: DagNode): boolean {
  return node.kind === 'clip' || node.kind === 'final' || node.kind === 'render'
}

/** Every id a note can be anchored to: visible nodes plus storyboard cells. */
export function anchorIds(nodes: DagNode[]): Set<string> {
  const ids = new Set<string>()
  for (const node of nodes) {
    ids.add(node.id)
    if (node.kind === 'storyboard' && !node.collapsed) {
      for (const cell of node.cells || []) ids.add(storyboardCellId(cell.shot))
    }
  }
  return ids
}

/** Scene key of every shot. */
export function sceneOfShot(scenes: SceneGroup[]): Map<number, string> {
  const map = new Map<number, string>()
  for (const scene of scenes) for (const shot of scene.shots) map.set(shot.index, scene.key)
  return map
}

/** The visible node that shows this note's target, or null (run notes, unknown targets). */
export function anchorNodeId(
  target: AnnotationTarget, visible: ReadonlySet<string>, scenes: ReadonlyMap<number, string>,
): string | null {
  const first = (...ids: Array<string | null | undefined>) => ids.find(id => !!id && visible.has(id)) ?? null
  if (target.node && visible.has(target.node)) return target.node
  switch (target.kind) {
    case 'shot': {
      const shot = Number(target.ref)
      const scene = scenes.get(shot)
      return first(
        `clip-${shot}`, `image-${shot}`, `prompt-${shot}`,
        scene ? `summary-${scene}` : null, storyboardCellId(shot), 'storyboard',
      )
    }
    case 'scene':
      return first(target.ref, `summary-${target.ref}`, `frame-${target.ref}`)
    case 'stage': {
      if (target.ref === 'final_edit') return first('final', `column-${STAGE_REFS.indexOf('final_edit')}`)
      if (target.ref === 'storyboard') return first('storyboard')
      const column = STAGE_REFS.indexOf(target.ref as typeof STAGE_REFS[number])
      return column >= 0 ? first(`column-${column}`) : null
    }
    case 'node':
      return first(target.ref)
    default:
      return null
  }
}

export interface NodeBadge {
  open: number
  resolved: number
  /** Highest severity among open notes (0 when all are resolved). */
  maxSeverity: 0 | AnnotationSeverity
}

/** Notes grouped by the anchor id that shows them on the canvas. */
export function annotationsByAnchor(
  annotations: RunAnnotation[], nodes: DagNode[], scenes: SceneGroup[],
): Map<string, RunAnnotation[]> {
  const visible = anchorIds(nodes)
  const sceneMap = sceneOfShot(scenes)
  const groups = new Map<string, RunAnnotation[]>()
  for (const annotation of annotations) {
    const anchor = anchorNodeId(annotation.target, visible, sceneMap)
    if (!anchor) continue
    const list = groups.get(anchor)
    if (list) list.push(annotation)
    else groups.set(anchor, [annotation])
  }
  return groups
}

/** Badge counts per anchor id. */
export function badgesByAnchor(groups: ReadonlyMap<string, RunAnnotation[]>): Map<string, NodeBadge> {
  const badges = new Map<string, NodeBadge>()
  for (const [anchor, list] of groups) {
    let open = 0
    let resolved = 0
    let maxSeverity: NodeBadge['maxSeverity'] = 0
    for (const annotation of list) {
      if (annotation.resolved) {
        resolved += 1
        continue
      }
      open += 1
      if (annotation.severity > maxSeverity) maxSeverity = annotation.severity
    }
    badges.set(anchor, { open, resolved, maxSeverity })
  }
  return badges
}

/** Open notes first, then most severe, then newest. */
export function sortAnnotations(list: RunAnnotation[]): RunAnnotation[] {
  return [...list].sort((a, b) => (
    Number(a.resolved) - Number(b.resolved)
    || b.severity - a.severity
    || b.created_at - a.created_at
  ))
}

export function filterAnnotations(
  list: RunAnnotation[], filter: { category: AnnotationCategory | 'all'; showResolved: boolean },
): RunAnnotation[] {
  return sortAnnotations(list.filter(annotation => (
    (filter.showResolved || !annotation.resolved)
    && (filter.category === 'all' || annotation.category === filter.category)
  )))
}

/** Notes about one shot, whichever node they were written on. */
export function annotationsForShot(list: RunAnnotation[], shot: number): RunAnnotation[] {
  return sortAnnotations(list.filter(a => a.target.kind === 'shot' && a.target.ref === String(shot)))
}

const NODE_LABEL: Record<string, string> = { prompt: 'prompt', image: 'start image', clip: 'clip', render: 'render' }

/** Human label of a note's target ("Shot 3 · start image", "Scene 2", "Stage · Final edit"). */
export function describeTarget(target: AnnotationTarget, scenes?: SceneGroup[]): string {
  switch (target.kind) {
    case 'run':
      return 'Whole run'
    case 'shot': {
      const label = `Shot ${Number(target.ref) + 1}`
      const node = target.node || ''
      const kind = node.startsWith('storyboard:') ? 'storyboard' : node.split('-')[0]
      if (kind === 'storyboard') return `${label} · storyboard`
      return NODE_LABEL[kind] ? `${label} · ${NODE_LABEL[kind]}` : label
    }
    case 'scene': {
      const scene = scenes?.find(s => s.key === target.ref)
      return scene?.title || target.ref.replace(/^scene-/, 'Scene ')
    }
    case 'stage':
      return `Stage · ${STAGE_LABEL[target.ref] || target.ref}`
    default:
      return target.ref
  }
}

/**
 * Parse a timecode typed by the user: seconds ("12.5") or "m:ss" ("1:05.2").
 * Empty text means no timecode (null); unreadable text returns undefined.
 */
export function parseTimecode(text: string): number | null | undefined {
  const clean = text.trim()
  if (!clean) return null
  const match = clean.match(/^(?:(\d+):)?(\d+(?:\.\d+)?)$/)
  if (!match) return undefined
  const minutes = match[1] ? Number(match[1]) : 0
  const seconds = Number(match[2])
  if (match[1] && seconds >= 60) return undefined
  return Math.round((minutes * 60 + seconds) * 1000) / 1000
}

/** "1:05.2"-style label of a timecode in seconds. */
export function formatNoteTimecode(sec: number): string {
  const tenths = Math.max(0, Math.round(sec * 10))
  const minutes = Math.floor(tenths / 600)
  const rest = tenths - minutes * 600
  const seconds = rest % 10 === 0 ? String(rest / 10).padStart(2, '0') : (rest / 10).toFixed(1).padStart(4, '0')
  return `${minutes}:${seconds}`
}
