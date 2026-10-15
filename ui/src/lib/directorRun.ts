/**
 * Pure view model of one Director run for the storyboard and canvas views.
 *
 * A run is seen through two payloads: the saved state on disk
 * (`GET /director/pipelines/{pid}`, written at phase boundaries) and the
 * live status (`GET /director/pipeline/{pid}`, refreshed every poll while
 * the worker runs). This module merges them into one list of shots, groups
 * the shots into scenes and into H3 render units, and derives per-stage
 * statuses. No React, no fetch: everything here is unit-tested.
 */
import type { SavedPipelineState } from '../types'

export type StageStatus = 'pending' | 'running' | 'done' | 'failed' | 'flagged' | 'skipped'

export type JudgeVerdict = 'pending' | 'skipped' | 'pass' | 'failed' | 'regenerated' | 'flagged'

export interface StoryboardJudgeShot {
  index: number
  verdict: JudgeVerdict
  attempts?: number
  issues?: string[]
  fix_hint?: string
  duplicate_of?: number | null
  image?: string
  cost_usd?: number
  error?: string
}

export interface StoryboardJudgeReport {
  status?: 'running' | 'completed' | 'cancelled' | string
  mode?: string
  model?: string | null
  cost_usd?: number
  shots?: StoryboardJudgeShot[]
}

/** The subset of the live status payload this view model reads. */
export interface LiveRunStatus {
  status?: string
  phase?: string
  progress?: {
    current?: number
    total?: number
    message?: string
    current_clip?: number
    total_clips?: number
  } | null
  clip_plans?: Array<Record<string, unknown>>
  planned_clips?: Array<Record<string, unknown> | null>
  clip_images?: string[]
  output_files?: string[]
  pause_reason?: string | null
  recovered_from_disk?: boolean
  storyboard_judge?: StoryboardJudgeReport | null
  cast?: { characters?: unknown[]; locations?: unknown[] } | null
  error?: string | null
}

export interface DialogueLine {
  speaker: string
  text: string
}

export interface SequenceInfo {
  id: string
  index: number
  size: number
  members: number[]
  durationSec: number | null
}

export interface RunShot {
  index: number
  imagePrompt: string
  videoPrompt: string
  windowPrompts: string[]
  startImage: string | null
  video: string | null
  videoStale: boolean
  sceneType: string
  framing: string
  location: string
  group: string
  strategy: string
  durationSec: number | null
  dialogue: DialogueLine[]
  action: string
  subjects: string[]
  sequence: SequenceInfo | null
  judge: StoryboardJudgeShot | null
  imageStatus: StageStatus
  videoStatus: StageStatus
  tag: 'good' | 'needs_work' | null
}

export interface SceneGroup {
  key: string
  number: number
  title: string
  location: string
  group: string
  shots: RunShot[]
}

export interface RenderUnit {
  id: string
  /** Shot indices rendered by this generation, in order. */
  members: number[]
  isSequence: boolean
}

type Dict = Record<string, unknown>

const ACTIVE_STATUSES = new Set(['running', 'paused', 'queued'])
const FAILED_STATUSES = new Set(['failed', 'crashed', 'cancelled'])
const IMAGE_PHASES = new Set(['generating_images'])
const VIDEO_PHASES = new Set(['preparing_video', 'generating_video'])

function asDict(value: unknown): Dict {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Dict : {}
}

function text(value: unknown): string {
  if (value == null) return ''
  if (typeof value === 'string') return value.trim()
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  return ''
}

function num(value: unknown): number | null {
  const n = typeof value === 'number' ? value : typeof value === 'string' ? Number(value) : NaN
  return Number.isFinite(n) && n > 0 ? n : null
}

function stringList(value: unknown): string[] {
  return Array.isArray(value) ? value.map(text).filter(Boolean) : []
}

/** First sentence of a prompt, trimmed to a storyboard caption. */
export function shortAction(prompt: string, max = 140): string {
  const clean = prompt.replace(/\s+/g, ' ').trim()
  if (!clean) return ''
  const sentence = clean.match(/^.*?[.!?](\s|$)/)?.[0]?.trim() || clean
  return sentence.length > max ? `${sentence.slice(0, max - 1).trimEnd()}…` : sentence
}

/** Speaker display names from the plan's speaker registry and on-screen subjects. */
function speakerNames(plan: Dict): Map<string, string> {
  const names = new Map<string, string>()
  const remember = (id: unknown, name: unknown) => {
    const key = text(id)
    const label = text(name)
    if (key && label && !names.has(key)) names.set(key, label)
  }
  const registry = plan._director_speaker_registry
  if (Array.isArray(registry)) {
    for (const entry of registry) {
      const e = asDict(entry)
      remember(e.speaker_id ?? e.id, e.name ?? e.display_name ?? e.label)
    }
  } else {
    for (const [id, entry] of Object.entries(asDict(registry))) {
      if (typeof entry === 'string') remember(id, entry)
      else {
        const e = asDict(entry)
        remember(id, e.name ?? e.display_name ?? e.label)
      }
    }
  }
  for (const subject of Array.isArray(plan._director_subjects_on_screen) ? plan._director_subjects_on_screen : []) {
    const s = asDict(subject)
    remember(s.speaker_id ?? s.subject_id ?? s.id, s.name ?? s.display_name ?? s.label)
  }
  return names
}

export function dialogueLines(plan: Dict): DialogueLine[] {
  const beats = Array.isArray(plan._director_dialogue_beats) ? plan._director_dialogue_beats : []
  const names = speakerNames(plan)
  return beats
    .map(beat => {
      const b = asDict(beat)
      const id = text(b.speaker_id ?? b.speaker)
      return { speaker: names.get(id) || id, text: text(b.spoken_text ?? b.text) }
    })
    .filter(line => line.text)
}

function subjectNames(plan: Dict): string[] {
  const list = Array.isArray(plan._director_subjects_on_screen) ? plan._director_subjects_on_screen : []
  return list
    .map(subject => (typeof subject === 'string' ? subject : text(asDict(subject).name ?? asDict(subject).display_name ?? asDict(subject).id)))
    .filter(Boolean)
}

function sequenceInfo(plan: Dict): SequenceInfo | null {
  const seq = asDict(plan._director_sequence)
  const id = text(seq.id)
  const members = Array.isArray(seq.members) ? seq.members.filter((m): m is number => typeof m === 'number') : []
  if (!id || members.length < 2) return null
  return {
    id,
    index: typeof seq.index === 'number' ? seq.index : 0,
    size: typeof seq.size === 'number' ? seq.size : members.length,
    members,
    durationSec: num(seq.duration_sec),
  }
}

function durationOf(plan: Dict, planned: Dict, sequence: SequenceInfo | null): number | null {
  const direct = num(plan._director_duration_sec) ?? num(plan._director_keep_sec) ?? sequence?.durationSec ?? num(planned.duration_sec)
  if (direct) return direct
  const start = typeof planned.start === 'number' ? planned.start : null
  const end = typeof planned.end === 'number' ? planned.end : null
  return start != null && end != null && end > start ? end - start : null
}

function isLiveRun(live: LiveRunStatus | null | undefined): boolean {
  return !!live && !live.recovered_from_disk && ACTIVE_STATUSES.has(text(live.status))
}

function imageStatusOf(
  hasImage: boolean, judge: StoryboardJudgeShot | null, needsImage: boolean, runStatus: string, active: boolean,
): StageStatus {
  if (!needsImage) return 'skipped'
  if (judge?.verdict === 'flagged') return 'flagged'
  if (hasImage) return 'done'
  if (active) return 'running'
  if (FAILED_STATUSES.has(runStatus) || runStatus === 'completed') return 'failed'
  return 'pending'
}

function videoStatusOf(hasVideo: boolean, stale: boolean, runStatus: string, active: boolean): StageStatus {
  if (hasVideo && !stale) return 'done'
  if (active) return 'running'
  if (FAILED_STATUSES.has(runStatus) || runStatus === 'completed') return 'failed'
  return 'pending'
}

/** Merge saved and live state into one ordered list of shots. */
export function buildRunShots(
  saved: SavedPipelineState | null | undefined,
  live?: LiveRunStatus | null,
): RunShot[] {
  const savedClips = saved?.clips || []
  const livePlans = isLiveRun(live) ? live!.clip_plans || [] : []
  const count = Math.max(savedClips.length, livePlans.length)
  const runStatus = isLiveRun(live) ? text(live!.status) : text(saved?.status || live?.status)
  const needsImage = !saved?.shot_image_policy || saved.shot_image_policy === 'generate'
  const judgeShots = (live?.storyboard_judge?.shots || [])
  const phase = isLiveRun(live) ? text(live!.phase) : ''
  const shots: RunShot[] = []

  for (let index = 0; index < count; index++) {
    const savedClip = asDict(savedClips[index])
    const livePlan = asDict(livePlans[index])
    // Live plans win while the worker runs: the saved file lags a phase behind.
    const plan: Dict = { ...savedClip, ...livePlan }
    const planned = asDict(live?.planned_clips?.[index] ?? savedClip.planned_clip)
    const startImage = text(isLiveRun(live) ? live!.clip_images?.[index] : '') || text(savedClip.start_image_filename) || null
    const video = text(savedClip.video_filename) || null
    const videoStale = savedClip.video_stale === true
    const sequence = sequenceInfo(plan)
    const judge = judgeShots.find(shot => shot?.index === index) || null
    const videoPrompt = text(plan.video_prompt)
    const sourcePrompt = text(plan._director_h3_source_prompt)
    shots.push({
      index,
      imagePrompt: text(plan.image_prompt),
      videoPrompt,
      windowPrompts: stringList(plan.window_prompts),
      startImage,
      video,
      videoStale,
      sceneType: text(plan._director_scene_type),
      framing: text(plan._director_camera_framing),
      location: text(plan._director_location),
      group: text(plan._director_continuity_group),
      strategy: text(plan._director_continuity_strategy),
      durationSec: durationOf(plan, planned, sequence),
      dialogue: dialogueLines(plan),
      action: shortAction(text(plan._director_opening_blocking) || sourcePrompt || videoPrompt || text(plan.image_prompt)),
      subjects: subjectNames(plan),
      sequence,
      judge,
      imageStatus: imageStatusOf(!!startImage, judge, needsImage, runStatus, false),
      videoStatus: videoStatusOf(!!video, videoStale, runStatus, false),
      tag: savedClip.tag === 'good' || savedClip.tag === 'needs_work' ? savedClip.tag : null,
    })
  }

  // Mark the shot the worker is on. Image phases fill shots in order, so it
  // is the first one without an image; video phases report a 1-based clip.
  if (IMAGE_PHASES.has(phase)) {
    const next = shots.find(shot => shot.imageStatus === 'pending')
    if (next) next.imageStatus = 'running'
  } else if (VIDEO_PHASES.has(phase)) {
    const clip = live?.progress?.current_clip
    const target = typeof clip === 'number' && clip > 0
      ? shots[clip - 1]
      : shots.find(shot => shot.videoStatus === 'pending')
    const members = target?.sequence ? target.sequence.members : target ? [target.index] : []
    for (const member of members) {
      const shot = shots[member]
      if (shot && shot.videoStatus !== 'done') shot.videoStatus = 'running'
    }
  }
  return shots
}

/**
 * Consecutive shots of one continuity group (or, without groups, one
 * location) form a scene. A group that comes back later starts a new scene,
 * since the storyboard reads in story order.
 */
export function groupScenes(shots: RunShot[]): SceneGroup[] {
  const scenes: SceneGroup[] = []
  let currentKey: string | null = null
  for (const shot of shots) {
    const key = shot.group ? `g:${shot.group}` : shot.location ? `l:${shot.location.toLowerCase()}` : ''
    const last = scenes[scenes.length - 1]
    if (last && key === currentKey) {
      last.shots.push(shot)
      if (!last.location && shot.location) last.location = shot.location
      continue
    }
    currentKey = key
    const number = scenes.length + 1
    scenes.push({
      key: `scene-${number}`,
      number,
      title: '',
      location: shot.location,
      group: shot.group,
      shots: [shot],
    })
  }
  for (const scene of scenes) {
    const parts = [`Scene ${scene.number}`]
    if (scene.location) parts.push(scene.location)
    else if (scene.group) parts.push(scene.group)
    scene.title = parts.join(' · ')
  }
  return scenes
}

/** Render units in shot order: one per H3 sequence, one per lone shot. */
export function renderUnits(shots: RunShot[]): RenderUnit[] {
  const units: RenderUnit[] = []
  const seen = new Set<number>()
  for (const shot of shots) {
    if (seen.has(shot.index)) continue
    const members = shot.sequence
      ? shot.sequence.members.filter(m => m >= 0 && m < shots.length && !seen.has(m))
      : []
    if (shot.sequence && members.length > 1 && members.includes(shot.index)) {
      members.forEach(m => seen.add(m))
      units.push({ id: shot.sequence.id, members, isSequence: true })
    } else {
      seen.add(shot.index)
      units.push({ id: `shot_${shot.index + 1}`, members: [shot.index], isSequence: false })
    }
  }
  return units
}

/** The joined edit among a run's outputs, when there is one. */
export function finalEditFile(outputFiles: string[] | null | undefined): string | null {
  const files = outputFiles || []
  return [...files].reverse().find(name => /multiclip|_joined|_final/i.test(name)) || null
}

export function isRunActive(status: string | null | undefined): boolean {
  return ACTIVE_STATUSES.has(text(status))
}

export function judgeLabel(judge: StoryboardJudgeShot | null): { label: string; tone: 'ok' | 'warn' | 'info' | 'muted' } | null {
  if (!judge) return null
  switch (judge.verdict) {
    case 'pass': return { label: 'pass', tone: 'ok' }
    case 'regenerated': return { label: 'regenerated', tone: 'info' }
    case 'flagged': return { label: 'flagged', tone: 'warn' }
    case 'failed': return { label: 'failing', tone: 'warn' }
    case 'pending': return { label: 'judging', tone: 'muted' }
    default: return null
  }
}

/** Short "1m 05s" style duration. */
export function formatSeconds(sec: number | null | undefined): string {
  if (sec == null || !Number.isFinite(sec) || sec <= 0) return '--'
  if (sec < 60) return sec < 10 ? `${Math.round(sec * 10) / 10}s` : `${Math.round(sec)}s`
  const minutes = Math.floor(sec / 60)
  const seconds = Math.round(sec % 60)
  if (minutes < 60) return `${minutes}m ${String(seconds).padStart(2, '0')}s`
  return `${Math.floor(minutes / 60)}h ${String(minutes % 60).padStart(2, '0')}m`
}

/**
 * Start time of every shot in the cut, cumulative from shot durations.
 *
 * `actual[i]`, when it is a positive number, is the measured length of shot
 * i's rendered clip and wins over the planned duration. A shot with neither
 * counts for zero seconds, so the shots after it keep their order.
 */
export function shotTimestamps(
  shots: ReadonlyArray<Pick<RunShot, 'durationSec'>>,
  actual?: ReadonlyArray<number | null | undefined>,
): number[] {
  const starts: number[] = []
  let cursor = 0
  shots.forEach((shot, i) => {
    starts.push(cursor)
    const measured = actual?.[i]
    const planned = shot.durationSec
    if (typeof measured === 'number' && Number.isFinite(measured) && measured > 0) cursor += measured
    else if (typeof planned === 'number' && Number.isFinite(planned) && planned > 0) cursor += planned
  })
  return starts
}

/** "MM:SS" timecode; minutes keep counting past the hour. */
export function formatTimecode(sec: number): string {
  const total = Number.isFinite(sec) && sec > 0 ? Math.floor(sec + 1e-6) : 0
  return `${String(Math.floor(total / 60)).padStart(2, '0')}:${String(total % 60).padStart(2, '0')}`
}
