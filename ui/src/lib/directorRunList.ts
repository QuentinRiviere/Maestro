/**
 * Pure helpers for lists of saved Director runs: ordering, page merging,
 * labels and times. The server pages `GET /director/pipelines` newest first;
 * these keep merged client lists in that same order. No React, no fetch.
 */
import type { PipelineListItem, SavedPipelineState } from '../types'

export interface RunPageQuery {
  limit?: number
  cursor?: string | null
  q?: string
  status?: string
}

export interface RunPage {
  pipelines: PipelineListItem[]
  next_cursor?: string | null
  total?: number
}

/** A positive Unix time in seconds, or null (ms values are scaled down). */
export function toTimestamp(value: unknown): number | null {
  let n: number
  if (typeof value === 'number') n = value
  else if (typeof value === 'string' && value.trim()) {
    n = Number(value)
    if (!Number.isFinite(n)) n = Date.parse(value) / 1000
  } else return null
  if (!Number.isFinite(n) || n <= 0) return null
  return n > 1e11 ? n / 1000 : n
}

/** The run's most recent known activity (0 when nothing is known). */
export function runActivityTime(run: Pick<PipelineListItem, 'created_at' | 'completed_at' | 'updated_at'>): number {
  return Math.max(
    0,
    toTimestamp(run.created_at) ?? 0,
    toTimestamp(run.completed_at) ?? 0,
    toTimestamp(run.updated_at) ?? 0,
  )
}

/** Newest activity first, ties broken by id: the server's page order. */
export function compareRuns(a: PipelineListItem, b: PipelineListItem): number {
  const diff = runActivityTime(b) - runActivityTime(a)
  if (diff !== 0) return diff
  return a.id < b.id ? -1 : a.id > b.id ? 1 : 0
}

/**
 * Merge a fetched page into the runs already shown. Fresher rows replace
 * older copies of the same run (a status may have changed between pages),
 * and the result is re-sorted so a run that moved is not shown twice.
 */
export function mergeRunPages(existing: PipelineListItem[], incoming: PipelineListItem[]): PipelineListItem[] {
  const byId = new Map<string, PipelineListItem>()
  for (const run of existing) byId.set(run.id, run)
  for (const run of incoming) byId.set(run.id, run)
  return [...byId.values()].sort(compareRuns)
}

/** Query string for `GET /director/pipelines` (empty values omitted). */
export function runPageSearch(query: RunPageQuery): string {
  const params = new URLSearchParams()
  if (query.limit) params.set('limit', String(query.limit))
  if (query.cursor) params.set('cursor', query.cursor)
  const q = query.q?.trim()
  if (q) params.set('q', q)
  if (query.status) params.set('status', query.status)
  const text = params.toString()
  return text ? `?${text}` : ''
}

export type RunTimeKind = 'started' | 'updated' | 'finished'

/**
 * The single time worth showing for a run: when it finished, else when it
 * was last updated if that is clearly after the start, else when it started.
 */
export function runDisplayTime(run: Pick<PipelineListItem, 'created_at' | 'completed_at' | 'updated_at'>): { ts: number; kind: RunTimeKind } | null {
  const created = toTimestamp(run.created_at)
  const completed = toTimestamp(run.completed_at)
  const updated = toTimestamp(run.updated_at)
  if (completed && (!updated || completed >= updated - 60)) return { ts: completed, kind: 'finished' }
  if (updated && (!created || updated - created > 60)) return { ts: updated, kind: 'updated' }
  if (created) return { ts: created, kind: 'started' }
  if (updated) return { ts: updated, kind: 'updated' }
  return null
}

/** "just now", "5 min ago", "3 h ago", "yesterday", "4 d ago", else a date. */
export function formatRelativeTime(ts: number | null | undefined, nowMs: number = Date.now()): string {
  if (!ts) return '—'
  const seconds = Math.round(nowMs / 1000 - ts)
  if (seconds < -60) return formatExactTime(ts)
  if (seconds < 45) return 'just now'
  const minutes = Math.round(seconds / 60)
  if (minutes < 60) return `${minutes} min ago`
  const hours = Math.round(seconds / 3600)
  if (hours < 24) return `${hours} h ago`
  const days = Math.floor(seconds / 86400)
  if (days === 1) return 'yesterday'
  if (days < 7) return `${days} d ago`
  const date = new Date(ts * 1000)
  const sameYear = date.getFullYear() === new Date(nowMs).getFullYear()
  return date.toLocaleDateString(undefined, sameYear
    ? { month: 'short', day: 'numeric' }
    : { year: 'numeric', month: 'short', day: 'numeric' })
}

export function formatExactTime(ts: number | null | undefined): string {
  if (!ts) return '—'
  return new Date(ts * 1000).toLocaleString(undefined, {
    year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit',
  })
}

/** Tooltip listing every known time of a run. */
export function runTimeTooltip(run: Pick<PipelineListItem, 'created_at' | 'completed_at' | 'updated_at'>): string {
  const lines: string[] = []
  const created = toTimestamp(run.created_at)
  const completed = toTimestamp(run.completed_at)
  const updated = toTimestamp(run.updated_at)
  if (created) lines.push(`Started ${formatExactTime(created)}`)
  if (completed) lines.push(`Finished ${formatExactTime(completed)}`)
  if (updated) lines.push(`Updated ${formatExactTime(updated)}`)
  return lines.join('\n')
}

/** Short one-line name of a run. */
export function runShortLabel(run: Pick<PipelineListItem, 'id' | 'scene_description' | 'pipeline_type'>, max = 60): string {
  const description = (run.scene_description || '').replace(/\s+/g, ' ').trim()
  const label = description || `${(run.pipeline_type || 'Director').replace(/_/g, ' ')} · ${run.id}`
  return label.length > max ? `${label.slice(0, max - 1).trimEnd()}…` : label
}

/** List-item view of a loaded run, for showing it when it is not in a page. */
export function runListItemFromState(state: SavedPipelineState, known?: PipelineListItem | null): PipelineListItem {
  if (known && known.id === state.pipeline_id) {
    return { ...known, status: state.status || known.status }
  }
  const clipCount = state.clips?.length || 0
  return {
    id: state.pipeline_id,
    status: state.status,
    pipeline_type: state.pipeline_type,
    created_at: state.created_at ?? null,
    completed_at: state.completed_at ?? null,
    clip_count: clipCount,
    output_count: state.output_files?.length || 0,
    scene_description: state.scene_description || '',
    workspace: state.workspace || 'default',
    thumbnail_url: clipCount || state.output_files?.length
      ? `/api/v1/director/pipelines/${encodeURIComponent(state.pipeline_id)}/thumbnail`
      : null,
    repair_status: state.repair?.status ?? null,
  }
}
