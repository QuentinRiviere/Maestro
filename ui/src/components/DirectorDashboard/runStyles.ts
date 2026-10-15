import type { StageStatus } from '../../lib/directorRun'

export const STATUS_DOT: Record<StageStatus, string> = {
  pending: 'bg-text-muted/40',
  running: 'bg-accent-blue animate-pulse',
  done: 'bg-indicator-success',
  failed: 'bg-chip-red',
  flagged: 'bg-indicator-warning',
  skipped: 'bg-transparent border border-border-light',
}

export const STATUS_BORDER: Record<StageStatus, string> = {
  pending: 'border-border',
  running: 'border-accent-blue',
  done: 'border-indicator-success/50',
  failed: 'border-chip-red/60',
  flagged: 'border-indicator-warning/70',
  skipped: 'border-border border-dashed',
}

export const STATUS_LABEL: Record<StageStatus, string> = {
  pending: 'pending', running: 'running', done: 'done', failed: 'missing', flagged: 'flagged', skipped: 'skipped',
}

/** CSS aspect ratio of the run's frames (9:16 when unknown). */
export function frameAspect(ratio: string | null | undefined): string {
  const match = String(ratio || '').match(/^(\d+):(\d+)$/)
  return match ? `${match[1]} / ${match[2]}` : '9 / 16'
}
