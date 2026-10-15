import { useState } from 'react'
import { Clapperboard, Wrench } from 'lucide-react'
import type { PipelineListItem } from '../../types'
import { AsyncCombobox } from '../shared/AsyncCombobox'
import {
  formatRelativeTime, runDisplayTime, runShortLabel, runTimeTooltip,
} from '../../lib/directorRunList'
import { useRunPages } from './useRunPages'
import { useNow } from '../../lib/useNow'

const STATUS_DOT: Record<string, string> = {
  completed: 'bg-indicator-success',
  running: 'bg-accent-blue animate-pulse',
  paused: 'bg-indicator-warning',
  failed: 'bg-red-500',
  crashed: 'bg-red-500',
  cancelled: 'bg-text-muted',
}

export function RunStatusDot({ status }: { status: string }) {
  return (
    <span
      className={`inline-block w-2 h-2 rounded-full shrink-0 ${STATUS_DOT[status] || 'bg-text-muted'}`}
      title={status}
      aria-label={status}
    />
  )
}

/** Relative time of a run's most meaningful moment, exact times on hover. */
export function RunTime({ run, now, className = '' }: {
  run: Pick<PipelineListItem, 'created_at' | 'completed_at' | 'updated_at'>
  now: number
  className?: string
}) {
  const shown = runDisplayTime(run)
  if (!shown) return <span className={className}>—</span>
  return (
    <time
      dateTime={new Date(shown.ts * 1000).toISOString()}
      title={runTimeTooltip(run)}
      className={className}
    >
      {shown.kind === 'started' ? '' : `${shown.kind} `}{formatRelativeTime(shown.ts, now)}
    </time>
  )
}

function RunThumb({ run, size }: { run: PipelineListItem; size: 'sm' | 'md' }) {
  const [failedUrl, setFailedUrl] = useState<string | null>(null)
  const box = size === 'sm' ? 'w-8 h-[18px]' : 'w-12 h-7'
  const url = run.thumbnail_url
  return (
    <span className={`relative shrink-0 ${box} rounded-sm overflow-hidden bg-media-canvas flex items-center justify-center text-text-muted`}>
      {url && failedUrl !== url ? (
        <img src={url} alt="" loading="lazy" onError={() => setFailedUrl(url)} className="absolute inset-0 w-full h-full object-cover" />
      ) : (
        <Clapperboard size={size === 'sm' ? 10 : 12} />
      )}
    </span>
  )
}

function RunOption({ run, now }: { run: PipelineListItem; now: number }) {
  return (
    <span className="flex items-center gap-2 min-w-0">
      <RunThumb run={run} size="md" />
      <span className="flex-1 min-w-0">
        <span className="flex items-center gap-1.5 min-w-0">
          <RunStatusDot status={run.status} />
          <span className="truncate text-[11px] text-text-primary">{runShortLabel(run, 80)}</span>
        </span>
        <span className="flex items-center gap-1.5 text-[10px] text-text-muted">
          <RunTime run={run} now={now} />
          <span>· {run.clip_count} clips</span>
          <span className="truncate">· {run.pipeline_type.replace(/_/g, ' ')}</span>
          {run.repair_status && (
            <span className="flex items-center gap-0.5 text-chip-orange"><Wrench size={9} /> {run.repair_status}</span>
          )}
        </span>
      </span>
    </span>
  )
}

/**
 * Searchable run selector backed by the paged runs API. Only the first page
 * is fetched when it opens; more pages load as the list scrolls.
 */
export function RunPicker({
  selected, onSelect, label = 'Director run', placeholder = 'Select a run…', compact = false,
  className = '', align = 'left',
}: {
  selected: PipelineListItem | null
  onSelect: (run: PipelineListItem) => void
  label?: string
  placeholder?: string
  compact?: boolean
  className?: string
  align?: 'left' | 'right'
}) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const pages = useRunPages({ query, enabled: open })
  const now = useNow()

  const value = selected ? (
    <span className="flex items-center gap-2 min-w-0">
      {!compact && <RunThumb run={selected} size="sm" />}
      <RunStatusDot status={selected.status} />
      <span className="truncate">{runShortLabel(selected, compact ? 28 : 60)}</span>
      {!compact && <RunTime run={selected} now={now} className="shrink-0 text-[10px] text-text-muted" />}
    </span>
  ) : (
    <span className="text-text-muted">{placeholder}</span>
  )

  return (
    <AsyncCombobox
      label={label}
      value={value}
      items={pages.items}
      getKey={run => run.id}
      selectedKey={selected?.id ?? null}
      renderOption={run => <RunOption run={run} now={now} />}
      onSelect={onSelect}
      onQueryChange={setQuery}
      onOpenChange={setOpen}
      loading={pages.loading}
      hasMore={pages.hasMore}
      onLoadMore={pages.loadMore}
      error={pages.error}
      emptyText={query ? 'No run matches this search' : 'No Director runs yet'}
      searchPlaceholder="Search runs by description or id…"
      className={className}
      triggerClassName={compact ? 'py-0.5 text-[10px]' : ''}
      align={align}
    />
  )
}
