import { useEffect, useRef, useState } from 'react'
import { Clapperboard, Clock, Film, Loader2, Search, Wrench } from 'lucide-react'
import { fetchPipelineCosts } from '../../api/client'
import type { PipelineListItem } from '../../types'
import { formatSeconds, isRunActive } from '../../lib/directorRun'
import { runActivityTime } from '../../lib/directorRunList'
import { useNow } from '../../lib/useNow'
import { RunTime } from './RunPicker'
import { useRunPages } from './useRunPages'

const STATUS_CHIP: Record<string, string> = {
  completed: 'bg-green-500/15 text-indicator-success',
  running: 'bg-blue-500/15 text-chip-blue',
  paused: 'bg-amber-500/15 text-indicator-warning',
  failed: 'bg-red-500/15 text-chip-red',
  crashed: 'bg-red-500/15 text-chip-red',
  cancelled: 'bg-bg-tertiary text-text-muted',
}

const STATUS_FILTERS: { value: string; label: string }[] = [
  { value: '', label: 'All' },
  { value: 'running,paused,repairing', label: 'Active' },
  { value: 'completed', label: 'Completed' },
  { value: 'failed,crashed,cancelled', label: 'Stopped' },
]

/** Cost of one run, read once the card scrolls into view. */
function useVisibleCost(pid: string, status: string) {
  const ref = useRef<HTMLButtonElement | null>(null)
  const [visible, setVisible] = useState(false)
  const [usd, setUsd] = useState<number | null>(null)
  useEffect(() => {
    const el = ref.current
    if (!el || visible) return
    const observer = new IntersectionObserver(entries => {
      if (entries.some(entry => entry.isIntersecting)) {
        setVisible(true)
        observer.disconnect()
      }
    }, { rootMargin: '200px' })
    observer.observe(el)
    return () => observer.disconnect()
  }, [visible])
  useEffect(() => {
    if (!visible) return
    let current = true
    fetchPipelineCosts(pid)
      .then(costs => { if (current) setUsd(costs.usd || null) })
      .catch(() => undefined)
    return () => { current = false }
  }, [visible, pid, status])
  return { ref, usd }
}

function RunCard({ run, selected, onOpen, now }: { run: PipelineListItem; selected: boolean; onOpen: () => void; now: number }) {
  const { ref, usd } = useVisibleCost(run.id, run.status)
  const [thumbFailed, setThumbFailed] = useState(false)
  const active = isRunActive(run.status)
  return (
    <button
      ref={ref}
      type="button"
      onClick={onOpen}
      className={`text-left rounded-lg border bg-bg-secondary overflow-hidden flex flex-col transition-colors hover:border-border-light focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-blue ${selected ? 'border-accent-blue' : 'border-border'}`}
    >
      <div className="relative aspect-video bg-media-canvas">
        {run.thumbnail_url && !thumbFailed ? (
          <img src={run.thumbnail_url} alt="" loading="lazy" onError={() => setThumbFailed(true)}
            className="absolute inset-0 w-full h-full object-contain" />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center text-text-muted"><Clapperboard size={22} /></div>
        )}
        <span className={`absolute top-1.5 left-1.5 px-1.5 py-0.5 rounded text-[9px] font-medium flex items-center gap-1 backdrop-blur-sm ${STATUS_CHIP[run.status] || 'bg-bg-tertiary text-text-muted'}`}>
          {active && <Loader2 size={9} className="animate-spin" />}
          {run.status}
        </span>
        {run.repair_status && (
          <span className="absolute top-1.5 right-1.5 px-1.5 py-0.5 rounded text-[9px] bg-orange-500/20 text-chip-orange flex items-center gap-1">
            <Wrench size={9} /> {run.repair_status}
          </span>
        )}
      </div>
      <div className="p-2.5 space-y-1">
        <p className="text-[11px] text-text-primary line-clamp-2 min-h-[2lh]">
          {run.scene_description || <span className="italic text-text-muted">No description</span>}
        </p>
        <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[10px] text-text-muted">
          <RunTime run={run} now={now} />
          <span className="flex items-center gap-0.5"><Film size={9} /> {run.clip_count}</span>
          {run.duration_sec ? <span className="flex items-center gap-0.5"><Clock size={9} /> {formatSeconds(run.duration_sec)}</span> : null}
          {usd != null && <span className="text-text-secondary">${usd.toFixed(2)}</span>}
        </div>
        <p className="text-[9px] text-text-muted truncate">
          {run.pipeline_type.replace(/_/g, ' ')}{run.workspace && run.workspace !== 'default' ? ` · ${run.workspace}` : ''}
        </p>
      </div>
    </button>
  )
}

/**
 * Saved Director runs, newest first, paged from the server: the next page
 * loads when the end of the grid scrolls into view.
 */
export function RunsView({ liveRuns, selectedPid, onOpenRun }: {
  /** The store's run list: fresher statuses for the cards already shown. */
  liveRuns: PipelineListItem[]
  selectedPid: string | null
  onOpenRun: (pid: string) => void
}) {
  const [input, setInput] = useState('')
  const [query, setQuery] = useState('')
  const [status, setStatus] = useState('')
  const pages = useRunPages({ query, status, pageSize: 24 })
  const { absorb, reload, loadMore, hasMore, loading, items } = pages
  const now = useNow()
  const sentinelRef = useRef<HTMLDivElement | null>(null)
  const reloadedForRef = useRef('')

  useEffect(() => {
    const timer = window.setTimeout(() => setQuery(input), 250)
    return () => window.clearTimeout(timer)
  }, [input])

  // Keep the shown cards current, and re-read the first page when the store
  // knows of a run newer than every card (one started meanwhile).
  useEffect(() => {
    absorb(liveRuns)
    if (query || status || items.length === 0) return
    const newestShown = runActivityTime(items[0])
    const shown = new Set(items.map(run => run.id))
    const missing = liveRuns
      .filter(run => !shown.has(run.id) && runActivityTime(run) > newestShown)
      .map(run => run.id)
      .join(',')
    // Once per set of missing runs, so a stale store row cannot loop.
    if (missing && missing !== reloadedForRef.current) {
      reloadedForRef.current = missing
      reload()
    }
  }, [liveRuns, absorb, reload, query, status, items])

  useEffect(() => {
    const el = sentinelRef.current
    if (!el || !hasMore || loading) return
    const observer = new IntersectionObserver(entries => {
      if (entries.some(entry => entry.isIntersecting)) loadMore()
    }, { rootMargin: '400px' })
    observer.observe(el)
    return () => observer.disconnect()
  }, [hasMore, loading, loadMore])

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <div className="relative flex-1 min-w-[12rem] max-w-sm">
          <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-text-muted pointer-events-none" />
          <input
            type="search"
            value={input}
            onChange={event => setInput(event.target.value)}
            placeholder="Search runs by description or id…"
            aria-label="Search Director runs"
            className="w-full rounded-lg border border-border bg-bg-tertiary py-1.5 pl-7 pr-2 text-xs text-text-primary outline-none placeholder:text-text-muted focus:border-accent-blue/60"
          />
        </div>
        <div role="radiogroup" aria-label="Filter runs by status" className="flex rounded-lg border border-border bg-bg-tertiary p-0.5">
          {STATUS_FILTERS.map(filter => (
            <button
              key={filter.value || 'all'}
              type="button"
              role="radio"
              aria-checked={status === filter.value}
              onClick={() => setStatus(filter.value)}
              className={`px-2 py-0.5 rounded-md text-[10px] ${status === filter.value ? 'bg-toggle-active text-white' : 'text-text-secondary hover:bg-bg-hover'}`}
            >
              {filter.label}
            </button>
          ))}
        </div>
        {pages.total != null && (
          <span className="text-[10px] text-text-muted">{pages.total} run{pages.total === 1 ? '' : 's'}</span>
        )}
        {loading && <Loader2 size={12} className="animate-spin text-text-muted" />}
      </div>

      {items.length === 0 ? (
        pages.initialLoading ? (
          <div className="flex items-center justify-center py-16 text-text-muted text-xs">
            <Loader2 size={16} className="animate-spin mr-2" /> Loading runs…
          </div>
        ) : pages.error ? (
          <div className="text-center py-16 text-text-muted">
            <p className="text-sm">Could not load runs</p>
            <button type="button" onClick={reload} className="text-xs mt-1 text-accent-blue hover:underline">Retry</button>
          </div>
        ) : (
          <div className="text-center py-16 text-text-muted">
            <p className="text-sm">{query || status ? 'No run matches these filters' : 'No Director runs yet'}</p>
            {!query && !status && <p className="text-xs mt-1">Start a run from the Director chat and it will appear here</p>}
          </div>
        )
      ) : (
        <div className={`grid gap-3 grid-cols-[repeat(auto-fill,minmax(220px,1fr))] transition-opacity ${pages.initialLoading ? 'opacity-60' : ''}`}>
          {items.map(run => (
            <RunCard key={run.id} run={run} now={now} selected={run.id === selectedPid} onOpen={() => onOpenRun(run.id)} />
          ))}
        </div>
      )}

      {hasMore && (
        <div ref={sentinelRef} className="flex justify-center py-3">
          <button
            type="button"
            onClick={loadMore}
            disabled={loading}
            className="flex items-center gap-1.5 rounded-lg border border-border px-3 py-1 text-[11px] text-text-secondary hover:bg-bg-hover disabled:opacity-60"
          >
            {loading ? <><Loader2 size={11} className="animate-spin" /> Loading…</> : 'Load more'}
          </button>
        </div>
      )}
      {pages.error && items.length > 0 && (
        <p className="text-center text-[11px] text-red-400">Could not load more runs: {pages.error}</p>
      )}
    </div>
  )
}
