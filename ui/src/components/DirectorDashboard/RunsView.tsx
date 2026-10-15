import { useEffect, useRef, useState } from 'react'
import { Clapperboard, Clock, Film, Loader2, Wrench } from 'lucide-react'
import { fetchPipelineCosts } from '../../api/client'
import type { PipelineListItem } from '../../types'
import { formatSeconds, isRunActive } from '../../lib/directorRun'

const STATUS_CHIP: Record<string, string> = {
  completed: 'bg-green-500/15 text-indicator-success',
  running: 'bg-blue-500/15 text-chip-blue',
  paused: 'bg-amber-500/15 text-indicator-warning',
  failed: 'bg-red-500/15 text-chip-red',
  crashed: 'bg-red-500/15 text-chip-red',
  cancelled: 'bg-bg-tertiary text-text-muted',
}

function formatDate(ts: number): string {
  return new Date(ts * 1000).toLocaleString(undefined, {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  })
}

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

function RunCard({ run, selected, onOpen }: { run: PipelineListItem; selected: boolean; onOpen: () => void }) {
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
          <span>{formatDate(run.created_at)}</span>
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

/** Every saved Director run, newest first. */
export function RunsView({ runs, selectedPid, onOpenRun }: {
  runs: PipelineListItem[]
  selectedPid: string | null
  onOpenRun: (pid: string) => void
}) {
  if (runs.length === 0) {
    return (
      <div className="text-center py-16 text-text-muted">
        <p className="text-sm">No Director runs yet</p>
        <p className="text-xs mt-1">Start a run from the Director chat and it will appear here</p>
      </div>
    )
  }
  return (
    <div className="grid gap-3 grid-cols-[repeat(auto-fill,minmax(220px,1fr))]">
      {runs.map(run => (
        <RunCard key={run.id} run={run} selected={run.id === selectedPid} onOpen={() => onOpenRun(run.id)} />
      ))}
    </div>
  )
}
