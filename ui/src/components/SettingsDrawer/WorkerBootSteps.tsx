import { useEffect, useState } from 'react'
import { Check, Loader2, Minus, X } from 'lucide-react'
import type { RemoteWorkerBoot, RemoteWorkerBootStep, RemoteWorkerDownload } from '../../types'

function formatSeconds(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)} s`
  return `${Math.floor(seconds / 60)} min ${String(Math.round(seconds % 60)).padStart(2, '0')}`
}

function stepDetail(step: RemoteWorkerBootStep): string | null {
  const details = step.details ?? {}
  if (step.state === 'failed' && typeof details.error === 'string') return details.error
  if (step.state === 'started' && typeof details.done_gigabytes === 'number' && typeof details.gigabytes === 'number') {
    const rate = typeof details.megabytes_per_second === 'number' ? ` · ${details.megabytes_per_second} MB/s` : ''
    return `${details.done_gigabytes} / ${details.gigabytes} GB${rate}`
  }
  if (typeof details.gigabytes === 'number' && typeof step.seconds === 'number' && step.seconds > 0) {
    const rate = (details.gigabytes * 8000) / step.seconds
    return `${details.gigabytes} GB · ${Math.round(rate)} Mb/s`
  }
  if (typeof details.gpu === 'string') return `${details.gpu}${typeof details.vcpu === 'number' ? ` · ${details.vcpu} vCPU` : ''}`
  return null
}

function StepIcon({ state }: { state: string }) {
  if (state === 'done') return <Check size={10} className="shrink-0 text-emerald-300" />
  if (state === 'failed') return <X size={10} className="shrink-0 text-red-300" />
  if (state === 'skipped') return <Minus size={10} className="shrink-0 text-text-muted" />
  return <Loader2 size={10} className="shrink-0 animate-spin text-accent-blue" />
}

/** The current pod's startup steps, with how long each took (or has been running). */
export function WorkerBootSteps({ boot }: { boot?: RemoteWorkerBoot }) {
  const running = (boot?.steps ?? []).some(step => step.state === 'started')
  const [now, setNow] = useState(() => Date.now() / 1000)

  useEffect(() => {
    if (!running) return
    const timer = window.setInterval(() => setNow(Date.now() / 1000), 1000)
    return () => window.clearInterval(timer)
  }, [running])

  const steps = boot?.steps ?? []
  if (!steps.length) return null
  const lastEnd = Math.max(...steps.map(step => step.ended_at ?? 0))
  const total = boot?.submitted_at ? (running ? now : lastEnd) - boot.submitted_at : null

  return (
    <div className="space-y-0.5 border-t border-border/60 pt-1.5 text-[10px]">
      <div className="flex items-center justify-between text-text-muted">
        <span>Startup</span>
        {total != null && total > 0 && <span>{formatSeconds(total)}</span>}
      </div>
      {steps.map(step => {
        const elapsed = step.seconds ?? (step.state === 'started' && step.started_at ? now - step.started_at : null)
        const detail = stepDetail(step)
        return (
          <div key={`${step.phase}:${step.label}`} className="flex items-start gap-1.5 text-text-secondary">
            <span className="mt-0.5"><StepIcon state={step.state} /></span>
            <span className="min-w-0 flex-1">
              <span className="block truncate">{step.label}</span>
              {detail && (
                <span className={`block truncate ${step.state === 'failed' ? 'text-red-200' : 'text-text-muted'}`} title={detail}>
                  {detail}
                </span>
              )}
            </span>
            {elapsed != null && elapsed >= 0 && <span className="shrink-0 tabular-nums text-text-muted">{formatSeconds(elapsed)}</span>}
          </div>
        )
      })}
    </div>
  )
}

const DOWNLOAD_PHASE: Record<string, string> = {
  pulling_cache: 'From R2 cache',
  downloading: 'From source',
  publishing_cache: 'Saving to R2',
}

/** Weights the ready worker is fetching right now, with their progress. */
export function WorkerDownloads({ downloads }: { downloads?: RemoteWorkerDownload[] }) {
  if (!downloads?.length) return null
  return (
    <div className="space-y-1.5 border-t border-border/60 pt-1.5 text-[10px]">
      <div className="text-text-muted">Downloads on the worker</div>
      {downloads.map(download => {
        const progress = download.progress
        const total = progress?.total_bytes ?? 0
        const percent = progress && total > 0 ? Math.min(100, (progress.done_bytes / total) * 100) : null
        const rate = progress?.bytes_per_second ? ` · ${Math.round(progress.bytes_per_second / 1e6)} MB/s` : ''
        const amount = progress
          ? `${(progress.done_bytes / 1e9).toFixed(1)}${total > 0 ? ` / ${(total / 1e9).toFixed(1)}` : ''} GB${rate}`
          : 'starting…'
        return (
          <div key={download.model_type} className="space-y-0.5 text-text-secondary">
            <div className="flex items-center gap-1.5">
              <Loader2 size={10} className="shrink-0 animate-spin text-accent-blue" />
              <span className="min-w-0 flex-1 truncate">{download.model_type}</span>
              <span className="shrink-0 text-text-muted">{DOWNLOAD_PHASE[download.phase ?? ''] ?? download.phase ?? ''}</span>
            </div>
            <div className="h-1 overflow-hidden rounded bg-bg-hover">
              <div className="h-full bg-accent-blue transition-all" style={{ width: `${percent ?? 0}%` }} />
            </div>
            <div className="truncate text-text-muted" title={progress?.file ?? undefined}>
              {amount}{progress?.file ? ` · ${progress.file}` : ''}
            </div>
          </div>
        )
      })}
    </div>
  )
}
