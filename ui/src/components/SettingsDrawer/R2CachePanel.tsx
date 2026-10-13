import { useCallback, useEffect, useMemo, useState } from 'react'
import { Check, CloudDownload, Loader2, RefreshCw, Square, X } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import * as api from '../../api/client'
import type { R2CacheJobModel, R2CacheStatus } from '../../types'

const JOB_ACTIVE = new Set(['starting', 'running'])
const JOB_LABEL: Record<string, string> = {
  starting: 'Starting a CPU pod…',
  running: 'Caching on a CPU pod',
  done: 'Last cache job finished',
  failed: 'Last cache job failed',
  stopped: 'Last cache job stopped',
}

function formatSize(bytes: number): string {
  return bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : `${Math.max(1, Math.round(bytes / 1e6))} MB`
}

function formatDate(seconds: number): string {
  return new Date(seconds * 1000).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}

function jobModelLabel(entry: R2CacheJobModel | undefined): string | null {
  if (!entry) return null
  if (entry.state === 'downloading') return 'downloading and uploading…'
  if (entry.state === 'failed') return `failed: ${entry.error || 'unknown error'}`
  return entry.message || (entry.seconds != null ? `cached in ${Math.round(entry.seconds / 60)} min` : 'cached')
}

/**
 * R2 checkpoint cache: which models a GPU pod can pull from R2 instead of
 * downloading, and a cheap CPU pod that fills the cache without the GPU.
 */
export function R2CachePanel() {
  const models = useStore(s => s.models)
  const [cache, setCache] = useState<R2CacheStatus | null>(null)
  const [selected, setSelected] = useState<string[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    try {
      setCache(await api.fetchR2CacheStatus())
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to read the R2 cache')
    }
  }, [])

  const jobActive = JOB_ACTIVE.has(cache?.job.state ?? 'idle')

  useEffect(() => {
    // Initial read, then follow a running job (and catch one started elsewhere).
    const first = window.setTimeout(() => void refresh(), 0)
    const timer = window.setInterval(() => void refresh(), jobActive ? 10000 : 60000)
    return () => {
      window.clearTimeout(first)
      window.clearInterval(timer)
    }
  }, [refresh, jobActive])

  const cachedByModel = useMemo(
    () => new Map((cache?.cached ?? []).map(entry => [entry.model_type, entry])),
    [cache],
  )
  const jobByModel = useMemo(
    () => new Map((cache?.job.models ?? []).map(entry => [entry.model_type, entry])),
    [cache],
  )
  // Local checkpoints only: cloud models have no weights to cache.
  const localModels = useMemo(
    () => [...models]
      .filter(model => !model.model_type.startsWith('cloud_'))
      .sort((a, b) => Number(cachedByModel.has(b.model_type)) - Number(cachedByModel.has(a.model_type))
        || a.name.localeCompare(b.name)),
    [models, cachedByModel],
  )

  const toggle = (modelType: string) => setSelected(current => (
    current.includes(modelType) ? current.filter(item => item !== modelType) : [...current, modelType]
  ))

  const warm = async () => {
    setBusy(true)
    setError(null)
    try {
      await api.warmR2Cache(selected)
      setSelected([])
      await refresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not start the cache job')
    } finally {
      setBusy(false)
    }
  }

  const stop = async () => {
    setBusy(true)
    try {
      await api.stopR2CacheWarm()
      await refresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not stop the cache job')
    } finally {
      setBusy(false)
    }
  }

  const job = cache?.job
  return (
    <section className="space-y-3 rounded-lg border border-border bg-bg-tertiary p-3">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <CloudDownload size={14} className="text-accent-blue" />
          <div>
            <div className="text-xs font-medium text-text-primary">R2 model cache</div>
            <div className="text-[10px] text-text-muted">
              {cache ? `${cache.cached.length} model${cache.cached.length === 1 ? '' : 's'} cached` : 'Reading…'}
            </div>
          </div>
        </div>
        <button
          type="button"
          onClick={() => void refresh()}
          aria-label="Refresh R2 cache"
          className="rounded p-1 text-text-muted transition-colors hover:bg-bg-hover hover:text-text-primary"
        >
          <RefreshCw size={12} />
        </button>
      </div>
      <div className="text-[9px] leading-relaxed text-text-muted">
        A GPU pod pulls cached weights from R2 in a few minutes instead of downloading them on GPU time.
        Select missing models to cache them from a CPU pod (about $0.25/h, it stops when done).
      </div>

      {(error || cache?.error) && (
        <div className="rounded-md border border-red-500/30 bg-red-500/5 px-2 py-1.5 text-[10px] text-red-100">
          {error || cache?.error}
        </div>
      )}

      {job && job.state !== 'idle' && (
        <div className="space-y-1 rounded-md border border-border/60 bg-bg-primary/40 px-2.5 py-2 text-[10px]">
          <div className="flex items-center justify-between gap-2 text-text-primary">
            <span className="flex items-center gap-1.5">
              {jobActive && <Loader2 size={11} className="animate-spin text-accent-blue" />}
              {JOB_LABEL[job.state] ?? job.state}
              {job.price_per_hour != null && <span className="text-text-muted">· ${job.price_per_hour.toFixed(2)}/h</span>}
              {job.cost != null && job.cost > 0 && <span className="text-text-muted">· ${job.cost.toFixed(2)} so far</span>}
            </span>
            {jobActive && (
              <button type="button" onClick={() => void stop()} disabled={busy}
                className="flex items-center gap-1 text-text-muted hover:text-text-primary disabled:opacity-40">
                <Square size={9} /> Stop
              </button>
            )}
          </div>
          {job.message && !jobActive && <div className="text-text-muted">{job.message}</div>}
          {(job.requested ?? []).map(modelType => (
            <div key={modelType} className="text-text-secondary">
              {modelType}: <span className="text-text-muted">{jobModelLabel(jobByModel.get(modelType)) ?? 'waiting'}</span>
            </div>
          ))}
        </div>
      )}

      <div className="max-h-64 space-y-0.5 overflow-y-auto rounded-md border border-border/60 bg-bg-primary/40 p-1">
        {localModels.map(model => {
          const cached = cachedByModel.get(model.model_type)
          return (
            <label key={model.model_type}
              className="flex cursor-pointer items-center gap-2 rounded px-1.5 py-1 text-[10px] hover:bg-bg-hover">
              {cached ? (
                <Check size={11} className="shrink-0 text-emerald-300" />
              ) : (
                <input
                  type="checkbox"
                  checked={selected.includes(model.model_type)}
                  onChange={() => toggle(model.model_type)}
                  disabled={jobActive}
                  className="shrink-0"
                />
              )}
              <span className={`min-w-0 flex-1 truncate ${cached ? 'text-text-primary' : 'text-text-secondary'}`}>{model.name}</span>
              {cached ? (
                <span className="shrink-0 text-text-muted" title={`${cached.files} files`}>
                  {formatSize(cached.bytes)} · {formatDate(cached.updated_at)}
                </span>
              ) : jobByModel.get(model.model_type)?.state === 'failed' ? (
                <X size={11} className="shrink-0 text-red-300" />
              ) : null}
            </label>
          )
        })}
      </div>

      <button
        type="button"
        onClick={() => void warm()}
        disabled={!selected.length || busy || jobActive}
        className="flex w-full items-center justify-center gap-1.5 rounded-md border border-accent-blue/50 bg-accent-blue/15 px-2 py-1.5 text-[10px] text-text-primary transition-colors hover:bg-accent-blue/25 disabled:cursor-not-allowed disabled:opacity-40"
      >
        {busy ? <Loader2 size={11} className="animate-spin" /> : <CloudDownload size={11} />}
        {selected.length ? `Cache ${selected.length} model${selected.length === 1 ? '' : 's'} on a CPU pod` : 'Select models to cache'}
      </button>
    </section>
  )
}
