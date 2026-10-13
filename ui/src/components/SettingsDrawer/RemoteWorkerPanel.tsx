import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  ChevronDown,
  ChevronRight,
  Cloud,
  Download,
  HardDrive,
  Play,
  PlugZap,
  RefreshCw,
  Save,
  Square,
} from 'lucide-react'
import { useStore } from '../../stores/useStore'
import * as api from '../../api/client'
import type {
  RemoteWorkerConnectionTest,
  RemoteWorkerLogLine,
  RemoteWorkerSettings,
  RemoteWorkerState,
  RemoteWorkerStatus,
} from '../../types'
import { Toggle } from './NotificationSettingsPanel'
import { R2CachePanel } from './R2CachePanel'
import { WorkerBootSteps } from './WorkerBootSteps'

const STATE_LABEL: Record<RemoteWorkerState, string> = {
  not_configured: 'Not configured',
  unknown: 'Unreachable',
  stopped: 'Stopped',
  starting: 'Starting · waiting for a GPU',
  booting: 'Booting Maestro',
  ready: 'Ready',
  stopping: 'Stopping',
}

const STATE_DOT: Record<RemoteWorkerState, string> = {
  not_configured: 'bg-text-muted',
  unknown: 'bg-amber-400',
  stopped: 'bg-text-muted',
  starting: 'bg-amber-400 animate-pulse',
  booting: 'bg-amber-400 animate-pulse',
  ready: 'bg-emerald-400',
  stopping: 'bg-amber-400',
}

const ACTIVE_STATES: RemoteWorkerState[] = ['starting', 'booting', 'ready', 'stopping']
const LOG_STATES: RemoteWorkerState[] = ['starting', 'booting']

function formatMinutes(seconds: number | null | undefined): string {
  if (seconds == null) return '—'
  const minutes = Math.ceil(seconds / 60)
  return minutes <= 1 ? '< 1 min' : `${minutes} min`
}

function Field({
  label,
  value,
  onChange,
  type = 'text',
  placeholder,
  hint,
  locked = false,
}: {
  label: string
  value: string | number
  onChange: (value: string) => void
  type?: 'text' | 'password' | 'number'
  placeholder?: string
  hint?: string
  locked?: boolean
}) {
  return (
    <label className="block space-y-1">
      <span className="text-[10px] text-text-secondary">{label}</span>
      <input
        type={type}
        value={value}
        placeholder={placeholder}
        disabled={locked}
        onChange={event => onChange(event.target.value)}
        className="w-full rounded-md border border-border bg-bg-primary px-2 py-1.5 text-[11px] text-text-primary placeholder:text-text-muted focus:border-accent-blue focus:outline-none disabled:cursor-not-allowed disabled:opacity-50"
      />
      {locked && <span className="block text-[9px] leading-relaxed text-amber-200">Set by environment</span>}
      {hint && <span className="block text-[9px] leading-relaxed text-text-muted">{hint}</span>}
    </label>
  )
}

function ActionButton({
  onClick,
  disabled,
  children,
  tone = 'neutral',
}: {
  onClick: () => void
  disabled?: boolean
  children: React.ReactNode
  tone?: 'neutral' | 'primary' | 'danger'
}) {
  const toneClass = tone === 'primary'
    ? 'border-accent-blue/50 bg-accent-blue/15 text-text-primary hover:bg-accent-blue/25'
    : tone === 'danger'
      ? 'border-red-500/40 bg-red-500/10 text-red-100 hover:bg-red-500/20'
      : 'border-border text-text-secondary hover:bg-bg-hover hover:text-text-primary'
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`flex items-center justify-center gap-1.5 rounded-md border px-2 py-1.5 text-[10px] transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${toneClass}`}
    >
      {children}
    </button>
  )
}

/**
 * Remote GPU worker: a headless Maestro running on a dstack/RunPod service.
 *
 * Generations routed here start the worker automatically when needed; the
 * worker stops itself after the idle TTL, and can be stopped right away.
 */
export function RemoteWorkerPanel() {
  const models = useStore(s => s.models)
  const [settings, setSettings] = useState<RemoteWorkerSettings | null>(null)
  const [draft, setDraft] = useState<Partial<RemoteWorkerSettings>>({})
  const [status, setStatus] = useState<RemoteWorkerStatus | null>(null)
  const [connection, setConnection] = useState<RemoteWorkerConnectionTest | null>(null)
  const [logs, setLogs] = useState<RemoteWorkerLogLine[]>([])
  const [showLogs, setShowLogs] = useState(false)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [busy, setBusy] = useState<string | null>(null)
  const [message, setMessage] = useState<{ tone: 'info' | 'error'; text: string } | null>(null)
  const [modelType, setModelType] = useState('')
  const lastLogTimestamp = useRef<string | undefined>(undefined)

  const merged = useMemo(
    () => (settings ? { ...settings, ...draft } as RemoteWorkerSettings : null),
    [settings, draft],
  )
  const dirty = Object.keys(draft).length > 0
  const isLocked = (key: string) => Boolean(settings?.env_locked?.includes(key))
  const state: RemoteWorkerState = status?.state ?? 'not_configured'

  const videoModels = useMemo(
    () => [...models]
      .filter(model => model.is_t2v || model.is_i2v)
      .sort((a, b) => a.name.localeCompare(b.name)),
    [models],
  )

  const run = useCallback(async (label: string, action: () => Promise<void>) => {
    setBusy(label)
    setMessage(null)
    try {
      await action()
    } catch (error) {
      setMessage({ tone: 'error', text: error instanceof Error ? error.message : String(error) })
    } finally {
      setBusy(null)
    }
  }, [])

  const refreshStatus = useCallback(async () => {
    try {
      setStatus(await api.fetchRemoteWorkerStatus())
    } catch (error) {
      setMessage({ tone: 'error', text: error instanceof Error ? error.message : String(error) })
    }
  }, [])

  useEffect(() => {
    void run('load', async () => {
      const loaded = await api.fetchRemoteWorkerSettings()
      setSettings(loaded)
      setModelType(current => current || loaded.default_model_type)
      await refreshStatus()
    })
  }, [run, refreshStatus])

  // Poll faster while the worker is changing state.
  useEffect(() => {
    if (!settings?.configured) return
    const delay = state === 'starting' || state === 'booting' || state === 'stopping' ? 4000 : 15000
    const timer = window.setInterval(() => void refreshStatus(), delay)
    return () => window.clearInterval(timer)
  }, [settings?.configured, state, refreshStatus])

  useEffect(() => {
    if (!showLogs || !LOG_STATES.includes(state)) return
    const tick = async () => {
      try {
        const lines = await api.fetchRemoteWorkerLogs(lastLogTimestamp.current)
        if (lines.length) {
          lastLogTimestamp.current = lines[lines.length - 1].timestamp
          setLogs(previous => [...previous, ...lines].slice(-200))
        }
      } catch {
        // Logs are best effort while the pod is still being scheduled.
      }
    }
    void tick()
    const timer = window.setInterval(() => void tick(), 4000)
    return () => window.clearInterval(timer)
  }, [showLogs, state])

  const update = <K extends keyof RemoteWorkerSettings>(key: K, value: RemoteWorkerSettings[K]) => {
    setDraft(previous => ({ ...previous, [key]: value }))
  }

  const save = () => run('save', async () => {
    const saved = await api.updateRemoteWorkerSettings(draft)
    setSettings(saved)
    setDraft({})
    setMessage({ tone: 'info', text: 'Remote worker settings saved.' })
    await refreshStatus()
  })

  const importDstack = () => run('import', async () => {
    const saved = await api.importRemoteWorkerDstackConfig()
    setSettings(saved)
    setDraft({})
    setMessage({ tone: 'info', text: `Imported project "${saved.project}" from ~/.dstack/config.yml.` })
    await refreshStatus()
  })

  const testConnection = () => run('test', async () => {
    const result = await api.testRemoteWorkerConnection()
    setConnection(result)
    setMessage({ tone: 'info', text: `Connected to dstack ${result.server_version}.` })
  })

  const start = () => run('start', async () => {
    setLogs([])
    lastLogTimestamp.current = undefined
    setStatus(await api.startRemoteWorker(modelType || undefined))
  })

  const stop = () => run('stop', async () => {
    setStatus(await api.stopRemoteWorker())
  })

  if (!merged) {
    return <div className="text-[10px] text-text-muted">{busy === 'load' ? 'Loading…' : message?.text}</div>
  }

  const secretField = (
    key: 'registry_token' | 'r2_access_key_id' | 'r2_secret_access_key' | 'cf_api_token',
    label: string,
    stored: boolean,
  ) => (
    <Field
      locked={isLocked(key)}
      label={label}
      type="password"
      value={draft[key] ?? ''}
      onChange={value => update(key, value)}
      placeholder={stored ? `Stored (${settings?.[key]}) · type to replace` : ''}
    />
  )
  const canStart = merged.configured && !dirty && !ACTIVE_STATES.includes(state) && busy === null
  const canStop = merged.configured && (state === 'starting' || state === 'booting' || state === 'ready') && busy === null

  return (
    <div className="space-y-4">
      {message && (
        <div className={`rounded-md border px-2.5 py-2 text-[10px] leading-relaxed ${
          message.tone === 'error'
            ? 'border-red-500/30 bg-red-500/5 text-red-100'
            : 'border-border/60 bg-bg-primary/40 text-text-secondary'
        }`}>
          {message.text}
        </div>
      )}

      {/* ── Worker status & controls ─────────────────────────────── */}
      <section className="space-y-3 rounded-lg border border-border bg-bg-tertiary p-3">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Cloud size={15} className="text-accent-blue" />
            <div>
              <div className="text-xs font-medium text-text-primary">Remote GPU worker</div>
              <div className="text-[10px] text-text-muted">Headless Maestro on RunPod via dstack</div>
            </div>
          </div>
          <button
            type="button"
            onClick={() => void refreshStatus()}
            aria-label="Refresh worker status"
            className="rounded p-1 text-text-muted transition-colors hover:bg-bg-hover hover:text-text-primary"
          >
            <RefreshCw size={12} />
          </button>
        </div>

        <div className="space-y-1.5 rounded-md border border-border/60 bg-bg-primary/40 px-2.5 py-2">
          <div className="flex items-center gap-2 text-[11px] text-text-primary">
            <span className={`h-2 w-2 rounded-full ${STATE_DOT[state]}`} />
            {STATE_LABEL[state]}
            {state === 'starting' && status?.image_pull_percent != null && (
              <span className="text-text-muted">· image {status.image_pull_percent}%</span>
            )}
          </div>
          {status?.message && <div className="text-[10px] leading-relaxed text-text-muted">{status.message}</div>}
          {ACTIVE_STATES.includes(state) && (
            <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-[10px] text-text-secondary">
              {status?.model_type && <span className="col-span-2 truncate">Model · {status.model_type}</span>}
              {status?.price_per_hour != null && <span>${status.price_per_hour.toFixed(2)}/h</span>}
              {status?.cost != null && <span>Spent ${status.cost.toFixed(2)}</span>}
              {status?.gpu_region && <span>{status.gpu_region}</span>}
              {state === 'ready' && (
                <span>
                  {status?.active_jobs
                    ? `${status.active_jobs} job running`
                    : status?.review_hold
                      ? 'Kept up for a Director review'
                      : `Auto-stop in ${formatMinutes(status?.idle_stop_in_seconds)}`}
                </span>
              )}
              {!!status?.preload_models?.length && (
                <span className="col-span-2 truncate" title={status.preload_models.join(', ')}>
                  Preloading · {status.preload_models.join(', ')}
                </span>
              )}
            </div>
          )}
          {ACTIVE_STATES.includes(state) && <WorkerBootSteps boot={status?.boot} />}
        </div>

        <label className="block space-y-1">
          <span className="text-[10px] text-text-secondary">Model to preload</span>
          <select
            value={modelType}
            onChange={event => setModelType(event.target.value)}
            className="w-full rounded-md border border-border bg-bg-primary px-2 py-1.5 text-[11px] text-text-primary focus:border-accent-blue focus:outline-none"
          >
            <option value="">None (download on first generation)</option>
            {videoModels.map(model => (
              <option key={model.model_type} value={model.model_type}>{model.name}</option>
            ))}
          </select>
        </label>

        <div className="grid grid-cols-2 gap-2">
          <ActionButton onClick={start} disabled={!canStart} tone="primary">
            <Play size={11} /> {busy === 'start' ? 'Starting…' : 'Start worker'}
          </ActionButton>
          <ActionButton onClick={stop} disabled={!canStop} tone="danger">
            <Square size={11} /> {busy === 'stop' ? 'Stopping…' : 'Stop worker'}
          </ActionButton>
        </div>
        {dirty && <div className="text-[9px] text-amber-200">Save settings before starting the worker.</div>}

        <button
          type="button"
          onClick={() => setShowLogs(value => !value)}
          className="flex items-center gap-1 text-[10px] text-text-secondary hover:text-text-primary"
        >
          {showLogs ? <ChevronDown size={11} /> : <ChevronRight size={11} />} Boot logs
        </button>
        {showLogs && (
          <pre className="max-h-48 overflow-auto whitespace-pre-wrap rounded-md bg-bg-primary/70 p-2 text-[9px] leading-relaxed text-text-secondary">
            {logs.length ? logs.map(line => line.message).join('') : 'Logs appear while the worker is starting.'}
          </pre>
        )}
      </section>

      {/* ── Generation routing ───────────────────────────────────── */}
      <section className="space-y-3 rounded-lg border border-border bg-bg-tertiary p-3">
        <Toggle
          checked={merged.route_generations}
          onChange={checked => update('route_generations', checked)}
          label="Run generations on the remote worker"
          description={isLocked('route_generations') ? 'Set by environment.' : "Studio and Director jobs execute on the remote GPU; outputs come back to this computer's gallery."}
          disabled={!merged.configured || isLocked('route_generations')}
        />
        <Toggle
          checked={merged.auto_start}
          onChange={checked => update('auto_start', checked)}
          label="Start the worker automatically"
          description={isLocked('auto_start') ? 'Set by environment.' : 'Clicking Generate deploys the worker when it is not running, then sends the job once it is ready.'}
          disabled={!merged.configured || isLocked('auto_start')}
        />
        <Toggle
          checked={merged.stop_when_queue_empty}
          onChange={checked => update('stop_when_queue_empty', checked)}
          label="Stop the worker when the queue is empty"
          description={isLocked('stop_when_queue_empty') ? 'Set by environment.' : 'Stops about 30 seconds after the last queued generation (or Director project) finishes, instead of waiting the idle time below.'}
          disabled={!merged.configured || isLocked('stop_when_queue_empty')}
        />
        <Field locked={isLocked('idle_ttl_minutes')}
          label="Stop after idle (minutes)"
          type="number"
          value={merged.idle_ttl_minutes}
          onChange={value => update('idle_ttl_minutes', Number(value) || 1)}
          hint="The idle clock starts once the worker is ready and pauses while a generation runs."
        />
      </section>

      {/* ── dstack connection ────────────────────────────────────── */}
      <section className="space-y-3 rounded-lg border border-border bg-bg-tertiary p-3">
        <div className="flex items-center gap-2">
          <PlugZap size={14} className="text-accent-blue" />
          <div className="text-xs font-medium text-text-primary">dstack access</div>
        </div>
        <Field locked={isLocked('server_url')} label="Server URL" value={merged.server_url} onChange={value => update('server_url', value)} placeholder="http://127.0.0.1:3000" />
        <Field locked={isLocked('project')} label="Project" value={merged.project} onChange={value => update('project', value)} placeholder="main" />
        <Field locked={isLocked('token')}
          label="Token"
          type="password"
          value={draft.token ?? ''}
          onChange={value => update('token', value)}
          placeholder={settings?.token_set ? `Stored (${settings.token}) · type to replace` : 'dstack user token'}
          hint="Stored on this computer only, in app/settings."
        />
        <div className="grid grid-cols-2 gap-2">
          <ActionButton
            onClick={importDstack}
            disabled={busy !== null || ['server_url', 'project', 'token'].every(isLocked)}
          >
            <Download size={11} /> Import ~/.dstack
          </ActionButton>
          <ActionButton onClick={testConnection} disabled={busy !== null || !merged.configured || dirty}>
            <PlugZap size={11} /> {busy === 'test' ? 'Testing…' : 'Test connection'}
          </ActionButton>
        </div>
        {connection && (
          <div className="text-[9px] leading-relaxed text-text-muted">
            dstack {connection.server_version} · fleets: {connection.fleets.join(', ') || 'none'}
          </div>
        )}
      </section>

      {/* ── Worker resources ─────────────────────────────────────── */}
      <section className="space-y-3 rounded-lg border border-border bg-bg-tertiary p-3">
        <div className="flex items-center gap-2">
          <HardDrive size={14} className="text-accent-blue" />
          <div className="text-xs font-medium text-text-primary">GPU and limits</div>
        </div>
        <Toggle
          checked={merged.manage_fleet}
          onChange={checked => update('manage_fleet', checked)}
          label="Dedicated Maestro fleet"
          description={isLocked('manage_fleet') ? 'Set by environment.' : 'Maestro creates and updates this dstack fleet with the GPU and price limits below. Other fleets keep their own caps.'}
          disabled={isLocked('manage_fleet')}
        />
        {merged.manage_fleet ? (
          <Field locked={isLocked('fleet')} label="Fleet name" value={merged.fleet} onChange={value => update('fleet', value)} placeholder="maestro" />
        ) : connection?.fleets.length ? (
          <label className="block space-y-1">
            <span className="text-[10px] text-text-secondary">Fleet</span>
            <select
              disabled={isLocked('fleet')}
              value={merged.fleet}
              onChange={event => update('fleet', event.target.value)}
              className="w-full rounded-md border border-border bg-bg-primary px-2 py-1.5 text-[11px] text-text-primary focus:border-accent-blue focus:outline-none"
            >
              <option value="">Any matching fleet</option>
              {connection.fleets.map(fleet => <option key={fleet} value={fleet}>{fleet}</option>)}
            </select>
          </label>
        ) : (
          <Field locked={isLocked('fleet')} label="Fleet" value={merged.fleet} onChange={value => update('fleet', value)} hint="Test the connection to pick from your fleets." />
        )}
        <div className="grid grid-cols-2 gap-2">
          <Field locked={isLocked('gpu')} label="GPU" value={merged.gpu} onChange={value => update('gpu', value)} placeholder="48GB.." />
          <Field locked={isLocked('max_price')} label="Max price ($/h)" type="number" value={merged.max_price} onChange={value => update('max_price', Number(value))} />
          <Field locked={isLocked('max_duration')} label="Max duration" value={merged.max_duration} onChange={value => update('max_duration', value)} placeholder="4h" />
          <Field locked={isLocked('disk')} label="Disk" value={merged.disk} onChange={value => update('disk', value)} placeholder="200GB.." />
          <Field locked={isLocked('memory')} label="RAM" value={merged.memory} onChange={value => update('memory', value)} placeholder="64GB.." />
          <Field locked={isLocked('cpu')} label="vCPUs" value={merged.cpu} onChange={value => update('cpu', value)} placeholder="8.." />
          <Field locked={isLocked('min_download_mbps')} label="Min Vast.ai download (Mb/s)" type="number" value={merged.min_download_mbps} onChange={value => update('min_download_mbps', Number(value))} placeholder="800" />
        </div>
        <div className="text-[9px] leading-relaxed text-text-muted">
          Max duration and max price are enforced by dstack even if this computer is off.
        </div>

        <button
          type="button"
          onClick={() => setShowAdvanced(value => !value)}
          className="flex items-center gap-1 text-[10px] text-text-secondary hover:text-text-primary"
        >
          {showAdvanced ? <ChevronDown size={11} /> : <ChevronRight size={11} />} Advanced
        </button>
        {showAdvanced && (
          <div className="space-y-2">
            <Field locked={isLocked('repo_url')} label="Repository" value={merged.repo_url} onChange={value => update('repo_url', value)} />
            <Field locked={isLocked('repo_ref')} label="Branch or tag" value={merged.repo_ref} onChange={value => update('repo_ref', value)} />
            <Field locked={isLocked('run_name')} label="Run name" value={merged.run_name} onChange={value => update('run_name', value)} />
            <Field locked={isLocked('image')} label="Container image" value={merged.image} onChange={value => update('image', value)} />
            <Field locked={isLocked('registry_username')}
              label="Registry username"
              value={merged.registry_username}
              onChange={value => update('registry_username', value)}
              hint="Only while the GHCR image is private; use a token with read:packages."
            />
            {secretField('registry_token', 'Registry token', merged.registry_token_set)}
            <Field locked={isLocked('spot_policy')} label="Spot policy" value={merged.spot_policy} onChange={value => update('spot_policy', value)} placeholder="on-demand | spot | auto" />
            <Field locked={isLocked('backends')}
              label="Backends"
              value={merged.backends.join(', ')}
              onChange={value => update('backends', value.split(',').map(part => part.trim()).filter(Boolean))}
            />
          </div>
        )}
      </section>

      {/* ── R2 checkpoint cache ──────────────────────────────────── */}
      <section className="space-y-3 rounded-lg border border-border bg-bg-tertiary p-3">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <HardDrive size={14} className="text-accent-blue" />
            <div className="text-xs font-medium text-text-primary">Cloudflare R2 storage</div>
          </div>
          <span className={`text-[9px] ${merged.r2_configured ? 'text-emerald-300' : 'text-text-muted'}`}>
            {merged.r2_configured ? 'Enabled' : 'Off'}
          </span>
        </div>
        <div className="text-[9px] leading-relaxed text-text-muted">
          Your Cloudflare R2 bucket carries model weights (cached between boots), reference assets (uploaded once, by content hash) and generated videos (kept under outputs/ as a remote copy of the gallery). Credentials reach the worker as dstack project secrets, never in the run spec.
        </div>
        <Field locked={isLocked('r2_account_id')} label="Account ID" value={merged.r2_account_id} onChange={value => update('r2_account_id', value)} />
        {secretField('r2_access_key_id', 'Access key ID', merged.r2_access_key_id_set)}
        {secretField('r2_secret_access_key', 'Secret access key', merged.r2_secret_access_key_set)}
        {secretField('cf_api_token', 'Cloudflare API token (per-pod keys)', merged.cf_api_token_set)}
        <Toggle
          checked={merged.untrusted_hosts}
          onChange={checked => update('untrusted_hosts', checked)}
          label="Pods may run on untrusted hosts"
          description={isLocked('untrusted_hosts') ? 'Set by environment.' : 'For community or marketplace GPUs. Pods can only read the model cache and their inputs, write their own outputs, never update the cache, and verify every cached file against its hash. Needs the Cloudflare API token; cache warming is disabled.'}
          disabled={isLocked('untrusted_hosts')}
        />
        <div className={`text-[9px] leading-relaxed ${status?.pod_r2_keys ? 'text-emerald-300' : 'text-amber-300'}`}>
          {status?.pod_r2_keys
            ? `Each pod gets its own key, limited to this bucket and revoked when it stops (${status.pod_r2_keys_active ?? 0} active). The keys above never leave this computer.`
            : 'Without a Cloudflare API token (permission: Account API Tokens Edit), pods receive the keys above. Add one so each pod gets its own revocable key.'}
        </div>
        <div className="grid grid-cols-2 gap-2">
          <Field locked={isLocked('r2_bucket')} label="Bucket" value={merged.r2_bucket} onChange={value => update('r2_bucket', value)} />
          <Field locked={isLocked('r2_prefix')} label="Prefix" value={merged.r2_prefix} onChange={value => update('r2_prefix', value)} placeholder="maestro" />
        </div>
      </section>

      {settings?.r2_configured && merged.configured && <R2CachePanel />}

      <div className="flex items-center gap-2">
        <ActionButton onClick={save} disabled={!dirty || busy !== null} tone="primary">
          <Save size={11} /> {busy === 'save' ? 'Saving…' : 'Save settings'}
        </ActionButton>
        {dirty && (
          <button type="button" onClick={() => setDraft({})} className="text-[10px] text-text-muted hover:text-text-primary">
            Discard
          </button>
        )}
      </div>

    </div>
  )
}
