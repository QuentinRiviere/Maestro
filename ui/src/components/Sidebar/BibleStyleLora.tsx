import { useCallback, useEffect, useState, type ReactNode } from 'react'
import { Image as ImageIcon, Loader2, Sparkles, Trash2, XCircle } from 'lucide-react'
import {
  cancelStyleLora, createStyleLora, deleteStyleLora, fetchStyleLoraDataset, fetchStyleLoras, resampleStyleLora,
  styleLoraSampleUrl, trainStyleLora, updateStyleLora, type StyleLora, type StyleLoraDatasetPreview, type StyleLoraRequest,
} from '../../api/directorBibles'

const input = 'w-full bg-bg-tertiary border border-border rounded px-2 py-1 text-[11px] text-text-primary focus:outline-none focus:border-accent-blue'
const label = 'text-[9px] text-text-muted uppercase tracking-wider block mb-0.5'
const button = 'flex items-center gap-1 rounded border border-border px-2 py-0.5 text-[10px] text-text-secondary hover:text-text-primary disabled:opacity-40'

type Ask = (request: { title: string; message: ReactNode; confirmLabel?: string; danger?: boolean; run: () => void | Promise<void> }) => void

const LIVE: StyleLora['status'][] = ['queued', 'preparing', 'training', 'sampling']
const STATUS_CLASS: Record<string, string> = {
  queued: 'text-text-muted', preparing: 'text-accent-blue', training: 'text-accent-blue', sampling: 'text-accent-blue',
  done: 'text-emerald-300', failed: 'text-chip-red',
}

const isTraining = (status: StyleLora['status']) => LIVE.includes(status)

function LoraCard({ bibleId, lora, ask, onChange, onDeleted }: {
  bibleId: string
  lora: StyleLora
  ask: Ask
  onChange: (lora: StyleLora) => void
  onDeleted: (loras: StyleLora[]) => void
}) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [scale, setScale] = useState(lora.scale)
  const live = isTraining(lora.status)
  const act = async (call: () => Promise<void>) => {
    setBusy(true)
    setError(null)
    try { await call() } catch (e) { setError(e instanceof Error ? e.message : 'Failed') } finally { setBusy(false) }
  }
  const percent = lora.total_steps ? Math.round(100 * lora.step / lora.total_steps) : 0
  const dataset = lora.dataset

  return (
    <div className="rounded-lg border border-border bg-bg-tertiary/30 p-2.5 space-y-2">
      <div className="flex items-center gap-2">
        <span className="min-w-0 flex-1 truncate text-[11px] font-medium text-text-primary" title={lora.file}>{lora.name}</span>
        <span className={`shrink-0 flex items-center gap-1 text-[10px] ${STATUS_CLASS[lora.status] ?? ''}`} title={lora.error || lora.progress}>
          {live && <Loader2 size={10} className="animate-spin" />} {lora.status}
        </span>
        <label className="shrink-0 flex items-center gap-1 text-[10px] text-text-secondary"
          title="The enabled LoRA rides on every shot image of a run that uses this bible (director_style_lora: auto)">
          <input type="checkbox" className="w-3 h-3" checked={lora.enabled} disabled={busy || lora.status !== 'done'}
            onChange={e => void act(async () => onChange(await updateStyleLora(bibleId, lora.id, { enabled: e.target.checked })))} /> Use
        </label>
        {lora.status === 'failed' && (
          <button className={button} disabled={busy} title="Train again (the dataset is rebuilt)"
            onClick={() => void act(async () => onChange(await trainStyleLora(bibleId, lora.id)))}>
            <Sparkles size={10} /> Retry
          </button>
        )}
        {live && lora.status !== 'queued' && (
          <button className={button} disabled={busy} title="Stop the training (nothing is kept)"
            onClick={() => ask({
              title: `Stop training "${lora.name}"?`, confirmLabel: 'Stop', danger: true,
              message: 'The steps done so far are lost.',
              run: () => act(async () => onChange(await cancelStyleLora(bibleId, lora.id))),
            })}>
            <XCircle size={10} /> Stop
          </button>
        )}
        {lora.status === 'done' && (
          <>
            <button className={button} disabled={busy} title="Render the three samples again at the slider's weight with the Director's sampler (turbo, 6 steps)"
              onClick={() => void act(async () => onChange(await resampleStyleLora(bibleId, lora.id, { sampler: 'turbo', scale })))}>
              <ImageIcon size={10} /> Samples
            </button>
            <button className={button} disabled={busy} title="Render the samples with the plain model (40 steps, CFG 4): tells a LoRA problem from a turbo-stack problem"
              onClick={() => void act(async () => onChange(await resampleStyleLora(bibleId, lora.id, { sampler: 'base', scale })))}>
              <ImageIcon size={10} /> Base
            </button>
          </>
        )}
        <button className={button} disabled={busy || live} title="Delete the LoRA, its samples and its registry file"
          onClick={() => ask({
            title: `Delete "${lora.name}"?`, confirmLabel: 'Delete', danger: true,
            message: 'The adapter file is removed from the LoRA registry too.',
            run: () => act(async () => onDeleted(await deleteStyleLora(bibleId, lora.id))),
          })}>
          <Trash2 size={10} />
        </button>
      </div>
      {live && (
        <div className="space-y-1">
          <div className="h-1 w-full overflow-hidden rounded bg-bg-tertiary">
            <div className="h-full bg-accent-blue transition-all" style={{ width: `${percent}%` }} />
          </div>
          <p className="text-[10px] text-text-secondary">{lora.progress || lora.status}</p>
        </div>
      )}
      {lora.error && <p role="alert" className="text-[10px] text-chip-red">{lora.error}</p>}
      {error && <p role="alert" className="text-[10px] text-chip-red">{error}</p>}
      <p className="text-[10px] text-text-muted">
        Trigger <code className="text-text-secondary">{lora.trigger}</code>
        {dataset && <> · {dataset.images} frames from {dataset.references.length} reference{dataset.references.length === 1 ? '' : 's'}
          {dataset.excluded_close_ups ? `, ${dataset.excluded_close_ups} close-ups left out` : ''}</>}
        {lora.config?.steps && <> · {lora.config.steps} steps, rank {lora.config.rank}, {lora.config.resolution}px</>}
        {lora.training?.final_loss !== undefined && lora.training?.final_loss !== null && <> · loss {lora.training.final_loss.toFixed(3)}</>}
        {lora.training?.seconds ? <> · {Math.round(lora.training.seconds / 60)} min</> : null}
      </p>
      {lora.status === 'done' && (
        <label className="flex items-center gap-2 text-[10px] text-text-secondary">
          <span className="shrink-0">Weight {scale.toFixed(2)}</span>
          <input type="range" min={0} max={1.5} step={0.05} value={scale} className="flex-1" aria-label="LoRA weight"
            onChange={e => setScale(Number(e.target.value))}
            onMouseUp={() => { if (scale !== lora.scale) void act(async () => onChange(await updateStyleLora(bibleId, lora.id, { scale }))) }}
            onTouchEnd={() => { if (scale !== lora.scale) void act(async () => onChange(await updateStyleLora(bibleId, lora.id, { scale }))) }}
            onKeyUp={() => { if (scale !== lora.scale) void act(async () => onChange(await updateStyleLora(bibleId, lora.id, { scale }))) }} />
        </label>
      )}
      {lora.samples.length > 0 && (
        <div>
          <span className={label}>Samples (an invented cast: nobody here should look like the references){lora.samples_note ? ` · ${lora.samples_note}` : ''}</span>
          <div className="flex gap-1.5">
            {lora.samples.map((name, index) => (
              <a key={name} href={styleLoraSampleUrl(bibleId, lora.id, index + 1)} target="_blank" rel="noreferrer">
                <img loading="lazy" alt={`Sample ${index + 1} of ${lora.name}`} width={160} height={90}
                  className="h-[90px] w-[160px] rounded bg-black object-cover"
                  src={`${styleLoraSampleUrl(bibleId, lora.id, index + 1)}?v=${encodeURIComponent(lora.updated_at)}`} />
              </a>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

/** The style LoRAs of a bible: a training form over the references' frames and the trained adapters. */
export function BibleStyleLora({ bibleId, ask }: { bibleId: string; ask: Ask }) {
  const [loras, setLoras] = useState<StyleLora[]>([])
  const [error, setError] = useState<string | null>(null)
  const [preview, setPreview] = useState<StyleLoraDatasetPreview | null>(null)
  const [showForm, setShowForm] = useState(false)
  const [starting, setStarting] = useState(false)
  const [form, setForm] = useState<StyleLoraRequest>({
    steps: 1500, rank: 16, resolution: 768, exclude_close_ups: true, caption_mode: 'content', targets: 'attention',
  })

  const refresh = useCallback(() => fetchStyleLoras(bibleId).then(setLoras)
    .catch(e => setError(e instanceof Error ? e.message : 'Could not load the style LoRAs')), [bibleId])
  useEffect(() => { void refresh() }, [refresh])

  const running = loras.some(l => isTraining(l.status))
  useEffect(() => {
    if (!running) return
    const timer = window.setInterval(() => { void refresh() }, 4000)
    return () => window.clearInterval(timer)
  }, [running, refresh])

  // The dataset preview follows the form's filters.
  useEffect(() => {
    if (!showForm) return
    let alive = true
    fetchStyleLoraDataset(bibleId, { exclude_close_ups: form.exclude_close_ups, caption_mode: form.caption_mode, trigger: form.trigger })
      .then(p => { if (alive) setPreview(p) })
      .catch(e => { if (alive) setError(e instanceof Error ? e.message : 'Could not read the frames') })
    return () => { alive = false }
  }, [bibleId, showForm, form.exclude_close_ups, form.caption_mode, form.trigger])

  const start = async () => {
    setStarting(true)
    setError(null)
    try {
      const lora = await createStyleLora(bibleId, form)
      setLoras(current => [...current, lora])
      setShowForm(false)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not start the training')
    } finally {
      setStarting(false)
    }
  }
  const enough = !!preview && preview.available >= preview.minimum

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <div>
          <span className={label}>Style LoRA</span>
          <p className="text-[10px] text-text-muted">
            An adapter for the image model trained on the references' frames: grade, light and framing, with the
            people under-represented. The enabled one rides on every shot image of a run that uses this bible.
          </p>
        </div>
        <button className={button} disabled={showForm || running} onClick={() => setShowForm(true)}>
          <Sparkles size={10} /> Train a style LoRA
        </button>
      </div>
      {error && <p role="alert" className="text-[10px] text-chip-red">{error}</p>}
      {showForm && (
        <div className="rounded-lg border border-border bg-bg-tertiary/30 p-2.5 space-y-2">
          <div className="grid grid-cols-2 gap-2 lg:grid-cols-4">
            <label className="block"><span className={label}>Trigger word</span>
              <input className={input} value={form.trigger ?? preview?.trigger ?? ''} placeholder={preview?.trigger}
                onChange={e => setForm({ ...form, trigger: e.target.value })} /></label>
            <label className="block"><span className={label}>Steps</span>
              <input className={input} type="number" min={50} max={20000} value={form.steps}
                onChange={e => setForm({ ...form, steps: Number(e.target.value) })} /></label>
            <label className="block"><span className={label}>Rank</span>
              <input className={input} type="number" min={4} max={128} value={form.rank}
                onChange={e => setForm({ ...form, rank: Number(e.target.value) })} /></label>
            <label className="block"><span className={label}>Resolution (long side)</span>
              <input className={input} type="number" min={384} max={1280} step={32} value={form.resolution}
                onChange={e => setForm({ ...form, resolution: Number(e.target.value) })} /></label>
          </div>
          <div className="flex flex-wrap gap-3 text-[10px] text-text-secondary">
            <label className="flex items-center gap-1" title="Close-ups and inserts carry the faces: leaving them out keeps the adapter on the look, not the actors">
              <input type="checkbox" className="h-3 w-3" checked={form.exclude_close_ups !== false}
                onChange={e => setForm({ ...form, exclude_close_ups: e.target.checked })} /> Leave close-ups out
            </label>
            <label className="flex items-center gap-1" title="Content: trigger + shot size, subject and action (the look is what the trigger learns). Full: the whole reverse prompt">
              Captions
              <select className="bg-bg-tertiary border border-border rounded px-1 py-0.5 text-[10px]" value={form.caption_mode}
                onChange={e => setForm({ ...form, caption_mode: e.target.value as 'content' | 'full' })}>
                <option value="content">content only</option>
                <option value="full">full reverse prompt</option>
              </select>
            </label>
            <label className="flex items-center gap-1" title="Attention only is the usual style adapter; all adds the MLPs (stronger, more memorising)">
              Layers
              <select className="bg-bg-tertiary border border-border rounded px-1 py-0.5 text-[10px]" value={form.targets}
                onChange={e => setForm({ ...form, targets: e.target.value as 'attention' | 'all' })}>
                <option value="attention">attention</option>
                <option value="all">attention + MLP</option>
              </select>
            </label>
          </div>
          {preview ? (
            <div className="text-[10px] text-text-secondary space-y-1">
              <p>
                <span className={enough ? 'text-emerald-300' : 'text-chip-red'}>{preview.available} frames</span>
                {' '}from {preview.references.map(r => `${r.label} (${r.used}/${r.frames})`).join(', ') || 'no analysed reference'}
                {preview.excluded_close_ups ? ` · ${preview.excluded_close_ups} close-ups left out` : ''}
                {preview.excluded_credits ? ` · ${preview.excluded_credits} credits frames left out` : ''}
                {!enough && ` · at least ${preview.minimum} needed: analyse more references or run their frame prompts`}
              </p>
              {preview.captions.length > 0 && (
                <ul className="list-disc pl-4 text-text-muted">
                  {preview.captions.slice(0, 3).map((caption, index) => <li key={index}>{caption}</li>)}
                </ul>
              )}
            </div>
          ) : <p className="text-[10px] text-text-muted">Counting the frames…</p>}
          <div className="flex gap-2">
            <button className={button} disabled={!enough || starting} onClick={() => void start()}>
              {starting ? <Loader2 size={10} className="animate-spin" /> : <Sparkles size={10} />} Start training on the Spark
            </button>
            <button className={button} disabled={starting} onClick={() => setShowForm(false)}>Cancel</button>
          </div>
        </div>
      )}
      {loras.map(lora => (
        <LoraCard key={lora.id} bibleId={bibleId} lora={lora} ask={ask}
          onChange={next => setLoras(current => current.map(l => (l.id === next.id ? next : l)))}
          onDeleted={setLoras} />
      ))}
    </div>
  )
}
