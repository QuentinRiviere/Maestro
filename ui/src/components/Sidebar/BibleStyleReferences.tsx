import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { Film, Image as ImageIcon, LayoutList, Link, Loader2, RefreshCw, Trash2, Upload } from 'lucide-react'
import { StyleReferenceShots } from './StyleReferenceShots'
import {
  analyseStyleReference, deleteStyleReference, fetchStyleReferenceFetches, fetchStyleReferencesFromUrls,
  fetchStyleReferences, parseVideoLinks, startStyleReferenceFrames, STYLE_REFERENCE_ACCEPT, STYLE_REFERENCE_MAX_BYTES,
  styleReferenceVideoUrl, updateStyleReference, uploadStyleReference,
  type StyleReference, type StyleReferenceFetch, type StyleReferencePatch,
} from '../../api/directorBibles'
import {
  formatSeconds, formatShotMix, isAnalysing, pacingBars, parseSceneTags, rulesFromText, rulesToText,
} from '../../lib/styleReference'

const input = 'w-full bg-bg-tertiary border border-border rounded px-2 py-1 text-[11px] text-text-primary focus:outline-none focus:border-accent-blue'
const label = 'text-[9px] text-text-muted uppercase tracking-wider block mb-0.5'
const button = 'flex items-center gap-1 rounded border border-border px-2 py-0.5 text-[10px] text-text-secondary hover:text-text-primary disabled:opacity-40'

type Ask = (request: { title: string; message: ReactNode; confirmLabel?: string; danger?: boolean; run: () => void | Promise<void> }) => void

const STATUS_CLASS: Record<string, string> = {
  queued: 'text-text-muted', analysing: 'text-accent-blue', done: 'text-emerald-300', failed: 'text-chip-red',
}

function Stat({ title, value }: { title: string; value: ReactNode }) {
  return (
    <div className="rounded bg-bg-tertiary px-1.5 py-1">
      <span className="block text-[8px] uppercase tracking-wider text-text-muted">{title}</span>
      <span className="text-[11px] text-text-primary">{value}</span>
    </div>
  )
}

function ReferenceCard({ bibleId, reference, ask, onChange, onDeleted }: {
  bibleId: string
  reference: StyleReference
  ask: Ask
  onChange: (reference: StyleReference) => void
  onDeleted: (references: StyleReference[]) => void
}) {
  const [labelText, setLabelText] = useState(reference.label)
  const [showShots, setShowShots] = useState(false)
  const framesRunning = reference.frames_status === 'queued' || reference.frames_status === 'running'
  const [tagsText, setTagsText] = useState(reference.scene_tags.join(', '))
  const [rulesText, setRulesText] = useState(rulesToText(reference.grammar?.rules))
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showSummary, setShowSummary] = useState(false)
  const storedRules = rulesToText(reference.grammar?.rules)
  // Keep local edits unless the server changed what they edit (a new analysis, a save).
  const [seen, setSeen] = useState({ rules: storedRules, label: reference.label })
  if (seen.rules !== storedRules || seen.label !== reference.label) {
    setSeen({ rules: storedRules, label: reference.label })
    if (seen.rules !== storedRules) setRulesText(storedRules)
    if (seen.label !== reference.label) setLabelText(reference.label)
  }

  const stats = reference.stats
  const grammar = reference.grammar
  const analysing = isAnalysing(reference.status)
  const bars = pacingBars(stats?.pacing)

  const act = async (call: () => Promise<void>) => {
    setBusy(true)
    setError(null)
    try { await call() } catch (e) { setError(e instanceof Error ? e.message : 'Failed') } finally { setBusy(false) }
  }
  const patch = (body: StyleReferencePatch) => act(async () => onChange(await updateStyleReference(bibleId, reference.id, body)))

  return (
    <div className="rounded-lg border border-border bg-bg-tertiary/30 p-2.5 space-y-2">
      <div className="flex items-center gap-2">
        <input className={`${input} font-medium`} value={labelText} aria-label="Reference label"
          onChange={e => setLabelText(e.target.value)}
          onBlur={() => { if (labelText.trim() && labelText.trim() !== reference.label) void patch({ label: labelText.trim() }) }} />
        <span className={`shrink-0 flex items-center gap-1 text-[10px] ${STATUS_CLASS[reference.status] ?? ''}`}
          title={reference.error || reference.progress}>
          {analysing && <Loader2 size={10} className="animate-spin" />} {reference.status}
        </span>
        <label className="shrink-0 flex items-center gap-1 text-[10px] text-text-secondary"
          title="An enabled, analysed reference guides the storyboard of every project made from this bible (unless the run picks another)">
          <input type="checkbox" className="w-3 h-3" checked={reference.enabled} disabled={busy}
            onChange={e => void patch({ enabled: e.target.checked })} /> Use
        </label>
        <label className="shrink-0 flex items-center gap-1 text-[10px] text-text-secondary"
          title="Read captions, subtitles, logos and other on-screen graphics as part of the staging (re-analyse to apply); off, they are left out">
          <input type="checkbox" className="w-3 h-3" checked={reference.graphics === 'include'} disabled={busy}
            onChange={e => void patch({ graphics: e.target.checked ? 'include' : 'ignore' })} /> Graphics
        </label>
        <button className={button} disabled={!reference.described_shots && !analysing}
          title="Browse the shots described so far, with their frames" onClick={() => setShowShots(true)}>
          <LayoutList size={10} /> Shots{reference.described_shots ? ` (${reference.described_shots})` : ''}
        </button>
        {showShots && <StyleReferenceShots bibleId={bibleId} reference={reference} onClose={() => setShowShots(false)} />}
        <button className={button} disabled={busy || analysing || reference.status !== 'done' || framesRunning}
          title="Describe the first and last frame of the shots as image prompts (runs after each analysis; click to run again)"
          onClick={() => act(async () => onChange(await startStyleReferenceFrames(bibleId, reference.id)))}>
          {framesRunning ? <Loader2 size={10} className="animate-spin" /> : <ImageIcon size={10} />}
          {framesRunning ? reference.frames_progress || 'Frame prompts…' : `Frame prompts${reference.frames_count ? ` (${reference.frames_count})` : ''}`}
        </button>
        <button className={button} disabled={busy || analysing} title="Analyse the video again (replaces the grammar and its rules)"
          onClick={() => ask({
            title: `Analyse "${reference.label}" again?`, confirmLabel: 'Re-analyse',
            message: 'The video is sent to the video model again (a few cents per minute). The new grammar replaces the current one, including edited rules.',
            run: () => act(async () => onChange(await analyseStyleReference(bibleId, reference.id))),
          })}>
          <RefreshCw size={10} /> Re-analyse
        </button>
        <button aria-label="Delete reference" className="shrink-0 text-text-muted hover:text-chip-red" disabled={busy}
          onClick={() => ask({
            title: `Delete "${reference.label}"?`, confirmLabel: 'Delete', danger: true,
            message: 'The video and its analysis are deleted. Projects already planned keep their storyboards.',
            run: () => act(async () => onDeleted(await deleteStyleReference(bibleId, reference.id))),
          })}><Trash2 size={12} /></button>
      </div>

      <div className="flex flex-col md:flex-row gap-2.5">
        <video className="w-full md:w-56 shrink-0 rounded bg-black aspect-video" controls preload="metadata"
          src={styleReferenceVideoUrl(bibleId, reference.id)} aria-label={`${reference.label} video`} />
        <div className="min-w-0 flex-1 space-y-1.5">
          {stats ? (
            <div className="grid grid-cols-3 sm:grid-cols-5 gap-1">
              <Stat title="Length" value={formatSeconds(stats.duration_sec)} />
              <Stat title="Shots" value={stats.shot_count} />
              <Stat title="Avg shot" value={formatSeconds(stats.asl_sec)} />
              <Stat title="Median" value={formatSeconds(stats.median_sec)} />
              <Stat title="Cuts/min" value={stats.cuts_per_minute ?? '–'} />
            </div>
          ) : <p className="text-[10px] text-text-muted">{analysing ? reference.progress || 'Measuring cuts…' : 'Not analysed yet.'}</p>}
          {stats?.distribution && (
            <p className="text-[10px] text-text-secondary" title="Shot lengths">
              {stats.distribution.map(d => `${d.bucket} ${Math.round(d.percent)}%`).join(' · ')}
            </p>
          )}
          {bars.length > 1 && (
            <div className="flex items-end gap-px h-6" title="Pace per 10 s window (taller = shorter shots)" aria-label="Pacing curve">
              {bars.map((height, index) => (
                <span key={index} className="flex-1 rounded-sm bg-accent-blue/60" style={{ height: `${Math.round(height * 100)}%` }} />
              ))}
            </div>
          )}
          {(grammar?.shot_mix || stats?.shot_mix) && (
            <p className="text-[10px] text-text-secondary"><span className="text-text-muted">Shot mix: </span>
              {formatShotMix(grammar?.shot_mix ?? stats?.shot_mix)}</p>
          )}
          {grammar?.rhythm.pace_with_emotion && (
            <p className="text-[10px] text-text-secondary"><span className="text-text-muted">Pace: </span>{grammar.rhythm.pace_with_emotion}</p>
          )}
          {analysing && reference.progress && stats && <p className="text-[10px] text-accent-blue">{reference.progress}</p>}
          {reference.error && <p className="text-[10px] text-chip-red">{reference.error}</p>}
          {reference.frames_error && <p className="text-[10px] text-chip-red">Frame prompts: {reference.frames_error}</p>}
          {reference.camera_profile && (
            <p className="text-[10px] text-text-secondary" title="What the frame prompts agree on">
              <span className="text-text-muted">Camera: </span>
              {[reference.camera_profile.lenses_mm?.length ? `${[...reference.camera_profile.lenses_mm].sort((a, b) => a - b).join('/')} mm` : '',
                reference.camera_profile.aperture, reference.camera_profile.format, reference.camera_profile.aspect_ratio].filter(Boolean).join(', ')}
              {reference.camera_profile.grade ? ` · ${reference.camera_profile.grade}` : ''}
            </p>
          )}
          {reference.cost_usd != null && (
            <p className="text-[9px] text-text-muted">Analysis: ${reference.cost_usd.toFixed(3)}{reference.model ? ` · ${reference.model}` : ''}
              {reference.described_shots ? ` · ${reference.described_shots} shots described` : ''}</p>
          )}
        </div>
      </div>

      <label className="block"><span className={label}>Scene tags (the kind of scenes it guides{reference.tags_auto ? ', set by the analysis' : ''}; empty = every scene)</span>
        <input className={input} value={tagsText} placeholder="argument, romance, tension" aria-label="Scene tags"
          onChange={e => setTagsText(e.target.value)}
          onBlur={() => {
            const tags = parseSceneTags(tagsText)
            if (tags.join(',') !== reference.scene_tags.join(',')) void patch({ scene_tags: tags })
            setTagsText(tags.join(', '))
          }} />
      </label>

      {grammar && (
        <div>
          <span className={label}>Rules for the storyboard planner (one per line)</span>
          <textarea className={input} rows={Math.min(10, Math.max(4, rulesText.split('\n').length + 1))} value={rulesText}
            aria-label="Planner rules" onChange={e => setRulesText(e.target.value)} />
          <div className="mt-1 flex items-center gap-2">
            <button className={button} disabled={busy || rulesText === storedRules || !rulesFromText(rulesText).length}
              onClick={() => void patch({ rules: rulesFromText(rulesText) })}>Save rules</button>
            {rulesText !== storedRules && (
              <button className="text-[10px] text-text-muted hover:text-text-primary" onClick={() => setRulesText(storedRules)}>Undo</button>
            )}
            {grammar.rules_edited && <span className="text-[9px] text-text-muted">edited</span>}
            {reference.summary && (
              <button className="ml-auto text-[10px] text-accent-blue hover:text-accent-blue/80" onClick={() => setShowSummary(v => !v)}>
                {showSummary ? 'Hide summary' : 'Show summary'}
              </button>
            )}
          </div>
        </div>
      )}
      {showSummary && reference.summary && (
        <pre className="whitespace-pre-wrap rounded bg-bg-tertiary p-2 text-[10px] leading-relaxed text-text-secondary font-sans">
          {reference.summary}
        </pre>
      )}
      {error && <p className="text-[10px] text-chip-red">{error}</p>}
    </div>
  )
}

/** Reference videos of a bible: their staging, editing and rhythm guide the storyboards of its projects. */
export function BibleStyleReferences({ bibleId, ask }: { bibleId: string; ask: Ask }) {
  const [references, setReferences] = useState<StyleReference[]>([])
  const [loaded, setLoaded] = useState(false)
  const [file, setFile] = useState<File | null>(null)
  const newLabel = ''
  const newTags = ''
  const [newGraphics, setNewGraphics] = useState(false)
  const [newUrl, setNewUrl] = useState('')
  const [fetches, setFetches] = useState<StyleReferenceFetch[]>([])
  const [fetching, setFetching] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const picker = useRef<HTMLInputElement>(null)

  const refresh = useCallback(() => fetchStyleReferences(bibleId)
    .then(list => { setReferences(list); setLoaded(true) })
    .catch(e => setError(e instanceof Error ? e.message : 'Could not load the style references')), [bibleId])
  useEffect(() => { void refresh() }, [refresh])

  const refreshFetches = useCallback(() => fetchStyleReferenceFetches(bibleId).then(setFetches).catch(() => {}),
    [bibleId])
  useEffect(() => { void refreshFetches() }, [refreshFetches])

  const downloading = fetches.some(f => f.status === 'queued' || f.status === 'downloading')
  const running = references.some(r => isAnalysing(r.status) || r.frames_status === 'queued' || r.frames_status === 'running')
    || downloading
  useEffect(() => {
    if (!running) return
    const timer = window.setInterval(() => { void refresh(); if (downloading) void refreshFetches() }, 3000)
    return () => window.clearInterval(timer)
  }, [running, downloading, refresh, refreshFetches])

  const links = parseVideoLinks(newUrl)
  const fetchLink = async () => {
    if (!links.length) return
    setFetching(true)
    setError(null)
    try {
      const queued = await fetchStyleReferencesFromUrls(bibleId, links, newLabel.trim(), parseSceneTags(newTags),
        newGraphics ? 'include' : 'ignore')
      const ids = new Set(queued.map(f => f.id))
      setFetches(list => [...list.filter(f => !ids.has(f.id)), ...queued])
      setNewUrl('')
      setNewGraphics(false)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not fetch the link')
    } finally {
      setFetching(false)
    }
  }

  const upload = async () => {
    if (!file) return
    if (file.size > STYLE_REFERENCE_MAX_BYTES) { setError('Video too large (max 4 GB)'); return }
    setUploading(true)
    setError(null)
    try {
      const created = await uploadStyleReference(bibleId, file, newLabel.trim(), parseSceneTags(newTags),
        newGraphics ? 'include' : 'ignore')
      setReferences(list => [...list.filter(r => r.id !== created.id), created])
      setFile(null)
      setNewGraphics(false)
      if (picker.current) picker.current.value = ''
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Upload failed')
    } finally {
      setUploading(false)
    }
  }
  const replace = (next: StyleReference) => setReferences(list => list.map(r => (r.id === next.id ? next : r)))

  return (
    <div className="space-y-2.5 max-w-4xl">
      <p className="text-[10px] text-text-secondary leading-relaxed">
        Upload a scene (or paste links) whose staging, editing and rhythm this series should follow. Its cuts are
        measured exactly here; the video model then describes every shot, writes transferable rules (never the
        reference's story or characters) and names the reference and the kinds of scenes it guides. Enabled
        references guide the storyboard of every project made from this bible.
      </p>
      <div className="rounded-lg border border-dashed border-border p-2.5 space-y-1.5">
        <div className="flex flex-wrap items-center gap-2">
          <input ref={picker} type="file" accept={STYLE_REFERENCE_ACCEPT} className="hidden" aria-label="Reference video"
            onChange={e => { const chosen = e.target.files?.[0] ?? null; setFile(chosen);  }} />
          <button className={button} onClick={() => picker.current?.click()} disabled={uploading}>
            <Film size={10} /> {file ? 'Change video' : 'Choose a video'}
          </button>
          <span className="min-w-0 truncate text-[10px] text-text-muted">
            {file ? `${file.name} · ${(file.size / 1024 / 1024).toFixed(1)} MB` : 'MP4, MOV, WebM or MKV, up to 4 GB (reduced to 480p here)'}
          </span>
        </div>
        <label className="flex items-center gap-1 text-[10px] text-text-secondary"
          title="Read captions, subtitles, logos and other on-screen graphics as part of the staging; off, the analyst leaves them out">
          <input type="checkbox" className="w-3 h-3" checked={newGraphics} onChange={e => setNewGraphics(e.target.checked)}
            aria-label="New reference graphics" /> Take on-screen graphics into account
        </label>
        <button onClick={() => void upload()} disabled={!file || uploading}
          className="px-3 py-1 rounded bg-accent-blue text-white text-[11px] disabled:opacity-40 flex items-center gap-1">
          {uploading ? <Loader2 size={10} className="animate-spin" /> : <Upload size={10} />} Upload and analyse
        </button>
        <label className="block">
          <span className={label}>Or links, one per line (YouTube…; fetched at 480p on the Spark, queued, then analysed)</span>
          <textarea className={`${input} h-16 resize-y font-mono`} value={newUrl} onChange={e => setNewUrl(e.target.value)}
            placeholder={'https://www.youtube.com/watch?v=…\nhttps://youtu.be/…'} aria-label="Reference video links" />
        </label>
        <button onClick={() => void fetchLink()} disabled={!links.length || fetching}
          className="px-3 py-1 rounded bg-accent-blue text-white text-[11px] disabled:opacity-40 flex items-center gap-1">
          {fetching ? <Loader2 size={10} className="animate-spin" /> : <Link size={10} />}
          {links.length > 1 ? `Fetch ${links.length} links and analyse` : 'Fetch and analyse'}
        </button>
        {fetches.filter(f => f.status !== 'done').map(f => (
          <p key={f.id} className={`text-[10px] ${f.status === 'failed' ? 'text-chip-red' : 'text-text-muted'}`}
            title={f.url}>
            {f.status !== 'failed' && <Loader2 size={10} className="inline animate-spin mr-1" />}
            {f.label || f.url}: {f.status === 'failed' ? f.error || 'failed' : f.progress || f.status}
          </p>
        ))}
      </div>
      {error && <p className="text-[10px] text-chip-red">{error}</p>}
      {loaded && !references.length && <p className="text-[10px] text-text-muted">No style reference yet.</p>}
      {references.map(reference => (
        <ReferenceCard key={reference.id} bibleId={bibleId} reference={reference} ask={ask} onChange={replace}
          onDeleted={setReferences} />
      ))}
    </div>
  )
}
