import { useCallback, useEffect, useState } from 'react'
import { BookOpen, Clapperboard, Gavel, Loader2, Plus, RefreshCw, Trash2, UserRound } from 'lucide-react'
import { AppModeToggle, MaestroBrand } from '../AppModeNavigation'
import { useConfirm } from '../Sidebar/ConfirmDialog'
import { ClipCard, type Ask } from '../Sidebar/BiblePersonaStudio'
import {
  addPersonaToBible, createPersona, deletePersona, evaluatePersona, fetchPersona, fetchPersonas, isLive, personaPortraitUrl,
  repaintPersona, startPersonaSeries, type ClipEngine, type ClipTemplate, type Persona, type PersonaClip, type PersonaFormat, type PersonaRequest,
} from '../../api/directorPersona'
import { fetchDirectorBible, fetchDirectorBibles, type DirectorBible, type DirectorBibleSummary } from '../../api/directorBibles'

const input = 'w-full bg-bg-tertiary border border-border rounded px-2 py-1 text-[11px] text-text-primary focus:outline-none focus:border-accent-blue'
const label = 'text-[9px] text-text-muted uppercase tracking-wider block mb-0.5'
const button = 'flex items-center gap-1 rounded border border-border px-2 py-0.5 text-[10px] text-text-secondary hover:text-text-primary disabled:opacity-40'
const primary = 'flex items-center gap-1 rounded bg-accent-blue px-2 py-1 text-[10px] text-white disabled:opacity-40'
const STATUS_CLASS: Record<string, string> = {
  queued: 'text-text-muted', running: 'text-accent-blue', done: 'text-emerald-300', failed: 'text-chip-red',
}

function useAct() {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const act = async (call: () => Promise<void>) => {
    setBusy(true)
    setError(null)
    try { await call() } catch (e) { setError(e instanceof Error ? e.message : 'Failed') } finally { setBusy(false) }
  }
  return { busy, error, act, setError }
}

/** The persona format in the recent trend: look never changes, no speech, one move, places that get more absurd. */
function PersonaForm({ bibles, formats, onCreated, onCancel }: {
  bibles: DirectorBibleSummary[]; formats: Record<string, string>; onCreated: (persona: Persona) => void; onCancel: () => void
}) {
  const [draft, setDraft] = useState<PersonaRequest>({ name: '', handle: '', look: '', move: '', format: 'silent_move', world: '', music: '', places: '', orientation: 'portrait', seconds: 6, speed: 1.5, template: 'trend', engine: 'ltx', audio: 'silent' })
  const [source, setSource] = useState<'look' | 'character'>('look')
  const [bibleId, setBibleId] = useState('')
  const [bible, setBible] = useState<DirectorBible | null>(null)
  const [characterId, setCharacterId] = useState('')
  const { busy, error, act } = useAct()
  useEffect(() => {
    let alive = true
    const load = bibleId ? fetchDirectorBible(bibleId) : Promise.resolve(null)
    load.then(b => { if (alive) { setBible(b); setCharacterId(b?.characters.find(c => c.image)?.id ?? '') } }).catch(() => {})
    return () => { alive = false }
  }, [bibleId])
  const set = (key: keyof PersonaRequest, value: string | number) => setDraft(d => ({ ...d, [key]: value }))
  const ready = !!draft.name.trim() && !!draft.move.trim() && (source === 'look' ? !!draft.look?.trim() : !!characterId)
  return (
    <div className="rounded-lg border border-border bg-bg-tertiary/30 p-3 space-y-2">
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
        <label className="block"><span className={label}>Name</span>
          <input className={input} value={draft.name} onChange={e => set('name', e.target.value)} placeholder="Marcel Ventôse" /></label>
        <label className="block"><span className={label}>Handle</span>
          <input className={input} value={draft.handle} onChange={e => set('handle', e.target.value)} placeholder="marcel.ventose" /></label>
        <label className="block"><span className={label}>Format</span>
          <select className={input} value={draft.format} onChange={e => set('format', e.target.value as PersonaFormat)} aria-label="Persona format">
            {Object.entries(formats).map(([id, text]) => <option key={id} value={id} disabled={id !== 'silent_move'}>{id === 'silent_move' ? 'Silent move' : 'Diary (soon)'} · {text}</option>)}
          </select></label>
      </div>
      <div className="flex gap-1">
        {(['look', 'character'] as const).map(option => (
          <button key={option} className={`rounded px-2 py-0.5 text-[10px] ${source === option ? 'bg-bg-hover text-text-primary' : 'text-text-secondary hover:text-text-primary'}`} onClick={() => setSource(option)}>
            {option === 'look' ? 'Describe the look (the editor paints the portrait)' : 'Take a bible character'}
          </button>
        ))}
      </div>
      {source === 'look' ? (
        <label className="block"><span className={label}>Look, word for word, the same forever (age, face, hair, one outfit, one prop)</span>
          <textarea className={input} rows={3} value={draft.look} onChange={e => set('look', e.target.value)}
            placeholder="a 68-year-old retired lifeguard, shaved head, grey handlebar moustache, swimming goggles on the forehead, burgundy bathrobe over a black competition swimsuit, white socks and pool slides, a referee whistle on a red lanyard" /></label>
      ) : (
        <div className="grid grid-cols-2 gap-2">
          <label className="block"><span className={label}>Bible</span>
            <select className={input} value={bibleId} onChange={e => setBibleId(e.target.value)} aria-label="Bible">
              <option value="">Pick a bible</option>
              {bibles.map(b => <option key={b.id} value={b.id}>{b.name}</option>)}
            </select></label>
          <label className="block"><span className={label}>Character</span>
            <select className={input} value={characterId} onChange={e => setCharacterId(e.target.value)} aria-label="Character" disabled={!bible}>
              {(bible?.characters ?? []).filter(c => c.image).map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
            </select></label>
        </div>
      )}
      <label className="block"><span className={label}>Signature move (the same every clip, in order, hands and face)</span>
        <textarea className={input} rows={2} value={draft.move} onChange={e => set('move', e.target.value)}
          placeholder="raises the whistle to his lips and puffs his cheeks, shades his eyes and scans the horizon, then points a stern finger at the camera" /></label>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        <label className="block"><span className={label}>World (where he lives, how people react)</span>
          <input className={input} value={draft.world} onChange={e => set('world', e.target.value)} placeholder="Paris, always in his bathrobe, nobody ever reacts" /></label>
        <label className="block"><span className={label}>Music note (laid over the silent clip, outside Maestro)</span>
          <input className={input} value={draft.music} onChange={e => set('music', e.target.value)} placeholder="French rap or 90s R&B" /></label>
      </div>
      <label className="block"><span className={label}>Places to visit, one per line, more absurd each time</span>
        <textarea className={input} rows={4} value={draft.places} onChange={e => set('places', e.target.value)}
          placeholder={'the queue of a busy boulangerie at 8 am\nthe platform of a crowded metro station\nthe Louvre pyramid courtyard\na cow pasture in Normandy'} /></label>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        <label className="block"><span className={label}>Orientation</span>
          <select className={input} value={draft.orientation} onChange={e => set('orientation', e.target.value)} aria-label="Orientation">
            <option value="portrait">Portrait 9:16</option><option value="landscape">Landscape 16:9</option>
          </select></label>
        <label className="block"><span className={label}>Clip length (s)</span>
          <input className={input} type="number" min={2} max={14} value={draft.seconds} onChange={e => set('seconds', Number(e.target.value) || 6)} /></label>
        <label className="block" title="LTX moves at about two thirds of real pace: the clip is rendered this many times longer and sped up in post. 1.5 reads as a phone video."><span className={label}>Pace ×</span>
          <input className={input} type="number" min={1} max={3} step={0.25} value={draft.speed} onChange={e => set('speed', Number(e.target.value) || 1.5)} /></label>
        <label className="block"><span className={label}>Sound</span>
          <select className={input} value={draft.audio} onChange={e => set('audio', e.target.value)} aria-label="Sound">
            <option value="silent">Silent</option><option value="model">Let LTX invent a soundtrack</option>
          </select></label>
      </div>
      <label className="block" title="How a clip is cut: one full-body take, or several phone shots (each with its own start image in the same place) joined together"><span className={label}>Shots</span>
        <select className={input} value={draft.template} onChange={e => set('template', e.target.value as ClipTemplate)} aria-label="Shots template">
          <option value="trend">Trend · wide 4 s, side 4 s, close 3 s (11 s)</option>
          <option value="street">Street · over a shoulder 4 s, low angle 4 s, close 3 s</option>
          <option value="reveal">Reveal · close 3 s, wide 5 s, side 3 s</option>
          <option value="single">Single · one full-body take of the clip length</option>
        </select></label>
      <label className="block" title="LTX-2.5 (576x1024, the default) or MiniMax H3 image-to-video (480x864 on the Spark, its own motion style)"><span className={label}>Engine</span>
        <select className={input} value={draft.engine} onChange={e => set('engine', e.target.value as ClipEngine)} aria-label="Clip engine">
          <option value="ltx">LTX-2.5</option><option value="h3">MiniMax H3 (480p)</option>
        </select></label>
      {error && <p role="alert" className="text-[10px] text-chip-red">{error}</p>}
      <div className="flex gap-2">
        <button className={primary} disabled={busy || !ready}
          onClick={() => void act(async () => onCreated(await createPersona({ ...draft, bible_id: source === 'character' ? bibleId : '', character_id: source === 'character' ? characterId : '' })))}>
          {busy ? <Loader2 size={10} className="animate-spin" /> : <Plus size={10} />} Create the persona
        </button>
        <button className={button} onClick={onCancel}>Cancel</button>
      </div>
    </div>
  )
}

function PersonaDetail({ persona, clips, bibles, ask, onChange, onClips, onDeleted }: {
  persona: Persona; clips: PersonaClip[]; bibles: DirectorBibleSummary[]; ask: Ask
  onChange: (persona: Persona) => void; onClips: (clips: PersonaClip[]) => void; onDeleted: () => void
}) {
  const { busy, error, act } = useAct()
  const [count, setCount] = useState(3)
  const [extra, setExtra] = useState('')
  const [template, setTemplate] = useState<ClipTemplate | ''>('')
  const [engine, setEngine] = useState<ClipEngine | ''>('')
  const [bibleId, setBibleId] = useState(bibles[0]?.id ?? '')
  const live = isLive(persona.status)
  const judged = clips.filter(c => c.evaluation && !c.evaluation.error)
  const mean = (key: 'overall' | 'identity' | 'move' | 'place' | 'artefacts' | 'hook') => {
    const values = judged.map(c => c.evaluation?.[key]).filter((v): v is number => typeof v === 'number')
    return values.length ? (values.reduce((a, b) => a + b, 0) / values.length).toFixed(1) : '–'
  }
  const stamp = encodeURIComponent(persona.updated_at)
  return (
    <div className="space-y-3">
      <div className="flex gap-3">
        <div className="shrink-0">
          {persona.portrait && persona.status === 'done' ? (
            <img alt={`Portrait of ${persona.name}`} className="h-[260px] w-auto rounded bg-black object-contain" src={`${personaPortraitUrl(persona.id)}?v=${stamp}`} />
          ) : (
            <div className="flex h-[260px] w-[146px] items-center justify-center rounded border border-border text-[10px] text-text-muted">
              {live ? <span className="flex items-center gap-1"><Loader2 size={10} className="animate-spin" /> {persona.progress || 'painting'}</span> : 'no portrait'}
            </div>
          )}
        </div>
        <div className="min-w-0 flex-1 space-y-1.5">
          <div className="flex items-center gap-2">
            <h2 className="truncate text-sm font-semibold text-text-primary">{persona.name}</h2>
            {persona.handle && <span className="text-[10px] text-text-muted">@{persona.handle}</span>}
            <span className={`text-[10px] ${STATUS_CLASS[persona.status] ?? ''}`} title={persona.error || persona.progress}>{persona.status}</span>
            {persona.character_id && <span className="text-[10px] text-text-muted" title={persona.bible_id}>· in bible as a character</span>}
          </div>
          <p className="text-[10px] text-text-secondary"><span className={label}>Look</span>{persona.look || '–'}</p>
          {persona.traits?.length > 0 && <p className="text-[10px] text-text-secondary"><span className={label}>Traits</span>{persona.traits.join(' · ')}</p>}
          <p className="text-[10px] text-text-secondary"><span className={label}>Signature move</span>{persona.move}</p>
          {persona.world && <p className="text-[10px] text-text-secondary"><span className={label}>World</span>{persona.world}</p>}
          {persona.music && <p className="text-[10px] text-text-secondary"><span className={label}>Music</span>{persona.music}</p>}
          <p className="text-[10px] text-text-muted">{persona.orientation} · {persona.template ?? 'single'} · {persona.engine ?? 'ltx'} · {persona.seconds} s · pace ×{persona.speed ?? 1} · {persona.audio} · {persona.places.length} places · {clips.length} clip{clips.length === 1 ? '' : 's'}</p>
          {judged.length > 0 && (
            <p className="text-[10px] text-text-secondary">
              Judge on {judged.length} clip{judged.length === 1 ? '' : 's'}: overall {mean('overall')} · identity {mean('identity')} · move {mean('move')} · place {mean('place')} · artefacts {mean('artefacts')} · hook {mean('hook')}
            </p>
          )}
          {persona.error && <p role="alert" className="text-[10px] text-chip-red">{persona.error}</p>}
          {error && <p role="alert" className="text-[10px] text-chip-red">{error}</p>}
          <div className="flex flex-wrap gap-2 pt-1">
            {!live && persona.look && (
              <button className={button} disabled={busy} title="Paint the portrait again from the look"
                onClick={() => void act(async () => onChange(await repaintPersona(persona.id)))}><RefreshCw size={10} /> Repaint</button>
            )}
            <button className={button} disabled={busy || clips.every(c => c.status !== 'done' || !!c.evaluation)}
              title="The analyst scores every finished clip: identity against the portrait, move, place, artefacts, hook (1 to 5)"
              onClick={() => void act(async () => onClips(await evaluatePersona(persona.id)))}><Gavel size={10} /> Judge the clips</button>
            {!persona.character_id && bibles.length > 0 && (
              <span className="flex items-center gap-1">
                <select className={`${input} w-auto`} value={bibleId} onChange={e => setBibleId(e.target.value)} aria-label="Bible to join">
                  {bibles.map(b => <option key={b.id} value={b.id}>{b.name}</option>)}
                </select>
                <button className={button} disabled={busy || !bibleId || persona.status !== 'done'} title="Add the persona to this bible as a character (its portrait approved), so episodes and covers can use it"
                  onClick={() => void act(async () => onChange(await addPersonaToBible(persona.id, bibleId)))}><BookOpen size={10} /> Join bible</button>
              </span>
            )}
            <button className={button} disabled={busy || live} title="Delete the persona (its clips stay)"
              onClick={() => ask({ title: `Delete "${persona.name}"?`, confirmLabel: 'Delete', danger: true, message: 'The portrait is removed; clips already rendered stay.', run: () => act(async () => { await deletePersona(persona.id); onDeleted() }) })}>
              <Trash2 size={10} />
            </button>
          </div>
        </div>
      </div>

      <div className="rounded-lg border border-border bg-bg-tertiary/30 p-2.5 space-y-2">
        <span className={label}>New clips (one per place, the move never changes)</span>
        <div className="grid grid-cols-1 sm:grid-cols-[1fr_80px_110px_110px_auto] gap-2 items-end">
          <label className="block"><span className={label}>Places (one per line; empty = the next {count} of the persona's list)</span>
            <textarea className={input} rows={2} value={extra} onChange={e => setExtra(e.target.value)} /></label>
          <label className="block"><span className={label}>How many</span>
            <input className={input} type="number" min={1} max={12} value={count} onChange={e => setCount(Number(e.target.value) || 1)} /></label>
          <label className="block"><span className={label}>Shots</span>
            <select className={input} value={template} onChange={e => setTemplate(e.target.value as ClipTemplate | '')} aria-label="Shots template for these clips">
              <option value="">Persona's ({persona.template ?? 'single'})</option>
              <option value="trend">Trend</option><option value="street">Street</option><option value="reveal">Reveal</option><option value="single">Single</option>
            </select></label>
          <label className="block"><span className={label}>Engine</span>
            <select className={input} value={engine} onChange={e => setEngine(e.target.value as ClipEngine | '')} aria-label="Engine for these clips">
              <option value="">Persona's ({persona.engine ?? 'ltx'})</option><option value="ltx">LTX-2.5</option><option value="h3">H3 480p</option>
            </select></label>
          <button className={primary} disabled={busy || persona.status !== 'done'}
            onClick={() => void act(async () => {
              const places = extra.split('\n').map(s => s.trim()).filter(Boolean)
              const made = await startPersonaSeries(persona.id, { places: places.length ? places : undefined, count: places.length ? 0 : count, template: template || undefined, engine: engine || undefined })
              onClips([...made, ...clips]); setExtra('')
            })}>
            {busy ? <Loader2 size={10} className="animate-spin" /> : <Clapperboard size={10} />} Render
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-2.5">
        {clips.map(clip => (
          <ClipCard key={clip.id} clip={clip} ask={ask} title={clip.settings.place || clip.name}
            onChange={next => onClips(clips.map(c => (c.id === next.id ? next : c)))}
            onDeleted={() => onClips(clips.filter(c => c.id !== clip.id))} />
        ))}
      </div>
    </div>
  )
}

/** The Personas page: the AI personas (one look, one move, a new place per clip) and their clips. */
export function PersonasWorkspace() {
  const [personas, setPersonas] = useState<Persona[]>([])
  const [formats, setFormats] = useState<Record<string, string>>({})
  const [bibles, setBibles] = useState<DirectorBibleSummary[]>([])
  const [selected, setSelected] = useState('')
  const [clips, setClips] = useState<PersonaClip[]>([])
  const [creating, setCreating] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const { ask, dialog } = useConfirm()

  const refreshList = useCallback(() => fetchPersonas().then(r => { setPersonas(r.personas); setFormats(r.formats) })
    .catch(e => setError(e instanceof Error ? e.message : 'Could not load the personas')), [])
  useEffect(() => { void refreshList(); fetchDirectorBibles().then(setBibles).catch(() => setBibles([])) }, [refreshList])

  const refreshSelected = useCallback(() => {
    if (!selected) return Promise.resolve()
    return fetchPersona(selected).then(r => {
      setPersonas(current => current.map(p => (p.id === r.persona.id ? r.persona : p)))
      setClips(r.clips)
    }).catch(e => setError(e instanceof Error ? e.message : 'Could not load the persona'))
  }, [selected])
  useEffect(() => { void refreshSelected() }, [refreshSelected])

  const persona = personas.find(p => p.id === selected) ?? null
  const running = !!persona && (isLive(persona.status) || clips.some(c => isLive(c.status)))
  useEffect(() => {
    if (!running) return
    const timer = window.setInterval(() => { void refreshSelected() }, 5000)
    return () => window.clearInterval(timer)
  }, [running, refreshSelected])

  return (
    <div className="flex h-full w-full flex-col">
      <header className="flex h-12 shrink-0 items-center gap-3 border-b border-border bg-bg-secondary px-3">
        <MaestroBrand />
        <AppModeToggle size="sm" />
      </header>
      <div className="flex min-h-0 flex-1">
        <aside className="flex w-60 shrink-0 flex-col border-r border-border bg-bg-secondary">
          <div className="flex items-center justify-between px-2.5 py-2">
            <span className={label}>Personas</span>
            <button className={button} onClick={() => setCreating(true)}><Plus size={10} /> New</button>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto">
            {personas.map(p => (
              <button key={p.id} onClick={() => { setSelected(p.id); setCreating(false) }}
                className={`flex w-full items-center gap-2 px-2.5 py-1.5 text-left text-[11px] ${selected === p.id ? 'bg-bg-hover text-text-primary' : 'text-text-secondary hover:text-text-primary'}`}>
                {p.portrait && p.status === 'done'
                  ? <img alt="" className="h-8 w-8 rounded object-cover" src={`${personaPortraitUrl(p.id)}?v=${encodeURIComponent(p.updated_at)}`} />
                  : <UserRound size={14} className="shrink-0 text-text-muted" />}
                <span className="min-w-0 flex-1 truncate">{p.name}</span>
                <span className="text-[9px] text-text-muted">{p.clips || ''}</span>
              </button>
            ))}
            {personas.length === 0 && <p className="px-2.5 text-[10px] text-text-muted">No persona yet.</p>}
          </div>
        </aside>
        <main className="min-w-0 flex-1 overflow-y-auto p-3">
          {error && <p role="alert" className="mb-2 text-[10px] text-chip-red">{error}</p>}
          {creating ? (
            <PersonaForm bibles={bibles} formats={formats} onCancel={() => setCreating(false)}
              onCreated={p => { setPersonas(current => [...current, p]); setSelected(p.id); setCreating(false) }} />
          ) : persona ? (
            <PersonaDetail persona={persona} clips={clips} bibles={bibles} ask={ask}
              onChange={next => setPersonas(current => current.map(p => (p.id === next.id ? next : p)))}
              onClips={setClips}
              onDeleted={() => { setPersonas(current => current.filter(p => p.id !== persona.id)); setSelected(''); setClips([]) }} />
          ) : (
            <div className="max-w-xl space-y-2 text-[11px] text-text-secondary">
              <p className="text-text-primary">An AI persona: one character whose look never changes, who never speaks, performs the same signature move in a new, more absurd place every clip. You lay a song over the silent clip when you post it.</p>
              <p>Create one from a look in words (the editor paints the portrait) or from a bible character, then render clips place by place and let the judge score them: identity, move, place, artefacts, hook.</p>
              <button className={primary} onClick={() => setCreating(true)}><Plus size={10} /> New persona</button>
            </div>
          )}
        </main>
      </div>
      {dialog}
    </div>
  )
}
