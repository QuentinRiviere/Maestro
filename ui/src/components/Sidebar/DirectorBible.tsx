import { useEffect, useState } from 'react'
import { BookOpen, Check, Loader2, Settings2 } from 'lucide-react'
import {
  fetchDirectorBible, fetchDirectorBibles, saveCastToBible,
  type BibleElement, type DirectorBible, type DirectorBibleSummary,
} from '../../api/directorBibles'
import { useStore } from '../../stores/useStore'
import type { DirectorBibleSelection } from '../../types'
import { BibleManagerDialog } from './BibleManagerDialog'

const NEW_BIBLE = '__new__'
const GROUPS = [['characters', 'Cast'], ['locations', 'Places'], ['props', 'Props']] as const

function useBibles(refreshKey = 0) {
  const [bibles, setBibles] = useState<DirectorBibleSummary[]>([])
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    let live = true
    fetchDirectorBibles()
      .then(list => { if (live) setBibles(list) })
      .catch(e => { if (live) setError(e instanceof Error ? e.message : 'Could not load bibles') })
    return () => { live = false }
  }, [refreshKey])
  return { bibles, error }
}

/** Which entries of the bible this project uses, and in which look. */
function EpisodeSelection({ bible, disabled }: { bible: DirectorBible; disabled: boolean }) {
  const selection = useStore(s => s.directorBibleSelection)
  const setSelection = useStore(s => s.setDirectorBibleSelection)
  const toggle = (group: typeof GROUPS[number][0], element: BibleElement, on: boolean) => {
    const all = bible[group].map(e => e.name)
    const current = selection[group] ?? all
    const next = on ? [...new Set([...current, element.name])] : current.filter(n => n !== element.name)
    setSelection({ ...selection, [group]: next.length === all.length ? undefined : next } as DirectorBibleSelection)
  }
  const setLook = (element: BibleElement, look: string) => {
    const variants = { ...(selection.variants ?? {}) }
    if (look === 'default') delete variants[element.name]
    else variants[element.name] = look
    setSelection({ ...selection, variants: Object.keys(variants).length ? variants : undefined })
  }

  return (
    <div className="space-y-1">
      {GROUPS.map(([group, title]) => bible[group].length > 0 && (
        <div key={group} className="flex flex-wrap items-center gap-1">
          <span className="w-10 shrink-0 text-[9px] text-text-muted">{title}</span>
          {bible[group].map(element => {
            const on = (selection[group] ?? bible[group].map(e => e.name)).includes(element.name)
            return (
              <span key={element.id} className={`flex items-center gap-0.5 rounded border px-1 py-0.5 text-[9px] ${
                on ? 'border-accent-blue/60 text-text-primary' : 'border-border text-text-muted'}`}>
                <input type="checkbox" className="w-2.5 h-2.5" checked={on} disabled={disabled}
                  onChange={e => toggle(group, element, e.target.checked)} aria-label={`Use ${element.name}`} />
                {element.name}
                {on && element.variants.length > 1 && (
                  <select value={selection.variants?.[element.name] ?? 'default'} disabled={disabled}
                    onChange={e => setLook(element, e.target.value)} aria-label={`${element.name} look`}
                    className="bg-transparent text-[9px] text-accent-blue focus:outline-none">
                    {element.variants.map(v => <option key={v.id} value={v.id}>{v.label}</option>)}
                  </select>
                )}
              </span>
            )
          })}
        </div>
      ))}
    </div>
  )
}

/** Start the project from a bible: its cast, style and world rules replace casting. */
export function DirectorBiblePicker({ disabled }: { disabled: boolean }) {
  const bibleId = useStore(s => s.directorBibleId)
  const setBibleId = useStore(s => s.setDirectorBibleId)
  const [refresh, setRefresh] = useState(0)
  const { bibles, error } = useBibles(refresh)
  const [loaded, setLoaded] = useState<DirectorBible | null>(null)
  const [managing, setManaging] = useState(false)
  const bible = bibleId && loaded?.id === bibleId ? loaded : null

  useEffect(() => {
    if (!bibleId) return
    let live = true
    fetchDirectorBible(bibleId).then(b => { if (live) setLoaded(b) }).catch(() => undefined)
    return () => { live = false }
  }, [bibleId, refresh])

  return (
    <div className="space-y-1">
      <div className="flex items-center gap-1.5 text-[10px] text-text-secondary"
        title="A bible keeps the approved cast, places, props, style and world rules of a series, for every episode">
        <BookOpen size={10} className="shrink-0 text-text-muted" />
        <span className="shrink-0">Bible</span>
        <select
          value={bibleId}
          disabled={disabled}
          onChange={e => setBibleId(e.target.value)}
          aria-label="Bible"
          className="min-w-0 flex-1 bg-bg-tertiary border border-border rounded px-1.5 py-0.5 text-[10px] text-text-primary focus:outline-none focus:border-accent-blue disabled:opacity-50"
        >
          <option value="">None (cast this project)</option>
          {bibles.map(b => <option key={b.id} value={b.id}>{b.name} · r{b.revision}</option>)}
          {bibleId && !bibles.some(b => b.id === bibleId) && <option value={bibleId}>{bibleId} (not found)</option>}
        </select>
        <button onClick={() => setManaging(true)} title="Manage bibles" aria-label="Manage bibles"
          className="shrink-0 text-text-muted hover:text-text-primary"><Settings2 size={12} /></button>
      </div>
      {bible && <EpisodeSelection bible={bible} disabled={disabled} />}
      {error && <p className="text-[9px] text-chip-red">{error}</p>}
      {managing && (
        <BibleManagerDialog initialId={bibleId || undefined}
          onClose={() => { setManaging(false); setRefresh(n => n + 1) }} />
      )}
    </div>
  )
}

/** Keep an approved cast for later episodes: a new bible, or merged into an existing one. */
export function SaveCastToBible({ pid, compact = false }: { pid: string; compact?: boolean }) {
  const currentBible = useStore(s => (s.pipelineId === pid ? s.pipelineStatus?.bible?.id : '') ?? '')
  const [refresh, setRefresh] = useState(0)
  const { bibles } = useBibles(refresh)
  const [target, setTarget] = useState(currentBible || NEW_BIBLE)
  const [name, setName] = useState('')
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState<{ ok: boolean; text: string } | null>(null)

  const save = async () => {
    setBusy(true)
    setMessage(null)
    try {
      const bible = await saveCastToBible(pid, target === NEW_BIBLE ? { name: name.trim() } : { bible_id: target })
      setTarget(bible.id)
      setRefresh(n => n + 1)
      setMessage({ ok: true, text: `Saved in ${bible.name} (r${bible.revision})` })
    } catch (e) {
      setMessage({ ok: false, text: e instanceof Error ? e.message : 'Save failed' })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className={compact ? 'space-y-1' : 'rounded-lg border border-border bg-bg-tertiary/40 p-2 space-y-1.5'}>
      {!compact && (
        <p className="text-[10px] text-text-secondary flex items-center gap-1">
          <BookOpen size={10} /> Keep this cast in a bible for later episodes
        </p>
      )}
      <div className="flex gap-1.5">
        <select
          value={target}
          onChange={e => setTarget(e.target.value)}
          aria-label="Target bible"
          className="min-w-0 flex-1 bg-bg-tertiary border border-border rounded px-1.5 py-1 text-[10px] text-text-primary focus:outline-none focus:border-accent-blue"
        >
          <option value={NEW_BIBLE}>New bible…</option>
          {bibles.map(b => <option key={b.id} value={b.id}>{b.name}</option>)}
        </select>
        {target === NEW_BIBLE && (
          <input
            value={name}
            onChange={e => setName(e.target.value)}
            placeholder="Series name"
            aria-label="New bible name"
            className="min-w-0 flex-1 bg-bg-tertiary border border-border rounded px-2 py-1 text-[10px] text-text-primary focus:outline-none focus:border-accent-blue"
          />
        )}
        <button
          onClick={() => void save()}
          disabled={busy || (target === NEW_BIBLE && !name.trim())}
          title="Save this project's cast into the bible"
          className="px-2 py-1 rounded border border-border text-[10px] text-text-secondary hover:text-text-primary hover:bg-bg-hover flex items-center gap-1 disabled:opacity-50"
        >
          {busy ? <Loader2 size={10} className="animate-spin" /> : <Check size={10} />} {compact ? 'Save cast' : 'Save'}
        </button>
      </div>
      {message && <p className={`text-[10px] ${message.ok ? 'text-emerald-300' : 'text-chip-red'}`}>{message.text}</p>}
    </div>
  )
}
