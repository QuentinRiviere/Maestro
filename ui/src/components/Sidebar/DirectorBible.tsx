import { useEffect, useState } from 'react'
import { BookOpen, Check, Loader2 } from 'lucide-react'
import { fetchDirectorBibles, saveCastToBible, type DirectorBibleSummary } from '../../api/directorBibles'
import { useStore } from '../../stores/useStore'

const NEW_BIBLE = '__new__'

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

/** Start the project from a bible: its cast, style and world rules replace casting. */
export function DirectorBiblePicker({ disabled }: { disabled: boolean }) {
  const bibleId = useStore(s => s.directorBibleId)
  const setBibleId = useStore(s => s.setDirectorBibleId)
  const { bibles, error } = useBibles()
  const chosen = bibles.find(b => b.id === bibleId)

  return (
    <div className="space-y-1">
      <label className="flex items-center gap-1.5 text-[10px] text-text-secondary" title="A bible keeps the approved cast, style and world rules of a series, for every episode">
        <BookOpen size={10} className="shrink-0 text-text-muted" />
        <span className="shrink-0">Bible</span>
        <select
          value={bibleId}
          disabled={disabled}
          onChange={e => setBibleId(e.target.value)}
          className="min-w-0 flex-1 bg-bg-tertiary border border-border rounded px-1.5 py-0.5 text-[10px] text-text-primary focus:outline-none focus:border-accent-blue disabled:opacity-50"
        >
          <option value="">None (cast this project)</option>
          {bibles.map(b => <option key={b.id} value={b.id}>{b.name} · r{b.revision}</option>)}
          {bibleId && !chosen && <option value={bibleId}>{bibleId} (not found)</option>}
        </select>
      </label>
      {chosen && (
        <p className="text-[9px] text-text-muted truncate" title={[...chosen.characters, ...chosen.locations].join(', ')}>
          {[...chosen.characters, ...chosen.locations].join(' · ') || 'Empty bible'}
        </p>
      )}
      {error && <p className="text-[9px] text-chip-red">{error}</p>}
    </div>
  )
}

/** Keep an approved cast for later episodes: a new bible, or merged into an existing one. */
export function SaveCastToBible({ pid }: { pid: string }) {
  const currentBible = useStore(s => s.pipelineStatus?.bible?.id ?? '')
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
    <div className="rounded-lg border border-border bg-bg-tertiary/40 p-2 space-y-1.5">
      <p className="text-[10px] text-text-secondary flex items-center gap-1">
        <BookOpen size={10} /> Keep this cast in a bible for later episodes
      </p>
      <div className="flex gap-1.5">
        <select
          value={target}
          onChange={e => setTarget(e.target.value)}
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
          className="px-2 py-1 rounded border border-border text-[10px] text-text-secondary hover:text-text-primary hover:bg-bg-hover flex items-center gap-1 disabled:opacity-50"
        >
          {busy ? <Loader2 size={10} className="animate-spin" /> : <Check size={10} />} Save
        </button>
      </div>
      {message && <p className={`text-[10px] ${message.ok ? 'text-emerald-300' : 'text-chip-red'}`}>{message.text}</p>}
    </div>
  )
}
