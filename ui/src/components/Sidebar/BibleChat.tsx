import { useEffect, useRef, useState } from 'react'
import { Loader2, Send, Sparkles, Undo2, Wand2 } from 'lucide-react'
import {
  askBibleAssistant, type BibleChatMessage, type DirectorBible,
} from '../../api/directorBibles'

interface Turn extends BibleChatMessage {
  changes?: string[]
  /** The draft before this turn's changes, for Undo. */
  before?: DirectorBible
  undone?: boolean
}

const EXAMPLES = [
  'A K-drama about first love in a Busan high school, 6 episodes of 2 minutes',
  'Une série noire à Hong Kong dans les années 90, très stylisée',
  'Help me find the story: I only have two characters in mind',
]

/** Build the bible by talking it through; once the assistant has what it needs, Generate renders the assets. */
export function BibleChat({ bibleId, draft, onChange, onGenerate, pendingImages, imageModel, rendering }: {
  bibleId: string
  draft: DirectorBible
  onChange: (bible: DirectorBible) => void
  onGenerate: () => Promise<void>
  pendingImages: number
  imageModel: string
  rendering: boolean
}) {
  const [turns, setTurns] = useState<Turn[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [ready, setReady] = useState(false)
  const [missing, setMissing] = useState<string[]>([])
  const [generating, setGenerating] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const end = useRef<HTMLDivElement>(null)

  useEffect(() => { end.current?.scrollIntoView({ block: 'end' }) }, [turns, busy, ready])

  const send = async (text: string) => {
    const content = text.trim()
    if (!content || busy) return
    const next: Turn[] = [...turns, { role: 'user', content }]
    setTurns(next)
    setInput('')
    setBusy(true)
    setError(null)
    try {
      const answer = await askBibleAssistant(bibleId, draft, next.map(({ role, content: c }) => ({ role, content: c })))
      const changed = answer.changes.length > 0
      if (changed) onChange(answer.bible)
      setTurns([...next, { role: 'assistant', content: answer.reply, changes: answer.changes, before: changed ? draft : undefined }])
      setReady(answer.ready)
      setMissing(answer.missing)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'The assistant did not answer')
    } finally {
      setBusy(false)
    }
  }
  const undo = (index: number) => {
    const turn = turns[index]
    if (!turn.before) return
    onChange(turn.before)
    setTurns(turns.map((t, i) => (i === index ? { ...t, undone: true } : t)))
  }
  const generate = async () => {
    setGenerating(true)
    setError(null)
    try { await onGenerate() } catch (e) { setError(e instanceof Error ? e.message : 'Generation failed') } finally { setGenerating(false) }
  }
  const lastUndoable = turns.map((t, i) => (t.before && !t.undone ? i : -1)).reduce((a, b) => Math.max(a, b), -1)

  return (
    <aside className="w-[26rem] shrink-0 border-l border-border flex flex-col min-h-0">
      <div className="flex items-center gap-1.5 border-b border-border px-2.5 py-2 text-[11px] text-text-primary">
        <Sparkles size={12} className="text-accent-blue" /> Bible assistant
      </div>
      <div className="flex-1 min-h-0 overflow-y-auto p-2.5 space-y-2">
        {!turns.length && (
          <div className="space-y-1.5">
            <p className="text-[11px] text-text-secondary">
              Tell me about your series. We will shape the story, the stakes, the characters, the recurring places,
              possible twists and the visual and narrative style together; the bible fills in as we go.
              When it is complete, Generate renders the character sheets, places and props.
            </p>
            {EXAMPLES.map(example => (
              <button key={example} onClick={() => setInput(example)}
                className="block w-full text-left rounded border border-border px-2 py-1 text-[10px] text-text-secondary hover:text-text-primary hover:bg-bg-hover">
                {example}
              </button>
            ))}
          </div>
        )}
        {turns.map((turn, index) => (
          <div key={index} className={turn.role === 'user' ? 'flex justify-end' : ''}>
            <div className={`max-w-[95%] rounded-lg px-2.5 py-1.5 text-[11px] whitespace-pre-wrap ${
              turn.role === 'user' ? 'bg-accent-blue/20 text-text-primary' : 'bg-bg-tertiary text-text-secondary'}`}>
              {turn.content}
              {!!turn.changes?.length && (
                <ul className={`mt-1.5 space-y-0.5 border-t border-border/60 pt-1.5 text-[10px] ${turn.undone ? 'text-text-muted line-through' : 'text-emerald-300/80'}`}>
                  {turn.changes.map((change, i) => <li key={i}>• {change}</li>)}
                </ul>
              )}
              {index === lastUndoable && (
                <button onClick={() => undo(index)}
                  className="mt-1 flex items-center gap-1 text-[10px] text-text-muted hover:text-text-primary">
                  <Undo2 size={10} /> Undo these changes
                </button>
              )}
            </div>
          </div>
        ))}
        {busy && <p className="flex items-center gap-1 text-[10px] text-text-muted"><Loader2 size={10} className="animate-spin" /> Thinking…</p>}
        {!busy && turns.length > 0 && !ready && missing.length > 0 && (
          <div className="rounded border border-border/60 px-2 py-1.5 text-[10px] text-text-muted">
            Still needed before generating: {missing.join(' · ')}
          </div>
        )}
        {!busy && ready && (
          <div className="rounded-lg border border-accent-blue/40 bg-accent-blue/10 p-2 space-y-1.5">
            <p className="text-[11px] text-text-primary">
              The bible is ready. Generate renders {pendingImages} image{pendingImages === 1 ? '' : 's'} with {imageModel}
              {' '}(default looks first, places matched to the first one, then the other looks).
            </p>
            <button onClick={() => void generate()} disabled={generating || rendering || pendingImages === 0}
              className="w-full flex items-center justify-center gap-1.5 rounded bg-accent-blue py-1.5 text-[11px] font-medium text-white disabled:opacity-40">
              {generating || rendering ? <Loader2 size={12} className="animate-spin" /> : <Wand2 size={12} />}
              {rendering ? 'Rendering…' : pendingImages === 0 ? 'Every look has an image' : 'Generate assets'}
            </button>
          </div>
        )}
        {error && <p className="text-[10px] text-chip-red">{error}</p>}
        <div ref={end} />
      </div>
      <div className="border-t border-border p-2 flex gap-1.5">
        <textarea
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); void send(input) } }}
          rows={3}
          placeholder="Your idea, an answer, a change…"
          aria-label="Message to the bible assistant"
          className="flex-1 resize-none bg-bg-tertiary border border-border rounded px-2 py-1 text-[11px] text-text-primary focus:outline-none focus:border-accent-blue"
        />
        <button onClick={() => void send(input)} disabled={busy || !input.trim()} aria-label="Send"
          className="self-end rounded bg-accent-blue p-1.5 text-white disabled:opacity-40">
          <Send size={12} />
        </button>
      </div>
    </aside>
  )
}
