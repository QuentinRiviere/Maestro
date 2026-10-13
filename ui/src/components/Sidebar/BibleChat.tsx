import { useEffect, useRef, useState } from 'react'
import { Check, Loader2, Send, Sparkles, X } from 'lucide-react'
import {
  askBibleAssistant, type BibleChatMessage, type DirectorBible,
} from '../../api/directorBibles'

interface Turn extends BibleChatMessage {
  changes?: string[]
  proposal?: DirectorBible
  status?: 'pending' | 'applied' | 'dismissed'
}

const EXAMPLES = [
  'Create a K-drama about first love in a Busan high school: two leads, a rival, the school and a rooftop',
  'Give her a winter look and add a clear umbrella prop',
  'Write a look block: soft pastel, luminous skin, shallow depth of field',
]

/** Chat with the assistant to write and revise the bible; its proposals apply to the draft. */
export function BibleChat({ bibleId, draft, onApply }: {
  bibleId: string
  draft: DirectorBible
  onApply: (bible: DirectorBible) => void
}) {
  const [turns, setTurns] = useState<Turn[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const end = useRef<HTMLDivElement>(null)

  useEffect(() => { end.current?.scrollIntoView({ block: 'end' }) }, [turns, busy])

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
      setTurns([...next, {
        role: 'assistant', content: answer.reply, changes: answer.changes,
        proposal: answer.changes.length ? answer.bible : undefined,
        status: answer.changes.length ? 'pending' : undefined,
      }])
    } catch (e) {
      setError(e instanceof Error ? e.message : 'The assistant did not answer')
    } finally {
      setBusy(false)
    }
  }
  const settle = (index: number, status: 'applied' | 'dismissed') => {
    const turn = turns[index]
    if (status === 'applied' && turn.proposal) onApply(turn.proposal)
    setTurns(turns.map((t, i) => (i === index ? { ...t, status } : t)))
  }

  return (
    <aside className="w-80 shrink-0 border-l border-border flex flex-col min-h-0">
      <div className="flex items-center gap-1.5 border-b border-border px-2.5 py-2 text-[11px] text-text-primary">
        <Sparkles size={12} className="text-accent-blue" /> Bible assistant
      </div>
      <div className="flex-1 min-h-0 overflow-y-auto p-2.5 space-y-2">
        {!turns.length && (
          <div className="space-y-1.5">
            <p className="text-[10px] text-text-muted">
              Describe the series or ask for changes. Proposals land in the draft when you apply them; nothing is saved until Save.
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
            <div className={`max-w-[95%] rounded-lg px-2 py-1.5 text-[11px] whitespace-pre-wrap ${
              turn.role === 'user' ? 'bg-accent-blue/20 text-text-primary' : 'bg-bg-tertiary text-text-secondary'}`}>
              {turn.content}
              {!!turn.changes?.length && (
                <ul className="mt-1.5 space-y-0.5 border-t border-border/60 pt-1.5 text-[10px] text-text-muted">
                  {turn.changes.map((change, i) => <li key={i}>• {change}</li>)}
                </ul>
              )}
              {turn.status === 'pending' && (
                <div className="mt-1.5 flex gap-1.5">
                  <button onClick={() => settle(index, 'applied')}
                    className="flex items-center gap-1 rounded bg-accent-blue px-2 py-0.5 text-[10px] text-white">
                    <Check size={10} /> Apply
                  </button>
                  <button onClick={() => settle(index, 'dismissed')}
                    className="flex items-center gap-1 rounded border border-border px-2 py-0.5 text-[10px] text-text-secondary hover:text-text-primary">
                    <X size={10} /> Dismiss
                  </button>
                </div>
              )}
              {turn.status === 'applied' && <p className="mt-1 text-[10px] text-emerald-300">Applied to the draft: review, then Save.</p>}
              {turn.status === 'dismissed' && <p className="mt-1 text-[10px] text-text-muted">Dismissed.</p>}
            </div>
          </div>
        ))}
        {busy && <p className="flex items-center gap-1 text-[10px] text-text-muted"><Loader2 size={10} className="animate-spin" /> Writing…</p>}
        {error && <p className="text-[10px] text-chip-red">{error}</p>}
        <div ref={end} />
      </div>
      <div className="border-t border-border p-2 flex gap-1.5">
        <textarea
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); void send(input) } }}
          rows={2}
          placeholder="Ask for characters, places, props, looks or rules…"
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
