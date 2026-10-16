import { useEffect, useState } from 'react'
import { Loader2, StickyNote } from 'lucide-react'
import { useStore } from '../stores/useStore'
import { hasAnnotation, parseTags } from '../lib/annotation'
import type { OutputAnnotation } from '../types'

/** The note and tags as a compact amber line under a gallery card's prompt. */
export function AnnotationSummary({ annotation, className = '' }: {
  annotation: OutputAnnotation | null | undefined
  className?: string
}) {
  if (!hasAnnotation(annotation)) return null
  const title = [annotation.note, annotation.tags.length ? `Tags: ${annotation.tags.join(', ')}` : '']
    .filter(Boolean).join('\n')
  return (
    <div className={`flex min-w-0 items-center gap-1 text-[11px] text-amber-300 ${className}`} title={title}>
      <StickyNote size={11} className="shrink-0" />
      {annotation.tags.map(tag => (
        <span key={tag} className="shrink-0 rounded-full bg-amber-500/15 px-1.5 text-[10px]">{tag}</span>
      ))}
      {annotation.note && <span className="truncate">{annotation.note}</span>}
    </div>
  )
}

/** Edit an output's note and comma-separated tags; saved into its sidecar. */
export function AnnotationEditor({ name, workspace, annotation, onSaved }: {
  name: string
  workspace?: string
  annotation: OutputAnnotation | null | undefined
  onSaved: (annotation: OutputAnnotation | null) => void
}) {
  const saveOutputAnnotation = useStore(s => s.saveOutputAnnotation)
  const [note, setNote] = useState(annotation?.note ?? '')
  const [tags, setTags] = useState((annotation?.tags ?? []).join(', '))
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    setNote(annotation?.note ?? '')
    setTags((annotation?.tags ?? []).join(', '))
  }, [annotation?.note, annotation?.tags])

  const dirty = note.trim() !== (annotation?.note ?? '')
    || parseTags(tags).join('\n') !== (annotation?.tags ?? []).join('\n')

  const save = async (next: { note: string; tags: string[] }) => {
    setSaving(true)
    setError('')
    try {
      onSaved(await saveOutputAnnotation(name, next, workspace))
      setSaved(true)
      setTimeout(() => setSaved(false), 1500)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not save the annotation')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="mb-3">
      <div className="mb-1 flex items-center gap-1 text-[10px] font-medium uppercase tracking-wide text-text-muted">
        <StickyNote size={11} /> Annotation
      </div>
      <textarea
        value={note}
        onChange={event => setNote(event.target.value)}
        onKeyDown={event => {
          if (event.key === 'Enter' && (event.metaKey || event.ctrlKey) && dirty && !saving) {
            event.preventDefault()
            void save({ note, tags: parseTags(tags) })
          }
        }}
        placeholder="What works, what is broken (e.g. hand morphs on the cup at 2 s)"
        rows={3}
        maxLength={4000}
        className="w-full resize-y rounded-lg border border-border bg-bg-tertiary px-2 py-1.5 text-xs text-text-primary placeholder:text-text-muted focus:border-accent-blue focus:outline-none"
        aria-label="Annotation note"
      />
      <input
        value={tags}
        onChange={event => setTags(event.target.value)}
        placeholder="Tags, comma separated (keep, redo, hand morph)"
        className="mt-1 w-full rounded-lg border border-border bg-bg-tertiary px-2 py-1 text-xs text-text-primary placeholder:text-text-muted focus:border-accent-blue focus:outline-none"
        aria-label="Annotation tags"
      />
      <div className="mt-1.5 flex items-center gap-2">
        <button
          type="button"
          disabled={!dirty || saving}
          onClick={() => void save({ note, tags: parseTags(tags) })}
          className="flex items-center gap-1 rounded-lg bg-accent-blue/20 px-2.5 py-1 text-xs text-accent-blue transition-colors hover:bg-accent-blue/30 disabled:opacity-40"
        >
          {saving && <Loader2 size={12} className="animate-spin" />} Save
        </button>
        {hasAnnotation(annotation) && (
          <button
            type="button"
            disabled={saving}
            onClick={() => void save({ note: '', tags: [] })}
            className="rounded-lg px-2 py-1 text-xs text-text-muted transition-colors hover:text-red-400 disabled:opacity-40"
          >
            Clear
          </button>
        )}
        {saved && <span className="text-[11px] text-text-muted">Saved</span>}
        {error && <span role="alert" className="text-[11px] text-red-400">{error}</span>}
      </div>
    </div>
  )
}
