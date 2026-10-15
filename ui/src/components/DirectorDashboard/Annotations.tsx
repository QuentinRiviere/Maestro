import { useState, type KeyboardEvent, type MouseEvent } from 'react'
import { Check, Loader2, MessageSquare, Pencil, RotateCcw, Trash2 } from 'lucide-react'
import {
  ANNOTATION_CATEGORIES, CATEGORY_LABEL, MAX_ANNOTATION_TEXT, SEVERITY_LABEL, describeTarget, formatNoteTimecode,
  parseTimecode, type AnnotationCategory, type AnnotationDraft, type AnnotationSeverity, type AnnotationTarget,
  type NodeBadge, type RunAnnotation,
} from '../../lib/directorAnnotations'
import type { SceneGroup } from '../../lib/directorRun'
import type { AnnotationActions } from './useRunAnnotations'

const SEVERITY_BG: Record<0 | AnnotationSeverity, string> = {
  0: 'bg-text-muted/60 text-bg-primary',
  1: 'bg-sky-500 text-white',
  2: 'bg-amber-500 text-black',
  3: 'bg-red-500 text-white',
}

const SEVERITY_RING: Record<AnnotationSeverity, string> = {
  1: 'border-sky-500/40', 2: 'border-amber-500/50', 3: 'border-red-500/60',
}

const SEVERITIES: AnnotationSeverity[] = [1, 2, 3]

/** Count of a node's notes, colored by the most severe open one. Clicking lists them. */
export function NoteBadge({ badge, onClick, className = '', small = false }: {
  badge: NodeBadge
  onClick?: () => void
  className?: string
  small?: boolean
}) {
  const count = badge.open || badge.resolved
  const label = badge.open
    ? `${badge.open} open note${badge.open === 1 ? '' : 's'}${badge.resolved ? `, ${badge.resolved} resolved` : ''}`
    : `${badge.resolved} resolved note${badge.resolved === 1 ? '' : 's'}`
  return (
    <button
      type="button"
      className={`nodrag nopan pointer-events-auto inline-flex items-center justify-center gap-0.5 rounded-full font-semibold shadow ring-1 ring-black/30 hover:brightness-110 ${small ? 'min-w-[14px] h-[14px] px-0.5 text-[8px]' : 'min-w-[18px] h-[18px] px-1 text-[10px]'} ${SEVERITY_BG[badge.maxSeverity]} ${className}`}
      title={`${label} · click to list`}
      aria-label={label}
      onClick={(event: MouseEvent) => { event.stopPropagation(); onClick?.() }}
    >
      {!small && <MessageSquare size={9} />}
      {count}
    </button>
  )
}

/** Category, severity, text and optional timecode of one note. */
export function AnnotationForm({ initial, target, allowTimecode, onSubmit, onCancel, submitLabel = 'Save note' }: {
  initial?: RunAnnotation
  target: AnnotationTarget
  allowTimecode: boolean
  onSubmit: (draft: AnnotationDraft) => Promise<void>
  onCancel: () => void
  submitLabel?: string
}) {
  const [category, setCategory] = useState<AnnotationCategory>(initial?.category ?? 'other')
  const [severity, setSeverity] = useState<AnnotationSeverity>(initial?.severity ?? 2)
  const [text, setText] = useState(initial?.text ?? '')
  const [timecode, setTimecode] = useState(initial?.timecode_sec != null ? formatNoteTimecode(initial.timecode_sec) : '')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const parsedTimecode = allowTimecode ? parseTimecode(timecode) : null
  const valid = text.trim().length > 0 && text.length <= MAX_ANNOTATION_TEXT && parsedTimecode !== undefined

  const submit = async () => {
    if (!valid || busy) return
    setBusy(true)
    setError(null)
    try {
      await onSubmit({ target, category, severity, text: text.trim(), timecode_sec: parsedTimecode ?? null })
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setBusy(false)
    }
  }

  const onKeyDown = (event: KeyboardEvent) => {
    if (event.key === 'Escape') {
      event.preventDefault()
      event.stopPropagation()
      onCancel()
    } else if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
      event.preventDefault()
      void submit()
    }
  }

  const field = 'bg-bg-tertiary border border-border rounded px-2 py-1 text-[11px] text-text-primary focus:outline-none focus:border-accent-blue'

  return (
    <form
      className="space-y-2"
      onKeyDown={onKeyDown}
      onSubmit={event => { event.preventDefault(); void submit() }}
    >
      <div className="flex items-center gap-2">
        <select
          value={category}
          onChange={event => setCategory(event.target.value as AnnotationCategory)}
          className={`${field} flex-1 min-w-0`}
          aria-label="Category"
        >
          {ANNOTATION_CATEGORIES.map(value => <option key={value} value={value}>{CATEGORY_LABEL[value]}</option>)}
        </select>
        <div className="flex rounded border border-border overflow-hidden" role="radiogroup" aria-label="Severity">
          {SEVERITIES.map(value => (
            <button
              key={value}
              type="button"
              role="radio"
              aria-checked={severity === value}
              title={`Severity ${value}: ${SEVERITY_LABEL[value]}`}
              onClick={() => setSeverity(value)}
              className={`w-6 h-6 text-[10px] font-semibold ${severity === value ? SEVERITY_BG[value] : 'bg-bg-tertiary text-text-secondary hover:bg-bg-hover'}`}
            >{value}</button>
          ))}
        </div>
      </div>
      <textarea
        value={text}
        onChange={event => setText(event.target.value)}
        rows={4}
        autoFocus
        maxLength={MAX_ANNOTATION_TEXT}
        placeholder="What is wrong, and what should the pipeline do instead?"
        aria-label="Note"
        className={`${field} w-full resize-y`}
      />
      <div className="flex items-center gap-2">
        {allowTimecode && (
          <label className="flex items-center gap-1 text-[10px] text-text-muted">
            At
            <input
              value={timecode}
              onChange={event => setTimecode(event.target.value)}
              placeholder="m:ss"
              aria-label="Timecode in the clip"
              aria-invalid={parsedTimecode === undefined}
              className={`${field} w-16 ${parsedTimecode === undefined ? 'border-chip-red' : ''}`}
            />
          </label>
        )}
        <span className="text-[9px] text-text-muted tabular-nums">{text.length}/{MAX_ANNOTATION_TEXT}</span>
        <div className="ml-auto flex items-center gap-1.5">
          <button type="button" onClick={onCancel} className="px-2 py-1 rounded border border-border text-[11px] text-text-secondary hover:bg-bg-hover">
            Cancel
          </button>
          <button
            type="submit"
            disabled={!valid || busy}
            title="Save (Ctrl/⌘ + Enter)"
            className="flex items-center gap-1 px-2 py-1 rounded bg-accent-blue text-white text-[11px] hover:bg-accent-blue-hover disabled:opacity-40"
          >
            {busy && <Loader2 size={11} className="animate-spin" />} {submitLabel}
          </button>
        </div>
      </div>
      {error && <p role="alert" className="text-[10px] text-chip-red">{error}</p>}
    </form>
  )
}

/** One note with edit, resolve and delete. */
export function AnnotationItem({ annotation, actions, allowTimecode, showTarget, scenes, onFocus }: {
  annotation: RunAnnotation
  actions: AnnotationActions
  allowTimecode: boolean
  showTarget?: boolean
  scenes?: SceneGroup[]
  onFocus?: (annotation: RunAnnotation) => void
}) {
  const [editing, setEditing] = useState(false)
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const run = async (label: string, action: () => Promise<void>) => {
    setBusy(label)
    setError(null)
    try {
      await action()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }

  if (editing) {
    return (
      <li className="rounded border border-accent-blue/40 bg-bg-tertiary/60 p-2">
        <AnnotationForm
          initial={annotation}
          target={annotation.target}
          allowTimecode={allowTimecode || annotation.timecode_sec != null}
          submitLabel="Update"
          onCancel={() => setEditing(false)}
          onSubmit={async draft => {
            await actions.update(annotation.id, {
              category: draft.category, severity: draft.severity, text: draft.text, timecode_sec: draft.timecode_sec,
            })
            setEditing(false)
          }}
        />
      </li>
    )
  }

  const iconButton = 'p-1 rounded text-text-muted hover:text-text-primary hover:bg-bg-hover disabled:opacity-40'
  return (
    <li className={`rounded border-l-2 border border-border ${SEVERITY_RING[annotation.severity]} bg-bg-tertiary/40 px-2 py-1.5 ${annotation.resolved ? 'opacity-60' : ''}`}>
      <div className="flex items-center gap-1.5 min-w-0">
        <span className={`shrink-0 w-4 h-4 rounded-full text-[9px] font-semibold inline-flex items-center justify-center ${SEVERITY_BG[annotation.severity]}`} title={`Severity ${annotation.severity}: ${SEVERITY_LABEL[annotation.severity]}`}>
          {annotation.severity}
        </span>
        <span className="text-[10px] font-semibold text-text-secondary truncate">{CATEGORY_LABEL[annotation.category] ?? annotation.category}</span>
        {annotation.timecode_sec != null && (
          <span className="text-[10px] font-mono text-text-muted">@{formatNoteTimecode(annotation.timecode_sec)}</span>
        )}
        {annotation.resolved && <span className="text-[9px] uppercase tracking-wide text-indicator-success">resolved</span>}
        <div className="ml-auto flex items-center shrink-0">
          <button type="button" className={iconButton} disabled={!!busy} title="Edit" aria-label="Edit note" onClick={() => setEditing(true)}>
            <Pencil size={11} />
          </button>
          <button
            type="button"
            className={iconButton}
            disabled={!!busy}
            title={annotation.resolved ? 'Reopen' : 'Mark resolved'}
            aria-label={annotation.resolved ? 'Reopen note' : 'Mark note resolved'}
            onClick={() => run('resolve', () => actions.update(annotation.id, { resolved: !annotation.resolved }))}
          >
            {busy === 'resolve' ? <Loader2 size={11} className="animate-spin" /> : annotation.resolved ? <RotateCcw size={11} /> : <Check size={11} />}
          </button>
          <button
            type="button"
            className={`${iconButton} hover:text-chip-red`}
            disabled={!!busy}
            title="Delete"
            aria-label="Delete note"
            onClick={() => {
              if (window.confirm('Delete this note?')) void run('delete', () => actions.remove(annotation.id))
            }}
          >
            {busy === 'delete' ? <Loader2 size={11} className="animate-spin" /> : <Trash2 size={11} />}
          </button>
        </div>
      </div>
      {showTarget && (
        onFocus ? (
          <button
            type="button"
            className="text-[10px] text-accent-blue hover:underline truncate max-w-full text-left"
            onClick={() => onFocus(annotation)}
            title="Show on the canvas"
          >{describeTarget(annotation.target, scenes)}</button>
        ) : <div className="text-[10px] text-text-muted truncate">{describeTarget(annotation.target, scenes)}</div>
      )}
      <p className="mt-0.5 text-[11px] text-text-primary whitespace-pre-wrap break-words">{annotation.text}</p>
      {error && <p role="alert" className="text-[10px] text-chip-red">{error}</p>}
    </li>
  )
}

/** A list of notes, or a hint when there are none. */
export function AnnotationList({ annotations, actions, allowTimecode, showTarget, scenes, onFocus, empty }: {
  annotations: RunAnnotation[]
  actions: AnnotationActions
  allowTimecode: boolean
  showTarget?: boolean
  scenes?: SceneGroup[]
  onFocus?: (annotation: RunAnnotation) => void
  empty?: string
}) {
  if (annotations.length === 0) {
    return empty ? <p className="text-[11px] text-text-muted">{empty}</p> : null
  }
  return (
    <ul className="space-y-1.5">
      {annotations.map(annotation => (
        <AnnotationItem
          key={annotation.id}
          annotation={annotation}
          actions={actions}
          allowTimecode={allowTimecode}
          showTarget={showTarget}
          scenes={scenes}
          onFocus={onFocus}
        />
      ))}
    </ul>
  )
}
