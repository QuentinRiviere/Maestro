import { useEffect, useState, type MouseEvent } from 'react'
import { Check, ChevronLeft, ChevronRight, Loader2, Sparkles, X } from 'lucide-react'
import { getFileUrl } from '../../api/client'
import type { StoryboardDraftCandidate, StoryboardDrafts } from '../../types'
import { chosenDraftFile, type RunShot } from '../../lib/directorRun'

/** Switching a shot's chosen draft: the guard and the call. */
export interface DraftActions {
  /** Why the chosen draft cannot be changed right now, or null. */
  blockedReason: string | null
  /** Re-render the shot's start image from the candidate `file`. */
  select: (shot: number, file: string) => Promise<void>
}

function draftLabel(candidate: StoryboardDraftCandidate): string {
  return `r${candidate.round} c${candidate.candidate}`
}

function draftIssues(candidate: StoryboardDraftCandidate): string[] {
  return (candidate.issues || []).filter(Boolean)
}

function draftTitle(candidate: StoryboardDraftCandidate, chosen: boolean): string {
  const verdict = candidate.verdict || 'unranked'
  return [
    `Draft ${draftLabel(candidate)} · ${verdict}${chosen ? ' · chosen' : ''}`,
    candidate.seed != null ? `seed ${candidate.seed}` : '',
    ...draftIssues(candidate).map(issue => `• ${issue}`),
  ].filter(Boolean).join('\n')
}

const VERDICT_DOT: Record<string, string> = {
  pass: 'bg-indicator-success',
  flagged: 'bg-indicator-warning',
}

const VERDICT_CHIP: Record<string, string> = {
  pass: 'bg-green-500/15 text-indicator-success border-green-500/30',
  flagged: 'bg-amber-500/15 text-indicator-warning border-amber-500/40',
}

function VerdictChip({ verdict }: { verdict: string | undefined }) {
  const tone = VERDICT_CHIP[verdict || ''] || 'bg-bg-tertiary text-text-muted border-border'
  return (
    <span className={`px-1.5 py-0.5 rounded border text-[9px] font-medium uppercase tracking-wide ${tone}`}>
      {verdict || 'unranked'}
    </span>
  )
}

/**
 * Compact strip of a shot's draft candidates, all rounds in order: a
 * thumbnail each, labelled "r{round} c{candidate}", a pass/flagged dot, and
 * the chosen one outlined. Clicking a thumbnail previews it large.
 */
export function DraftStrip({ drafts, aspect, onPreview }: {
  drafts: StoryboardDrafts
  /** CSS aspect ratio of the frames ("9 / 16"). */
  aspect: string
  onPreview?: (candidateIndex: number) => void
}) {
  const chosen = chosenDraftFile(drafts)
  return (
    <div
      className="nodrag nopan nowheel flex items-stretch gap-0.5 h-[36px] shrink-0 overflow-x-auto overflow-y-hidden [scrollbar-width:none]"
      aria-label={`${drafts.candidates.length} storyboard drafts`}
    >
      {drafts.candidates.map((candidate, index) => {
        const isChosen = !!chosen && candidate.file === chosen
        const label = draftLabel(candidate)
        return (
          <button
            key={`${candidate.file}-${index}`}
            type="button"
            className={`relative shrink-0 h-full rounded-sm overflow-hidden bg-media-canvas border ${isChosen
              ? 'border-accent-blue ring-1 ring-accent-blue'
              : 'border-border hover:border-border-light'} focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent-blue focus-visible:-outline-offset-2`}
            style={{ aspectRatio: aspect, minWidth: 20 }}
            title={draftTitle(candidate, isChosen)}
            aria-label={`Preview draft ${label}${isChosen ? ' (chosen)' : ''}`}
            onClick={(event: MouseEvent) => { event.stopPropagation(); onPreview?.(index) }}
          >
            <img src={getFileUrl(candidate.file)} alt="" loading="lazy" className="absolute inset-0 w-full h-full object-cover" />
            {candidate.verdict && (
              <span
                className={`absolute top-0.5 right-0.5 w-1.5 h-1.5 rounded-full ring-1 ring-black/60 ${VERDICT_DOT[candidate.verdict] || 'bg-text-muted/60'}`}
                aria-hidden
              />
            )}
            <span className="absolute left-0 right-0 bottom-0 px-0.5 text-[7px] leading-[9px] font-mono text-white text-center bg-gradient-to-t from-black/75 to-transparent [text-shadow:0_1px_1px_rgba(0,0,0,0.8)]">
              {label}
            </span>
          </button>
        )
      })}
    </div>
  )
}

/**
 * Full-screen preview of one draft candidate, with the shot's other
 * candidates to step through and a "Use this draft" action that re-renders
 * the start image from a non-chosen candidate.
 */
export function DraftLightbox({ shot, index, aspect, actions, onIndex, onClose }: {
  shot: RunShot
  index: number
  aspect: string
  actions?: DraftActions
  onIndex: (index: number) => void
  onClose: () => void
}) {
  const drafts = shot.drafts
  const candidates = drafts?.candidates || []
  const candidate = candidates[index]
  const chosen = chosenDraftFile(drafts)
  const [confirming, setConfirming] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // A different candidate drops a pending confirmation.
  useEffect(() => { setConfirming(false); setError(null) }, [index, shot.index])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (busy) return
      if (event.key === 'ArrowLeft' && index > 0) onIndex(index - 1)
      else if (event.key === 'ArrowRight' && index < candidates.length - 1) onIndex(index + 1)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [busy, index, candidates.length, onIndex])

  if (!drafts || !candidate) return null
  const isChosen = !!chosen && candidate.file === chosen
  const label = draftLabel(candidate)
  const issues = draftIssues(candidate)
  const round = (drafts.rounds || []).find(r => r.round === candidate.round)
  const blocked = actions?.blockedReason || null

  const applyDraft = async () => {
    if (!actions || isChosen) return
    if (!confirming) { setConfirming(true); return }
    setBusy(true)
    setError(null)
    try {
      await actions.select(shot.index, candidate.file)
      onClose()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setConfirming(false)
    } finally {
      setBusy(false)
    }
  }

  const navButton = 'p-1.5 rounded-full bg-black/50 text-white hover:bg-black/70 disabled:opacity-30 disabled:pointer-events-none'

  return (
    <div
      className="fixed inset-0 z-[80] bg-black/80 flex items-stretch"
      role="dialog"
      aria-modal="true"
      aria-label={`Shot ${shot.index + 1} draft ${label}`}
      onClick={busy ? undefined : onClose}
    >
      <div className="relative flex-1 min-w-0 flex items-center justify-center p-6 gap-3" onClick={event => event.stopPropagation()}>
        <button
          type="button"
          className={navButton}
          aria-label="Previous draft"
          disabled={busy || index === 0}
          onClick={() => onIndex(index - 1)}
        ><ChevronLeft size={18} /></button>
        <div className="relative max-w-full max-h-full flex items-center justify-center">
          <img
            src={getFileUrl(candidate.file)}
            alt={`Shot ${shot.index + 1} draft ${label}`}
            className="max-h-[85vh] max-w-full object-contain rounded-lg border border-border bg-media-canvas"
            style={{ aspectRatio: aspect }}
          />
          <span className="absolute top-2 left-2 px-1.5 py-0.5 rounded bg-black/65 text-white text-[11px] font-mono">{label}</span>
        </div>
        <button
          type="button"
          className={navButton}
          aria-label="Next draft"
          disabled={busy || index >= candidates.length - 1}
          onClick={() => onIndex(index + 1)}
        ><ChevronRight size={18} /></button>
      </div>

      <aside
        className="w-[300px] max-w-[45%] shrink-0 h-full bg-bg-secondary border-l border-border flex flex-col"
        onClick={event => event.stopPropagation()}
      >
        <header className="px-3 py-2 border-b border-border flex items-center gap-2 shrink-0">
          <h3 className="text-xs font-semibold text-text-primary">Shot {shot.index + 1} · draft {label}</h3>
          <VerdictChip verdict={candidate.verdict} />
          <button
            type="button"
            className="ml-auto p-0.5 rounded text-text-muted hover:text-text-primary hover:bg-bg-hover disabled:opacity-40"
            aria-label="Close"
            disabled={busy}
            onClick={onClose}
          ><X size={13} /></button>
        </header>

        <div className="flex-1 min-h-0 overflow-y-auto p-3 space-y-3 text-[11px]">
          <div className="space-y-0.5 text-text-secondary">
            {isChosen && (
              <p className="flex items-center gap-1 text-accent-blue">
                <Check size={11} /> Chosen draft{drafts.chosen?.override ? ' (your pick)' : ' (judge\'s pick)'}
              </p>
            )}
            <p>Round {candidate.round}, candidate {candidate.candidate} of {candidates.length} drafts{drafts.resolution ? ` · ${drafts.resolution}` : ''}</p>
            {candidate.seed != null && <p className="font-mono text-text-muted">seed {candidate.seed}</p>}
            {drafts.judge && <p className="text-text-muted">Ranked by: {drafts.judge === 'vision' ? 'vision judge' : drafts.judge}</p>}
          </div>
          {issues.length > 0 ? (
            <div>
              <div className="text-[10px] text-text-muted uppercase tracking-wider font-medium mb-1">Issues</div>
              <ul className="list-disc pl-4 text-text-secondary space-y-0.5">
                {issues.map((issue, i) => <li key={i}>{issue}</li>)}
              </ul>
            </div>
          ) : candidate.verdict ? <p className="text-text-muted">No issues reported.</p> : null}
          {round?.fix_hint && (
            <p className="text-text-muted">Fix hint after this round: {round.fix_hint}</p>
          )}
          {round?.hint && (
            <p className="text-text-muted">Rendered with hint: {round.hint}</p>
          )}
          {error && <p role="alert" className="text-chip-red break-words">{error}</p>}
        </div>

        <div className="px-3 py-2 border-t border-border shrink-0">
          <div className="flex gap-0.5 mb-2 overflow-x-auto [scrollbar-width:none]">
            {candidates.map((c, i) => (
              <button
                key={`${c.file}-${i}`}
                type="button"
                className={`relative shrink-0 h-[44px] rounded-sm overflow-hidden bg-media-canvas border ${i === index
                  ? 'border-accent-blue ring-1 ring-accent-blue'
                  : c.file === chosen ? 'border-accent-blue/60' : 'border-border hover:border-border-light'}`}
                style={{ aspectRatio: aspect, minWidth: 24 }}
                title={draftTitle(c, c.file === chosen)}
                aria-label={`Show draft ${draftLabel(c)}`}
                aria-current={i === index}
                disabled={busy}
                onClick={() => onIndex(i)}
              >
                <img src={getFileUrl(c.file)} alt="" loading="lazy" className="absolute inset-0 w-full h-full object-cover" />
              </button>
            ))}
          </div>
          {actions && !isChosen && (
            <button
              type="button"
              onClick={() => void applyDraft()}
              disabled={busy || !!blocked}
              title={blocked || (confirming
                ? 'Re-render the start image from this draft (the shot\'s clip becomes stale)'
                : 'Make this the chosen draft and re-render the start image')}
              className={`w-full flex items-center justify-center gap-1 px-2.5 py-1.5 rounded border text-[11px] disabled:opacity-40 ${confirming
                ? 'border-amber-500/60 bg-amber-500/15 text-indicator-warning hover:bg-amber-500/25'
                : 'border-accent-blue bg-accent-blue/15 text-accent-blue hover:bg-accent-blue/25'}`}
            >
              {busy ? <Loader2 size={11} className="animate-spin" /> : <Sparkles size={11} />}
              {busy ? 'Re-rendering…' : confirming ? 'Confirm? Re-render from this draft' : 'Use this draft'}
            </button>
          )}
          {actions && !isChosen && blocked && <p className="mt-1 text-[10px] text-text-muted">{blocked}</p>}
        </div>
      </aside>
    </div>
  )
}
