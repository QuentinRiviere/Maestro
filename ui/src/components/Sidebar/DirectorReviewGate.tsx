import { useState } from 'react'
import { Check, Loader2, RefreshCw, Square } from 'lucide-react'
import {
  directorCastImageUrl, getFileUrl, regenerateDirectorCastEntry,
  type DirectorCastEntry, type DirectorCastKind,
} from '../../api/client'
import { useStore } from '../../stores/useStore'

/** One cast entry; the parent re-keys it when the server's text changes. */
function CastCard({ pid, kind, index, entry }: {
  pid: string
  kind: DirectorCastKind
  index: number
  entry: DirectorCastEntry
}) {
  const [description, setDescription] = useState(entry.description)
  const [wardrobe, setWardrobe] = useState(entry.wardrobe ?? '')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const regenerate = async () => {
    setBusy(true)
    setError(null)
    try {
      // The status poll picks up the new image and text.
      await regenerateDirectorCastEntry(pid, kind, index, { description, wardrobe })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Regenerate failed')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="rounded-lg border border-border bg-bg-tertiary/40 p-2 space-y-1.5">
      <div className="aspect-video rounded bg-bg-tertiary overflow-hidden flex items-center justify-center">
        {busy
          ? <Loader2 size={16} className="animate-spin text-text-muted" />
          : entry.image
            ? <img src={directorCastImageUrl(pid, kind, index, entry.image)} alt={entry.name} className="w-full h-full object-contain" />
            : <span className="text-[10px] text-text-muted">No image</span>}
      </div>
      <div className="text-[11px] font-medium text-text-primary">
        {entry.name}{entry.role ? <span className="text-text-muted font-normal"> — {entry.role}</span> : null}
      </div>
      <textarea
        value={description}
        onChange={e => setDescription(e.target.value)}
        rows={3}
        aria-label={`${entry.name} description`}
        className="w-full bg-bg-tertiary border border-border rounded px-2 py-1 text-[10px] text-text-primary focus:outline-none focus:border-accent-blue"
      />
      {kind === 'character' && (
        <input
          value={wardrobe}
          onChange={e => setWardrobe(e.target.value)}
          placeholder="Wardrobe"
          aria-label={`${entry.name} wardrobe`}
          className="w-full bg-bg-tertiary border border-border rounded px-2 py-1 text-[10px] text-text-primary focus:outline-none focus:border-accent-blue"
        />
      )}
      <button
        onClick={() => void regenerate()}
        disabled={busy}
        className="w-full py-1 rounded border border-border text-[10px] text-text-secondary hover:text-text-primary hover:bg-bg-hover flex items-center justify-center gap-1 disabled:opacity-50"
      >
        <RefreshCw size={10} /> Regenerate
      </button>
      {error && <p className="text-[10px] text-chip-red">{error}</p>}
    </div>
  )
}

/** Validation stops of a server-side Director run: the cast, then the storyboard. */
export function DirectorReviewGate() {
  const status = useStore(s => s.pipelineStatus)
  const pid = useStore(s => s.pipelineId)
  const workspace = useStore(s => s.activeWorkspace)
  const continuePipeline = useStore(s => s.continuePipeline)
  const stopPipeline = useStore(s => s.stopPipeline)
  const [submittedFor, setSubmittedFor] = useState<string | null>(null)

  const reason = status?.status === 'paused' ? status.pause_reason : null
  const submitting = reason !== null && submittedFor === `${pid}:${reason}`
  const cast = status?.cast

  if (!pid || (reason !== 'review_cast' && reason !== 'review_images')) return null

  const proceed = async () => {
    setSubmittedFor(`${pid}:${reason}`)
    await continuePipeline()
  }

  const actions = (label: string) => (
    <div className="flex gap-2">
      <button
        onClick={() => void proceed()}
        disabled={submitting}
        className="flex-1 py-2 rounded-lg bg-accent-green hover:bg-accent-green-hover text-white text-xs font-semibold flex items-center justify-center gap-1.5 disabled:opacity-50"
      >
        {submitting ? <Loader2 size={12} className="animate-spin" /> : <Check size={12} />} {label}
      </button>
      <button
        onClick={() => void stopPipeline()}
        className="px-3 py-2 rounded-lg border border-border text-xs text-text-secondary hover:text-text-primary flex items-center gap-1"
      >
        <Square size={10} /> Stop
      </button>
    </div>
  )

  if (reason === 'review_cast') {
    const entries: Array<[DirectorCastKind, DirectorCastEntry, number]> = [
      ...(cast?.characters ?? []).map((entry, i) => ['character', entry, i] as [DirectorCastKind, DirectorCastEntry, number]),
      ...(cast?.locations ?? []).map((entry, i) => ['location', entry, i] as [DirectorCastKind, DirectorCastEntry, number]),
    ]
    return (
      <div className="space-y-2">
        <p className="text-xs text-text-primary font-medium">Review the cast</p>
        <p className="text-[10px] text-text-muted">
          Every shot will describe and show these people and places. Edit a description and regenerate until it is right.
        </p>
        <div className="grid grid-cols-2 gap-2">
          {entries.map(([kind, entry, index]) => (
            <CastCard
              key={`${kind}-${index}-${entry.description}-${entry.wardrobe ?? ''}`}
              pid={pid} kind={kind} index={index} entry={entry}
            />
          ))}
        </div>
        {actions('Approve cast and plan shots')}
      </div>
    )
  }

  const plans = status?.clip_plans ?? []
  const images = status?.clip_images ?? []
  return (
    <div className="space-y-2">
      <p className="text-xs text-text-primary font-medium">Review the storyboard</p>
      <p className="text-[10px] text-text-muted">
        One start image per shot. Video rendering starts only after you approve.
      </p>
      <div className="grid grid-cols-2 gap-2">
        {plans.map((plan, i) => (
          <div key={i} className="rounded-lg border border-border bg-bg-tertiary/40 p-1.5 space-y-1">
            <div className="aspect-video rounded bg-bg-tertiary overflow-hidden flex items-center justify-center">
              {images[i]
                ? <img src={getFileUrl(images[i], status?.workspace || workspace)} alt={`Shot ${i + 1}`} className="w-full h-full object-cover" />
                : <span className="text-[10px] text-text-muted">No image</span>}
            </div>
            <p className="text-[10px] text-text-secondary line-clamp-3">
              <span className="text-text-primary font-medium">{i + 1}.</span> {plan.image_prompt || plan.video_prompt}
            </p>
          </div>
        ))}
      </div>
      {actions('Approve storyboard and render video')}
    </div>
  )
}
