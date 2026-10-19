import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { Loader2, X } from 'lucide-react'
import {
  fetchStyleReference, styleReferenceStillUrl, styleReferenceVideoUrl,
  type StyleReference, type StyleReferenceFrame, type StyleReferenceShot,
} from '../../api/directorBibles'
import { formatSeconds, isAnalysing, SHOT_SIZE_LABELS } from '../../lib/styleReference'

const chip = 'rounded border border-border px-1 py-px text-[9px] text-text-secondary'

/** The shots of a reference as the analysis describes them, each with its frame; a click plays the shot. */
export function StyleReferenceShots({ bibleId, reference, onClose }: {
  bibleId: string
  reference: StyleReference
  onClose: () => void
}) {
  const dialog = useRef<HTMLDialogElement>(null)
  const video = useRef<HTMLVideoElement>(null)
  const [shots, setShots] = useState<StyleReferenceShot[]>([])
  const [frames, setFrames] = useState<StyleReferenceFrame[]>([])
  const [framesStatus, setFramesStatus] = useState(reference.frames_status || 'idle')
  const [status, setStatus] = useState(reference.status)
  const [progress, setProgress] = useState(reference.progress)
  const [describedOnly, setDescribedOnly] = useState(true)
  const [error, setError] = useState('')
  const [current, setCurrent] = useState<number | null>(null)

  useEffect(() => {
    const element = dialog.current
    element?.showModal()
    return () => element?.close()
  }, [])

  useEffect(() => {
    let alive = true
    const load = () => fetchStyleReference(bibleId, reference.id).then(loaded => {
      if (!alive) return
      setShots(loaded.shots || [])
      setFrames(loaded.frames || [])
      setFramesStatus(loaded.frames_status || 'idle')
      setStatus(loaded.status)
      setProgress(loaded.progress)
    }).catch(e => { if (alive) setError(e instanceof Error ? e.message : 'Could not load the shots') })
    void load()
    const live = isAnalysing(status) || framesStatus === 'queued' || framesStatus === 'running'
    const timer = window.setInterval(() => { if (live) void load() }, 5000)
    return () => { alive = false; window.clearInterval(timer) }
  }, [bibleId, reference.id, status, framesStatus])

  const framesOf = (index: number) => frames.filter(f => f.index === index)

  const described = shots.filter(s => s.shot_size)
  const shown = describedOnly ? described : shots
  const play = (shot: StyleReferenceShot) => {
    setCurrent(shot.index)
    const element = video.current
    if (!element) return
    element.currentTime = shot.start
    void element.play()
  }
  // Stop at the end of the shot being played.
  const onTime = () => {
    const element = video.current
    if (!element || current === null) return
    const shot = shots[current]
    if (shot && element.currentTime >= shot.end) element.pause()
  }

  return createPortal(
    <dialog ref={dialog} onCancel={onClose} aria-labelledby="style-shots-title"
      className="m-auto max-h-[92dvh] w-[900px] max-w-[calc(100vw-24px)] overflow-y-auto rounded-xl border border-border bg-bg-secondary p-0 text-text-primary shadow-2xl backdrop:bg-black/70">
      <div className="sticky top-0 z-10 border-b border-border bg-bg-secondary">
      <div className="flex items-center justify-between gap-2 p-3 pb-2">
        <div className="min-w-0">
          <h2 id="style-shots-title" className="truncate text-sm font-semibold">{reference.label}</h2>
          <p className="text-[10px] text-text-muted">
            {described.length} of {shots.length} shots described
            {frames.length > 0 && <> · {frames.length} frame prompts</>}
            {isAnalysing(status) && <span className="text-accent-blue"> · <Loader2 size={10} className="inline animate-spin" /> {progress || 'analysing'}</span>}
            {(framesStatus === 'queued' || framesStatus === 'running') && <span className="text-accent-blue"> · <Loader2 size={10} className="inline animate-spin" /> {reference.frames_progress || 'frame prompts'}</span>}
          </p>
        </div>
        <label className="flex shrink-0 items-center gap-1 text-[10px] text-text-secondary">
          <input type="checkbox" className="h-3 w-3" checked={describedOnly} onChange={e => setDescribedOnly(e.target.checked)} />
          Described only
        </label>
        <button type="button" onClick={onClose} aria-label="Close shots" className="p-1"><X size={16} /></button>
      </div>
      {/* The player stays in view while the list scrolls, so the shot being played is always visible. */}
      <div className="px-3 pb-2">
        <video ref={video} className="mx-auto max-h-56 w-full rounded bg-black" controls preload="metadata" onTimeUpdate={onTime}
          src={styleReferenceVideoUrl(bibleId, reference.id)} aria-label={`${reference.label} video`} />
      </div>
      </div>
      <div className="p-3 space-y-3">
        {error && <p role="alert" className="text-xs text-chip-red">{error}</p>}
        {!shots.length && !error && <p className="text-xs text-text-muted">No shot yet: the cuts are being measured.</p>}
        <ul className="space-y-1.5" aria-label="Shots">
          {shown.map(shot => (
            <li key={shot.index}
              className={`flex gap-2 rounded border p-1.5 ${current === shot.index ? 'border-accent-blue' : 'border-border'} ${shot.credits ? 'opacity-60' : ''}`}>
              <button type="button" className="shrink-0" title="Play this shot" onClick={() => play(shot)}>
                <img loading="lazy" alt={`Shot ${shot.index + 1}`} width={128} height={72}
                  className="h-[72px] w-[128px] rounded bg-black object-contain"
                  src={styleReferenceStillUrl(bibleId, reference.id, shot.start + shot.duration / 2)} />
              </button>
              <div className="min-w-0 flex-1 space-y-0.5 text-[10px]">
                <p className="text-text-secondary">
                  <span className="font-medium text-text-primary">#{shot.index + 1}</span>
                  {' '}{formatSeconds(shot.start)} – {formatSeconds(shot.end)} ({shot.duration.toFixed(1)} s)
                  {shot.credits && <span className="ml-1 rounded bg-chip-red/20 px-1 text-chip-red">credits</span>}
                  {shot.false_cut && <span className="ml-1 rounded bg-bg-tertiary px-1 text-text-muted">not a cut</span>}
                </p>
                {shot.shot_size ? (
                  <>
                    <p className="flex flex-wrap gap-1">
                      <span className={chip}>{SHOT_SIZE_LABELS[shot.shot_size] ?? shot.shot_size}</span>
                      {shot.angle && <span className={chip}>{shot.angle.replace('_', ' ')}</span>}
                      {shot.movement && <span className={chip}>{shot.movement.replace('_', ' ')}</span>}
                      {shot.cut_reason && <span className={chip}>cut: {shot.cut_reason.replace('_', ' ')}</span>}
                      {shot.transition && shot.transition !== 'hard_cut' && <span className={chip}>{shot.transition.replace('_', ' ')}</span>}
                      {shot.audio && <span className={chip}>{shot.audio}</span>}
                    </p>
                    <p className="text-text-primary">{shot.subject}{shot.action ? `: ${shot.action}` : ''}</p>
                    {shot.lighting && <p className="text-text-muted">Light: {shot.lighting}</p>}
                    {shot.graphics && <p className="text-text-muted">Graphics: {shot.graphics}</p>}
                    {framesOf(shot.index).map(frame => (
                      <details key={frame.position} className="rounded border border-border/60 bg-bg-tertiary/40 px-1.5 py-0.5">
                        <summary className="cursor-pointer text-text-secondary">
                          {frame.position} frame · image prompt{frame.mood ? ` · ${frame.mood}` : ''}
                        </summary>
                        <div className="mt-1 flex gap-2">
                          <img loading="lazy" alt={`${frame.position} frame of shot ${shot.index + 1}`} width={96} height={54}
                            className="h-[54px] w-[96px] shrink-0 rounded bg-black object-contain"
                            src={styleReferenceStillUrl(bibleId, reference.id, frame.t)} />
                          <div className="min-w-0 space-y-0.5">
                            <p className="select-all text-text-primary">{frame.prompt}</p>
                            {frame.palette?.length ? <p className="text-text-muted">Palette: {frame.palette.join(', ')}</p> : null}
                          </div>
                        </div>
                      </details>
                    ))}
                  </>
                ) : <p className="text-text-muted">Not described yet.</p>}
              </div>
            </li>
          ))}
        </ul>
      </div>
    </dialog>,
    document.body,
  )
}
