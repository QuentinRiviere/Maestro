import { useRef } from 'react'
import { ImageIcon } from 'lucide-react'
import { getFileUrl } from '../../api/client'
import { judgeLabel, type StageStatus, type StoryboardJudgeShot } from '../../lib/directorRun'
import { STATUS_DOT, STATUS_LABEL } from './runStyles'
import { trimProps } from '../../lib/videoTrim'

export function StatusDot({ status, title }: { status: StageStatus; title?: string }) {
  return (
    <span
      className={`inline-block w-2 h-2 rounded-full shrink-0 ${STATUS_DOT[status]}`}
      title={title ?? STATUS_LABEL[status]}
      aria-label={title ?? STATUS_LABEL[status]}
    />
  )
}

const JUDGE_TONE = {
  ok: 'bg-green-500/15 text-indicator-success border-green-500/30',
  warn: 'bg-amber-500/15 text-indicator-warning border-amber-500/40',
  info: 'bg-blue-500/15 text-chip-blue border-blue-500/30',
  muted: 'bg-bg-tertiary text-text-muted border-border',
}

/** Storyboard judge verdict, with its issues on hover. */
export function JudgeBadge({ judge, className = '' }: { judge: StoryboardJudgeShot | null; className?: string }) {
  const verdict = judgeLabel(judge)
  if (!verdict || !judge) return null
  const issues = (judge.issues || []).filter(Boolean)
  const title = [
    `Storyboard judge: ${verdict.label}`,
    judge.attempts ? `${judge.attempts} regeneration attempt(s)` : '',
    ...issues.map(issue => `• ${issue}`),
    judge.fix_hint ? `Fix: ${judge.fix_hint}` : '',
    judge.error ? `Check error: ${judge.error}` : '',
  ].filter(Boolean).join('\n')
  return (
    <span
      className={`px-1.5 py-0.5 rounded border text-[9px] font-medium uppercase tracking-wide ${JUDGE_TONE[verdict.tone]} ${className}`}
      title={title}
    >
      {verdict.label}
    </span>
  )
}

/**
 * A shot's start frame; hovering plays the rendered clip when there is one.
 * Falls back to the clip's first frame, then to a placeholder.
 */
export function ShotMedia({ image, video, alt, aspect, keep = null, playOnHover = true }: {
  image: string | null
  video: string | null
  keep?: [number, number] | null
  alt: string
  aspect: string
  playOnHover?: boolean
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const play = () => { void videoRef.current?.play().catch(() => undefined) }
  const pause = () => {
    const el = videoRef.current
    if (!el) return
    el.pause()
    el.currentTime = keep ? keep[0] : 0
  }
  return (
    <div
      className="relative w-full overflow-hidden bg-media-canvas"
      style={{ aspectRatio: aspect }}
      onMouseEnter={playOnHover ? play : undefined}
      onMouseLeave={playOnHover ? pause : undefined}
    >
      {image ? (
        <img src={getFileUrl(image)} alt={alt} loading="lazy" className="absolute inset-0 w-full h-full object-cover" />
      ) : !video ? (
        <div className="absolute inset-0 flex items-center justify-center text-text-muted">
          <ImageIcon size={18} />
        </div>
      ) : null}
      {video && (
        <video
          ref={videoRef}
          src={`${getFileUrl(video)}#t=${keep ? keep[0] + 0.05 : 0.1}`}
          muted
          loop
          {...trimProps(keep, true)}
          playsInline
          preload="metadata"
          aria-label={`${alt} clip`}
          className={`absolute inset-0 w-full h-full object-cover transition-opacity ${image ? 'opacity-0 hover:opacity-100' : ''}`}
        />
      )}
    </div>
  )
}
