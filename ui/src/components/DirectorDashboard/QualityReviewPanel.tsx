import { useCallback, useEffect, useState } from 'react'
import { Loader2, ScanEye } from 'lucide-react'
import { fetchPipelineStatus, requestPipelineReview, type VideoReview, type VideoReviewShot, type VideoReviewTake } from '../../api/client'

const SEVERITY_CLASS: Record<string, string> = {
  critical: 'text-chip-red',
  major: 'text-indicator-warning',
  minor: 'text-text-muted',
}

const FIX_LABEL: Record<string, string> = {
  retake_segment: 'retake this part',
  new_seed: 'new seed',
  prompt_revision: 'revise prompt',
  new_start_image: 'new start image',
  none_model_limit: 'model limit',
}

const ACTION_LABEL: Record<string, string> = {
  accept: 'Keep',
  retake_segment: 'Retake a segment',
  regenerate: 'Regenerate',
  regenerate_start_image: 'New start image',
}

function scoreClass(score: number | null | undefined) {
  if (score == null) return 'bg-bg-tertiary text-text-muted'
  if (score >= 8) return 'bg-green-500/15 text-indicator-success'
  if (score >= 6) return 'bg-amber-500/15 text-indicator-warning'
  return 'bg-red-500/15 text-chip-red'
}

function formatTime(seconds: number | null | undefined) {
  return seconds == null ? '' : `${seconds.toFixed(1)}s`
}

function TakeLine({ take }: { take?: VideoReviewTake }) {
  if (!take || take.takes.length < 2) return null
  const keptIndex = take.kept ? take.takes.indexOf(take.kept) : -1
  const decisive = take.comparisons.find(c => c.winner !== 'tie')
  return (
    <p className="text-[10px] text-text-muted mt-1">
      Kept take {keptIndex + 1} of {take.takes.length}
      {decisive?.reasons?.[0] ? ` — ${decisive.reasons[0]}` : take.comparisons.length ? ' — no clear winner, original kept' : ''}
    </p>
  )
}

function ShotRow({ shot, take }: { shot: VideoReviewShot; take?: VideoReviewTake }) {
  return (
    <div className="border-t border-border/60 pt-2 first:border-t-0 first:pt-0">
      <div className="flex items-center gap-2 text-[11px]">
        <span className={`px-1.5 py-0.5 rounded font-medium ${scoreClass(shot.score)}`}>
          {shot.score ?? '–'}/10
        </span>
        <span className="text-text-primary font-medium">Shot {shot.index + 1}</span>
        {shot.start_sec != null && (
          <span className="text-text-muted">{formatTime(shot.start_sec)}–{formatTime(shot.end_sec)}</span>
        )}
        {shot.action && <span className="ml-auto text-text-secondary">{ACTION_LABEL[shot.action] ?? shot.action}</span>}
      </div>
      <TakeLine take={take} />
      {shot.error && <p className="text-[10px] text-chip-red mt-1">Review failed: {shot.error}</p>}
      {shot.summary && <p className="text-[11px] text-text-secondary mt-1">{shot.summary}</p>}
      {!!shot.issues?.length && (
        <ul className="mt-1 space-y-0.5">
          {shot.issues.map((issue, i) => (
            <li key={i} className="text-[10px] leading-snug">
              <span className={`font-medium ${SEVERITY_CLASS[issue.severity] ?? ''}`}>{issue.severity}</span>
              {issue.at_sec != null && <span className="text-text-muted"> @{formatTime(issue.at_sec)}</span>}
              <span className="text-text-secondary"> — {issue.description}</span>
              <span className="text-text-muted"> ({FIX_LABEL[issue.fixable_by] ?? issue.fixable_by})</span>
            </li>
          ))}
        </ul>
      )}
      {shot.revised_prompt_hint && (
        <p className="text-[10px] text-text-muted mt-1">Suggested prompt fix: {shot.revised_prompt_hint}</p>
      )}
    </div>
  )
}

/** Read-only AI quality review of a finished Director project. */
export function QualityReviewPanel({ pipelineId, canReview }: { pipelineId: string; canReview: boolean }) {
  const [review, setReview] = useState<VideoReview | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [starting, setStarting] = useState(false)

  const load = useCallback(async () => {
    try {
      const status = await fetchPipelineStatus(pipelineId)
      setReview(status.video_review ?? null)
    } catch {
      setReview(null)
    }
  }, [pipelineId])

  useEffect(() => {
    setReview(null)
    setError(null)
    void load()
  }, [load])

  const running = starting || review?.status === 'running'
  useEffect(() => {
    if (!running) return
    const timer = window.setInterval(() => { void load() }, 4000)
    return () => window.clearInterval(timer)
  }, [running, load])

  const start = async () => {
    setStarting(true)
    setError(null)
    try {
      await requestPipelineReview(pipelineId)
      await load()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setStarting(false)
    }
  }

  return (
    <div className="bg-bg-secondary rounded-lg border border-border p-3">
      <div className="flex items-center gap-2 mb-2">
        <h3 className="text-[11px] text-text-secondary uppercase tracking-wider font-medium">Quality review</h3>
        {review?.score != null && (
          <span className={`text-[10px] px-1.5 py-0.5 rounded ${scoreClass(review.score)}`}>{review.score}/10</span>
        )}
        {review?.model && <span className="text-[10px] text-text-muted truncate">{review.model}</span>}
        {review?.cost_usd != null && review.status !== 'running' && (
          <span className="text-[10px] text-text-muted">${review.cost_usd.toFixed(3)}</span>
        )}
        {canReview && (
          <button
            onClick={start}
            disabled={running}
            title="Ask a video-capable LLM (through fal.ai) to judge each shot against its plan. About one cent per shot."
            className="ml-auto flex items-center gap-1 px-2 py-1 text-[10px] bg-bg-tertiary border border-border rounded text-text-secondary hover:text-text-primary disabled:opacity-50"
          >
            {running ? <Loader2 size={10} className="animate-spin" /> : <ScanEye size={10} />}
            {running ? 'Reviewing…' : review ? 'Review again' : 'Review quality'}
          </button>
        )}
      </div>
      {error && <p className="text-[10px] text-chip-red mb-1">{error}</p>}
      {review?.status === 'failed' && review.error && (
        <p className="text-[10px] text-chip-red mb-1">{review.error}</p>
      )}
      {!review && !error && (
        <p className="text-[10px] text-text-muted">
          No review yet. Advisory only: the review never changes the project.
        </p>
      )}
      {review?.style_contract?.medium && review.style_contract.medium !== 'unspecified' && (
        <p className="text-[10px] text-text-muted mb-2">
          Judged as <span className="text-text-secondary">{review.style_contract.medium}</span>
          {review.style_contract.allowed_stylization?.length
            ? ` (intended: ${review.style_contract.allowed_stylization.join(', ')})` : ''}
        </p>
      )}
      {review?.status === 'running' && review.phase === 'takes' && (
        <p className="text-[10px] text-text-muted mb-2">
          Rendering extra takes and keeping the best ({review.takes?.length ?? 0} shot(s) done)…
        </p>
      )}
      {!!review?.shots?.length && (
        <div className="space-y-2">
          {review.shots.map(shot => (
            <ShotRow key={shot.index} shot={shot} take={review.takes?.find(t => t.index === shot.index)} />
          ))}
        </div>
      )}
    </div>
  )
}
