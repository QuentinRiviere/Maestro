import { lazy, Suspense, useCallback, useMemo, useState, type ReactNode } from 'react'
import { Loader2, Pause, Play, RefreshCw, Square } from 'lucide-react'
import {
  continuePipeline as apiContinuePipeline, directorCastImageUrl, stopPipeline as apiStopPipeline, updateClipPrompt,
  type DirectorCastKind,
} from '../../api/client'
import { useStore } from '../../stores/useStore'
import type { SavedPipelineState } from '../../types'
import {
  buildRunShots, finalEditFile, groupScenes, isRunActive, type RunShot,
} from '../../lib/directorRun'
import type { CastRef } from '../../lib/directorDag'
import type { DirectorView } from '../../lib/directorRoute'
import { ShotDrawer, type PromptUpdate, type ShotActions } from './ShotDrawer'
import { StoryboardView } from './StoryboardView'
import { frameAspect } from './runStyles'
import { useLiveRun, useRunCosts } from './useLiveRun'

const CanvasView = lazy(() => import('./CanvasView'))

type StagedEdit = { image_prompt?: string; video_prompt?: string; window_prompts?: string[] }

const PHASE_LABEL: Record<string, string> = {
  resuming: 'Resuming', casting: 'Casting', planning: 'Planning', polishing_prompts: 'Polishing prompts',
  generating_images: 'Start images', preparing_video: 'Preparing video', generating_video: 'Video renders',
  post_processing: 'Final edit', completed: 'Completed', failed: 'Failed', cancelled: 'Cancelled',
}

function castRefs(pid: string, kind: DirectorCastKind, list: unknown[] | undefined): CastRef[] {
  return (list || []).map((entry, index) => {
    const e = (entry && typeof entry === 'object' ? entry : {}) as { name?: unknown; image?: unknown }
    const image = typeof e.image === 'string' && e.image ? directorCastImageUrl(pid, kind, index, e.image) : null
    return { name: typeof e.name === 'string' ? e.name : '', image }
  })
}

function applyStaged(shot: RunShot, edit: StagedEdit | undefined): RunShot {
  if (!edit) return shot
  return {
    ...shot,
    imagePrompt: edit.image_prompt ?? shot.imagePrompt,
    videoPrompt: edit.window_prompts ? edit.window_prompts.join('\n') : edit.video_prompt ?? shot.videoPrompt,
    windowPrompts: edit.window_prompts ?? shot.windowPrompts,
  }
}

/**
 * One run's storyboard or canvas, with a live status bar while it runs and
 * the shot drawer. Keyed by pipeline id by the parent, so staged edits and
 * the open shot never leak from one run to another.
 */
export function RunWorkspace({ pipeline, view, busy, details }: {
  pipeline: SavedPipelineState
  view: Exclude<DirectorView, 'runs'>
  /** A repair or another long dashboard operation is running. */
  busy: boolean
  /** The legacy details view (LLM log, review, clip cards). */
  details: ReactNode
}) {
  const pid = pipeline.pipeline_id
  const { live, refresh } = useLiveRun(pid, pipeline.status)
  const chatPipelineId = useStore(s => s.pipelineId)
  const storeContinue = useStore(s => s.continuePipeline)
  const storeStop = useStore(s => s.stopPipeline)
  const rerunClipImage = useStore(s => s.rerunClipImage)
  const rerunClipVideo = useStore(s => s.rerunClipVideo)
  const tagClip = useStore(s => s.tagClip)
  const refreshSaved = useStore(s => s.refreshSavedPipeline)
  const [openShot, setOpenShot] = useState<number | null>(null)
  const [staged, setStaged] = useState<Record<number, StagedEdit>>({})
  const [actionBusy, setActionBusy] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)

  const liveActive = !!live && !live.recovered_from_disk && isRunActive(live.status)
  const paused = liveActive && live?.status === 'paused'
  const runStatus = liveActive ? String(live?.status) : pipeline.status
  const phase = liveActive ? String(live?.phase || '') : String(live?.phase || pipeline.status)
  const costs = useRunCosts(pid, `${runStatus}:${phase}:${pipeline.output_files?.length || 0}`)

  const shots = useMemo(() => buildRunShots(pipeline, live).map(shot => applyStaged(shot, staged[shot.index])),
    [pipeline, live, staged])
  const scenes = useMemo(() => groupScenes(shots), [shots])
  const stagedShots = useMemo(() => new Set(Object.keys(staged).map(Number)), [staged])
  const cast = (liveActive ? live?.cast : null) || pipeline.cast || live?.cast || null
  const characters = useMemo(() => castRefs(pid, 'character', cast?.characters), [pid, cast])
  const locations = useMemo(() => castRefs(pid, 'location', cast?.locations), [pid, cast])
  const finalFile = finalEditFile(liveActive ? live?.output_files : pipeline.output_files)
  const aspect = frameAspect(pipeline.director_aspect_ratio)
  const needsImage = !pipeline.shot_image_policy || pipeline.shot_image_policy === 'generate'

  const runAction = async (label: string, action: () => Promise<void>) => {
    setActionBusy(label)
    setActionError(null)
    try {
      await action()
      await refresh()
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e))
    } finally {
      setActionBusy(null)
    }
  }

  const continueRun = () => runAction('continue', async () => {
    const indices = Object.keys(staged).map(Number)
    let updates: { clip_plans?: Array<{ video_prompt: string; image_prompt: string }> } | undefined
    if (indices.length > 0 && live?.clip_plans?.length) {
      // Send the full live plans (they carry the Director contracts) with
      // the staged prompt edits merged in.
      const plans = live.clip_plans.map((plan, i) => {
        const edit = staged[i]
        if (!edit) return plan
        const merged: Record<string, unknown> = { ...plan, ...edit }
        if (edit.window_prompts) merged.video_prompt = edit.window_prompts.join('\n')
        return merged
      })
      updates = { clip_plans: plans as unknown as Array<{ video_prompt: string; image_prompt: string }> }
    }
    if (pid === chatPipelineId) await storeContinue(updates)
    else await apiContinuePipeline(pid, updates)
    setStaged({})
  })

  const stopRun = () => runAction('stop', async () => {
    if (pid === chatPipelineId) await storeStop()
    else await apiStopPipeline(pid)
  })

  const savePrompt = useCallback(async (index: number, update: PromptUpdate) => {
    if (paused) {
      setStaged(current => ({ ...current, [index]: { ...current[index], ...update } }))
      return
    }
    if (liveActive) throw new Error('Prompts can be edited while the run is paused or after it ends.')
    await updateClipPrompt(pid, index, update)
    await refreshSaved(pid)
  }, [paused, liveActive, pid, refreshSaved])

  const actions: ShotActions = {
    editBlockedReason: liveActive && !paused
      ? 'The run is rendering: prompts can be edited while it is paused (review gate) or after it ends.'
      : busy ? 'A repair is running on this project.' : null,
    editsAreStaged: paused,
    rerunBlockedReason: liveActive
      ? 'Reruns are available once the run has ended.'
      : busy ? 'Wait for the repair to finish.' : null,
    needsImage,
    savePrompt,
    rerunImage: async (index, prompt) => { await rerunClipImage(pid, index, prompt) },
    rerunVideo: async (index, prompt) => { await rerunClipVideo(pid, index, prompt) },
    tag: async (index, tag) => { await tagClip(pid, index, tag) },
  }

  const progress = live?.progress
  const shot = openShot != null ? shots[openShot] : null
  const stagedCount = Object.keys(staged).length

  return (
    <div className="flex flex-col gap-3 h-full min-h-0">
      {(liveActive || actionError) && (
        <div className={`shrink-0 rounded-lg border px-3 py-2 flex flex-wrap items-center gap-2 text-[11px] ${paused
          ? 'border-amber-500/40 bg-amber-500/10' : 'border-accent-blue/40 bg-accent-blue/10'}`}>
          {liveActive && (
            <>
              {paused ? <Pause size={12} className="text-indicator-warning" /> : <Loader2 size={12} className="animate-spin text-accent-blue" />}
              <span className="font-semibold text-text-primary">{paused ? 'Paused' : PHASE_LABEL[phase] || phase}</span>
              <span className="text-text-secondary min-w-0 truncate max-w-[50ch]" title={progress?.message || ''}>
                {paused ? live?.pause_reason || 'Waiting for your review' : progress?.message}
              </span>
              {progress?.total ? (
                <span className="text-text-muted tabular-nums">{progress.current ?? 0}/{progress.total}</span>
              ) : null}
              <span className="text-[10px] text-text-muted">live · refreshes every 3 s</span>
              <div className="ml-auto flex items-center gap-1.5">
                {paused && (
                  <button
                    onClick={continueRun}
                    disabled={!!actionBusy}
                    className="flex items-center gap-1 px-2 py-1 rounded bg-green-500/15 border border-green-500/40 text-indicator-success hover:bg-green-500/25 disabled:opacity-40"
                    title={stagedCount ? `Continue with ${stagedCount} edited shot prompt(s)` : 'Approve and continue the run'}
                  >
                    {actionBusy === 'continue' ? <Loader2 size={11} className="animate-spin" /> : <Play size={11} />}
                    Continue{stagedCount ? ` (${stagedCount} edited)` : ''}
                  </button>
                )}
                <button
                  onClick={stopRun}
                  disabled={!!actionBusy}
                  className="flex items-center gap-1 px-2 py-1 rounded bg-red-500/10 border border-red-500/30 text-red-400 hover:bg-red-500/20 disabled:opacity-40"
                  title="Stop this run"
                >
                  {actionBusy === 'stop' ? <Loader2 size={11} className="animate-spin" /> : <Square size={11} />} Stop
                </button>
                <button onClick={() => void refresh()} className="p-1 rounded hover:bg-bg-hover text-text-muted" title="Refresh now">
                  <RefreshCw size={12} />
                </button>
              </div>
            </>
          )}
          {actionError && <span role="alert" className="basis-full text-chip-red">{actionError}</span>}
        </div>
      )}

      <div className="flex-1 min-h-0">
        {view === 'details' ? details : view === 'canvas' ? (
          <Suspense fallback={<div className="py-16 flex items-center justify-center text-text-muted text-xs"><Loader2 size={14} className="animate-spin mr-2" />Loading canvas…</div>}>
            <CanvasView
              shots={shots}
              scenes={scenes}
              characters={characters}
              locations={locations}
              brief={pipeline.scene_description}
              finalFile={finalFile}
              runStatus={runStatus}
              phase={phase}
              costUsd={costs?.usd ?? null}
              onOpenShot={setOpenShot}
            />
          </Suspense>
        ) : (
          <div className="h-full overflow-y-auto pr-1">
            <StoryboardView
              scenes={scenes}
              aspect={aspect}
              stagedShots={stagedShots}
              onOpenShot={setOpenShot}
              emptyHint={liveActive ? 'Planning the storyboard… shots appear here as soon as the plan is ready.' : 'This run has no planned shots.'}
            />
          </div>
        )}
      </div>

      {shot && (
        <ShotDrawer
          key={shot.index}
          shot={shot}
          aspect={pipeline.director_aspect_ratio}
          stagedNote={stagedShots.has(shot.index)}
          actions={actions}
          onClose={() => setOpenShot(null)}
        />
      )}
    </div>
  )
}
