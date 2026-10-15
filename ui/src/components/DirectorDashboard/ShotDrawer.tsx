import { useEffect, useState, type ReactNode } from 'react'
import { AlertTriangle, Camera, Check, Film, Loader2, Save, X } from 'lucide-react'
import { getFileUrl } from '../../api/client'
import { formatSeconds, type RunShot } from '../../lib/directorRun'
import { JudgeBadge, StatusDot } from './runUi'
import { STATUS_LABEL, frameAspect } from './runStyles'

export interface PromptUpdate {
  image_prompt?: string
  video_prompt?: string
  window_prompts?: string[]
}

export interface ShotActions {
  /** Why prompt edits are unavailable, or null when they can be saved. */
  editBlockedReason: string | null
  /** Edits are staged for Continue instead of written to disk. */
  editsAreStaged: boolean
  /** Why reruns are unavailable, or null. */
  rerunBlockedReason: string | null
  needsImage: boolean
  savePrompt: (index: number, update: PromptUpdate) => Promise<void>
  rerunImage: (index: number, prompt?: string) => Promise<void>
  rerunVideo: (index: number, prompt?: string) => Promise<void>
  tag: (index: number, tag: 'good' | 'needs_work' | null) => Promise<void>
}

function Section({ title, children, right }: { title: string; children: ReactNode; right?: ReactNode }) {
  return (
    <section className="space-y-1.5">
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-[10px] text-text-muted uppercase tracking-wider font-medium">{title}</h3>
        {right}
      </div>
      {children}
    </section>
  )
}

function Fact({ label, value }: { label: string; value: ReactNode }) {
  if (value == null || value === '') return null
  return (
    <div className="flex gap-2 text-[11px]">
      <span className="w-24 shrink-0 text-text-muted">{label}</span>
      <span className="text-text-secondary min-w-0 break-words">{value}</span>
    </div>
  )
}

/** Right-side drawer with everything about one shot, and its edit/rerun actions. */
export function ShotDrawer({ shot, aspect, stagedNote, actions, onClose }: {
  shot: RunShot
  aspect: string | null | undefined
  stagedNote: boolean
  actions: ShotActions
  onClose: () => void
}) {
  const windowed = shot.windowPrompts.length > 1
  const [imagePrompt, setImagePrompt] = useState(shot.imagePrompt)
  const [videoPrompt, setVideoPrompt] = useState(shot.videoPrompt)
  const [windowPrompts, setWindowPrompts] = useState(shot.windowPrompts)
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  // The parent keys this drawer by shot, so local edits reset per shot;
  // a field is "dirty" while it differs from the latest saved/live value.
  const imageDirty = imagePrompt !== shot.imagePrompt
  const videoDirty = windowed
    ? windowPrompts.join('\n') !== shot.windowPrompts.join('\n')
    : videoPrompt !== shot.videoPrompt

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => { if (event.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

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

  const save = () => run('save', async () => {
    const update: PromptUpdate = {}
    if (imageDirty) update.image_prompt = imagePrompt
    if (videoDirty) {
      if (windowed) update.window_prompts = windowPrompts
      else update.video_prompt = videoPrompt
    }
    await actions.savePrompt(shot.index, update)
  })

  const editDisabled = !!actions.editBlockedReason || !!busy
  const rerunDisabled = !!actions.rerunBlockedReason || !!busy
  const videoPromptForRerun = videoDirty ? (windowed ? windowPrompts.join('\n') : videoPrompt) : undefined

  return (
    <div className="fixed inset-0 z-[70] flex justify-end" role="dialog" aria-modal="true" aria-label={`Shot ${shot.index + 1} details`}>
      <button className="absolute inset-0 bg-black/40 cursor-default" aria-label="Close shot details" onClick={onClose} />
      <aside className="relative w-[440px] max-w-full h-full bg-bg-secondary border-l border-border shadow-2xl flex flex-col">
        <header className="px-4 py-3 border-b border-border flex items-center gap-2 shrink-0">
          <h2 className="text-sm font-semibold text-text-primary">Shot {shot.index + 1}</h2>
          <span className="text-[11px] text-text-muted truncate">
            {[shot.sceneType, shot.framing, formatSeconds(shot.durationSec)].filter(v => v && v !== '--').join(' · ')}
          </span>
          <JudgeBadge judge={shot.judge} className="ml-auto" />
          <button onClick={onClose} className="p-1 rounded hover:bg-bg-hover text-text-muted" aria-label="Close">
            <X size={16} />
          </button>
        </header>

        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          <div className="grid grid-cols-2 gap-2">
            <figure className="rounded-lg overflow-hidden border border-border">
              <div className="relative bg-media-canvas" style={{ aspectRatio: frameAspect(aspect) }}>
                {shot.startImage
                  ? <img src={getFileUrl(shot.startImage)} alt={`Shot ${shot.index + 1} start frame`} className="absolute inset-0 w-full h-full object-cover" />
                  : <span className="absolute inset-0 flex items-center justify-center text-[10px] text-text-muted">No start frame</span>}
              </div>
              <figcaption className="px-2 py-1 text-[10px] text-text-muted flex items-center gap-1.5">
                <StatusDot status={shot.imageStatus} /> Start frame · {STATUS_LABEL[shot.imageStatus]}
              </figcaption>
            </figure>
            <figure className="rounded-lg overflow-hidden border border-border">
              <div className="relative bg-media-canvas" style={{ aspectRatio: frameAspect(aspect) }}>
                {shot.video
                  ? <video src={getFileUrl(shot.video)} controls playsInline preload="metadata" className="absolute inset-0 w-full h-full object-cover" />
                  : <span className="absolute inset-0 flex items-center justify-center text-[10px] text-text-muted">No clip yet</span>}
              </div>
              <figcaption className="px-2 py-1 text-[10px] text-text-muted flex items-center gap-1.5">
                <StatusDot status={shot.videoStatus} /> Clip · {shot.videoStale ? 'stale' : STATUS_LABEL[shot.videoStatus]}
              </figcaption>
            </figure>
          </div>

          <div className="flex items-center gap-1.5">
            <button
              onClick={() => run('tag', () => actions.tag(shot.index, shot.tag === 'good' ? null : 'good'))}
              disabled={!!busy}
              className={`flex items-center gap-1 px-2 py-1 rounded border text-[10px] disabled:opacity-40 ${shot.tag === 'good' ? 'bg-green-500 text-white border-green-500' : 'border-border text-text-secondary hover:text-indicator-success'}`}
            ><Check size={11} /> Good</button>
            <button
              onClick={() => run('tag', () => actions.tag(shot.index, shot.tag === 'needs_work' ? null : 'needs_work'))}
              disabled={!!busy}
              className={`flex items-center gap-1 px-2 py-1 rounded border text-[10px] disabled:opacity-40 ${shot.tag === 'needs_work' ? 'bg-amber-500 text-white border-amber-500' : 'border-border text-text-secondary hover:text-indicator-warning'}`}
            ><AlertTriangle size={11} /> Needs work</button>
          </div>

          {shot.judge && (shot.judge.issues?.length || shot.judge.fix_hint || shot.judge.error) ? (
            <Section title="Storyboard judge">
              <ul className="list-disc pl-4 text-[11px] text-text-secondary space-y-0.5">
                {(shot.judge.issues || []).map((issue, i) => <li key={i}>{issue}</li>)}
              </ul>
              {shot.judge.fix_hint && <p className="text-[11px] text-text-muted">Fix hint: {shot.judge.fix_hint}</p>}
              {shot.judge.error && <p className="text-[11px] text-chip-red">{shot.judge.error}</p>}
            </Section>
          ) : null}

          {shot.dialogue.length > 0 && (
            <Section title="Dialogue">
              <div className="space-y-1">
                {shot.dialogue.map((line, i) => (
                  <p key={i} className="text-[11px] text-text-secondary">
                    {line.speaker && <span className="font-semibold text-text-primary uppercase text-[10px] mr-1">{line.speaker}</span>}
                    “{line.text}”
                  </p>
                ))}
              </div>
            </Section>
          )}

          <Section title="Shot facts">
            <div className="space-y-0.5">
              <Fact label="Action" value={shot.action} />
              <Fact label="Location" value={shot.location} />
              <Fact label="Continuity" value={[shot.group, shot.strategy].filter(Boolean).join(' · ')} />
              <Fact label="On screen" value={shot.subjects.join(', ')} />
              <Fact label="H3 sequence" value={shot.sequence
                ? `${shot.sequence.id}: shot ${shot.sequence.index + 1} of ${shot.sequence.size} (shots ${shot.sequence.members.map(m => m + 1).join(', ')})`
                : null} />
              <Fact label="Start frame" value={shot.startImage} />
              <Fact label="Clip file" value={shot.video} />
            </div>
          </Section>

          {(actions.editBlockedReason || stagedNote) && (
            <p className={`text-[11px] rounded px-2 py-1.5 border ${stagedNote
              ? 'border-accent-blue/40 bg-accent-blue/10 text-accent-blue'
              : 'border-border bg-bg-tertiary text-text-muted'}`}>
              {stagedNote ? 'Edited prompt is staged: it is sent when you click Continue.' : actions.editBlockedReason}
            </p>
          )}

          {actions.needsImage && (
            <Section title="Image prompt" right={
              <button
                onClick={() => run('image', () => actions.rerunImage(shot.index, imageDirty ? imagePrompt : undefined))}
                disabled={rerunDisabled}
                title={actions.rerunBlockedReason || 'Re-generate the start image (with the edited prompt, if any)'}
                className="flex items-center gap-1 px-2 py-0.5 rounded border border-border text-[10px] text-text-secondary hover:text-accent-blue disabled:opacity-40"
              >{busy === 'image' ? <Loader2 size={10} className="animate-spin" /> : <Camera size={10} />} Rerun image</button>
            }>
              <textarea
                value={imagePrompt}
                onChange={e => setImagePrompt(e.target.value)}
                disabled={editDisabled}
                rows={5}
                className="w-full bg-bg-tertiary border border-border rounded px-2 py-1.5 text-[11px] text-text-primary resize-y focus:outline-none focus:border-accent-blue disabled:opacity-70"
              />
            </Section>
          )}

          <Section title={windowed ? `Video prompt (${windowPrompts.length} windows)` : 'Video prompt'} right={
            <button
              onClick={() => run('video', () => actions.rerunVideo(shot.index, videoPromptForRerun))}
              disabled={rerunDisabled || (actions.needsImage && !shot.startImage)}
              title={actions.rerunBlockedReason
                || (actions.needsImage && !shot.startImage ? 'Generate the start image first' : 'Re-generate the clip (with the edited prompt, if any)')}
              className="flex items-center gap-1 px-2 py-0.5 rounded border border-border text-[10px] text-text-secondary hover:text-indicator-success disabled:opacity-40"
            >{busy === 'video' ? <Loader2 size={10} className="animate-spin" /> : <Film size={10} />} Rerun video</button>
          }>
            {windowed ? windowPrompts.map((prompt, i) => (
              <div key={i}>
                <div className="text-[9px] text-text-muted mb-0.5">Window {i + 1}</div>
                <textarea
                  value={prompt}
                  onChange={e => setWindowPrompts(current => current.map((p, j) => (j === i ? e.target.value : p)))}
                  disabled={editDisabled}
                  rows={4}
                  className="w-full bg-bg-tertiary border border-border rounded px-2 py-1.5 text-[11px] text-text-primary resize-y focus:outline-none focus:border-accent-blue disabled:opacity-70"
                />
              </div>
            )) : (
              <textarea
                value={videoPrompt}
                onChange={e => setVideoPrompt(e.target.value)}
                disabled={editDisabled}
                rows={8}
                className="w-full bg-bg-tertiary border border-border rounded px-2 py-1.5 text-[11px] text-text-primary resize-y focus:outline-none focus:border-accent-blue disabled:opacity-70"
              />
            )}
          </Section>
          {shot.sequence && (
            <p className="text-[10px] text-text-muted">
              This shot renders inside {shot.sequence.id}; a video rerun regenerates this shot on its own.
            </p>
          )}
        </div>

        <footer className="px-4 py-3 border-t border-border flex items-center gap-2 shrink-0">
          {error && <span role="alert" className="text-[10px] text-chip-red flex-1 min-w-0 truncate" title={error}>{error}</span>}
          <div className="ml-auto flex items-center gap-2">
            <button
              onClick={() => { setImagePrompt(shot.imagePrompt); setVideoPrompt(shot.videoPrompt); setWindowPrompts(shot.windowPrompts) }}
              disabled={!!busy || (!imageDirty && !videoDirty)}
              className="px-2.5 py-1 rounded border border-border text-[11px] text-text-secondary hover:bg-bg-hover disabled:opacity-40"
            >Revert</button>
            <button
              onClick={save}
              disabled={editDisabled || (!imageDirty && !videoDirty)}
              title={actions.editBlockedReason || (actions.editsAreStaged ? 'Stage these prompts for Continue' : 'Save the prompts to the project')}
              className="flex items-center gap-1 px-2.5 py-1 rounded bg-accent-blue text-white text-[11px] hover:bg-accent-blue-hover disabled:opacity-40"
            >{busy === 'save' ? <Loader2 size={11} className="animate-spin" /> : <Save size={11} />}
              {actions.editsAreStaged ? 'Stage edit' : 'Save prompts'}</button>
          </div>
        </footer>
      </aside>
    </div>
  )
}
