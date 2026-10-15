import { Layers } from 'lucide-react'
import { formatSeconds, type RunShot, type SceneGroup } from '../../lib/directorRun'
import { JudgeBadge, ShotMedia, StatusDot } from './runUi'
import { STATUS_LABEL } from './runStyles'

function ShotPanel({ shot, aspect, staged, onOpen }: {
  shot: RunShot
  aspect: string
  staged: boolean
  onOpen: () => void
}) {
  const tagRing = shot.tag === 'good'
    ? 'border-green-500/70'
    : shot.tag === 'needs_work' ? 'border-amber-500/70' : 'border-border hover:border-border-light'
  const kind = [shot.sceneType, shot.framing].filter(Boolean).join(' · ')
  return (
    <button
      type="button"
      onClick={onOpen}
      className={`group text-left rounded-lg border ${tagRing} bg-bg-secondary overflow-hidden flex flex-col transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-blue`}
    >
      <div className="relative">
        <ShotMedia image={shot.startImage} video={shot.video} keep={shot.keep} alt={`Shot ${shot.index + 1}`} aspect={aspect} />
        <span className="absolute top-1.5 left-1.5 px-1.5 py-0.5 rounded bg-black/65 text-white text-[10px] font-semibold tabular-nums">
          {shot.index + 1}
        </span>
        <JudgeBadge judge={shot.judge} className="absolute top-1.5 right-1.5 backdrop-blur-sm" />
        <div className="absolute bottom-1.5 left-1.5 flex items-center gap-1 px-1.5 py-0.5 rounded bg-black/65 text-[9px] text-white/90">
          <StatusDot status={shot.imageStatus} title={`Start frame: ${STATUS_LABEL[shot.imageStatus]}`} />
          <StatusDot status={shot.videoStatus} title={`Clip: ${shot.videoStale ? 'stale' : STATUS_LABEL[shot.videoStatus]}${shot.videoProvisional ? ' (preview until the batch ends)' : ''}`} />
          {shot.sequence && <span className="ml-0.5 flex items-center gap-0.5" title={`Rendered in ${shot.sequence.id}`}><Layers size={9} />{shot.sequence.index + 1}/{shot.sequence.size}</span>}
        </div>
        {shot.durationSec != null && (
          <span className="absolute bottom-1.5 right-1.5 px-1.5 py-0.5 rounded bg-black/65 text-white text-[9px] tabular-nums">
            {formatSeconds(shot.durationSec)}
          </span>
        )}
      </div>
      <div className="p-2 space-y-1 flex-1">
        <div className="flex items-center gap-1 min-w-0">
          <span className="text-[10px] font-medium text-text-primary truncate">{kind || `Shot ${shot.index + 1}`}</span>
          {staged && <span className="ml-auto text-[8px] px-1 rounded bg-accent-blue/15 text-accent-blue shrink-0">edited</span>}
        </div>
        {shot.dialogue.slice(0, 2).map((line, i) => (
          <p key={i} className="text-[10px] text-text-secondary line-clamp-2">
            {line.speaker && <span className="font-semibold uppercase text-[9px] text-text-primary mr-1">{line.speaker}</span>}
            “{line.text}”
          </p>
        ))}
        {shot.dialogue.length > 2 && <p className="text-[9px] text-text-muted">+{shot.dialogue.length - 2} more line(s)</p>}
        {shot.action && <p className="text-[10px] text-text-muted line-clamp-2 italic">{shot.action}</p>}
      </div>
    </button>
  )
}

/** The run's shots as storyboard panels, grouped by scene. */
export function StoryboardView({ scenes, aspect, stagedShots, onOpenShot, emptyHint }: {
  scenes: SceneGroup[]
  aspect: string
  stagedShots: Set<number>
  onOpenShot: (index: number) => void
  emptyHint: string
}) {
  const [w, h] = aspect.split('/').map(part => Number(part.trim()))
  const gridCols = w > h
    ? 'grid-cols-[repeat(auto-fill,minmax(240px,1fr))]'
    : 'grid-cols-[repeat(auto-fill,minmax(148px,1fr))]'
  if (scenes.length === 0) {
    return <div className="py-16 text-center text-xs text-text-muted">{emptyHint}</div>
  }
  return (
    <div className="space-y-6">
      {scenes.map(scene => {
        const total = scene.shots.reduce((sum, shot) => sum + (shot.durationSec || 0), 0)
        const done = scene.shots.filter(shot => shot.videoStatus === 'done').length
        return (
          <section key={scene.key} aria-label={scene.title}>
            <header className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5 mb-2 pb-1 border-b border-border">
              <h3 className="text-xs font-semibold text-text-primary">{scene.title}</h3>
              {scene.group && scene.location && (
                <span className="text-[10px] text-text-muted font-mono">{scene.group}</span>
              )}
              <span className="text-[10px] text-text-muted">
                {scene.shots.length} shot{scene.shots.length === 1 ? '' : 's'}
                {total > 0 && <> · {formatSeconds(total)}</>}
                {' '}· {done}/{scene.shots.length} rendered
              </span>
            </header>
            <div className={`grid gap-3 ${gridCols}`}>
              {scene.shots.map(shot => (
                <ShotPanel
                  key={shot.index}
                  shot={shot}
                  aspect={aspect}
                  staged={stagedShots.has(shot.index)}
                  onOpen={() => onOpenShot(shot.index)}
                />
              ))}
            </div>
          </section>
        )
      })}
    </div>
  )
}
