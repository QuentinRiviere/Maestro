import { Clapperboard } from 'lucide-react'
import { useStore } from '../stores/useStore'
import { openDirectorRuns } from './DirectorDashboard/useDirectorHashRoute'
import type { AppMode } from '../types'

const APP_MODES: Array<{ mode: AppMode; label: string }> = [
  { mode: 'director', label: 'Director' },
  { mode: 'studio', label: 'Studio' },
  { mode: 'editor', label: 'Editor' },
  { mode: 'bibles', label: 'Bibles' },
]

export function MaestroBrand({
  compact = false,
  className = '',
}: {
  compact?: boolean
  className?: string
}) {
  const appVersion = useStore(state => state.systemConfig?.app_version)

  return (
    <div className={`flex shrink-0 items-center gap-2 ${className}`}>
      <img
        src="/maestro-home-icon-orange.png"
        alt=""
        className={`${compact ? 'h-7 w-7 rounded-[7px]' : 'h-8 w-8 rounded-lg'} shrink-0`}
      />
      <>
        {!compact && (
          <span className="text-sm font-semibold tracking-tight text-text-primary">Maestro</span>
        )}
        {appVersion && (
          <span className={`${compact ? 'text-[9px]' : 'mt-0.5 text-[10px]'} whitespace-nowrap font-normal text-text-muted`}>v{appVersion}</span>
        )}
      </>
    </div>
  )
}

export function AppModeToggle({ size = 'md' }: { size?: 'sm' | 'md' }) {
  const activeMode = useStore(state => state.sidebarMode)
  const setSidebarMode = useStore(state => state.setSidebarMode)

  return (
    <div
      className="flex shrink-0 rounded-lg border border-border bg-bg-tertiary p-0.5"
      role="group"
      aria-label="Application mode"
    >
      {APP_MODES.map(({ mode, label }) => (
        <button
          key={mode}
          type="button"
          onClick={() => setSidebarMode(mode)}
          className={`${size === 'sm' ? 'px-2 py-1 text-[11px]' : 'px-3 py-1 text-xs'} rounded-md transition-all ${
            activeMode === mode
              ? 'bg-toggle-active text-white shadow-accent-glow'
              : 'text-text-secondary hover:bg-bg-hover hover:text-text-primary'
          }`}
        >
          {label}
        </button>
      ))}
    </div>
  )
}

/** Persistent link to the Director runs page (#/director). */
export function DirectorRunsButton({ iconSize = 16, className = '' }: { iconSize?: number; className?: string }) {
  // A dot tells that a run is live without opening the page.
  const live = useStore(state => state.dashboardPipelineList.some(run => run.status === 'running' || run.status === 'paused'))
  return (
    <button
      type="button"
      onClick={openDirectorRuns}
      className={`relative p-1.5 rounded-lg hover:bg-bg-hover text-text-secondary hover:text-text-primary transition-colors ${className}`}
      title="Director runs: storyboards and pipeline canvas"
      aria-label="Open Director runs"
    >
      <Clapperboard size={iconSize} />
      {live && <span className="absolute top-1 right-1 w-1.5 h-1.5 rounded-full bg-accent-blue animate-pulse" />}
    </button>
  )
}
