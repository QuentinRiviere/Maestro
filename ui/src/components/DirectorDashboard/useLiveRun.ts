import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchPipelineCosts, fetchPipelineStatus, type PipelineCosts } from '../../api/client'
import { useStore } from '../../stores/useStore'
import { isRunActive, type LiveRunStatus } from '../../lib/directorRun'

const LIVE_POLL_MS = 3000
/** Re-read the saved file every Nth live poll (it carries clip videos). */
const SAVED_REFRESH_EVERY = 4

/**
 * Live status of one run, polled every few seconds while the worker runs.
 *
 * The saved run in the store is refreshed on every phase change, every few
 * polls, and once more when the run ends, so the storyboard and the canvas
 * pick up new images and clips without a manual reload. A change of the
 * saved status (e.g. Resume from the header) restarts polling.
 */
export function useLiveRun(pid: string | null, savedStatus?: string): {
  live: LiveRunStatus | null
  refresh: () => Promise<void>
} {
  // Keyed by run so a previous run's status never shows for a new one.
  const [state, setState] = useState<{ pid: string; live: LiveRunStatus | null } | null>(null)
  const refreshSaved = useStore(s => s.refreshSavedPipeline)
  const loadPipelineList = useStore(s => s.loadPipelineList)
  const tokenRef = useRef(0)
  // Bumped by refresh() so polling restarts after an action resumes a run.
  const [epoch, setEpoch] = useState(0)

  const fetchOnce = useCallback(async (target: string, token: number) => {
    try {
      const status = await fetchPipelineStatus(target) as unknown as LiveRunStatus
      if (token !== tokenRef.current) return null
      setState({ pid: target, live: status })
      return status
    } catch {
      if (token === tokenRef.current) setState({ pid: target, live: null })
      return null
    }
  }, [])

  useEffect(() => {
    if (!pid) return
    const token = ++tokenRef.current
    let timer: number | null = null
    let polls = 0
    let lastPhase = ''
    let lastClip = 0
    let wasActive = false

    const tick = async () => {
      timer = null
      const status = await fetchOnce(pid, token)
      if (token !== tokenRef.current) return
      const active = !!status && !status.recovered_from_disk && isRunActive(status.status)
      const phase = String(status?.phase || '')
      // A new clip number means the previous clip just finished: its video
      // is published into the saved run before the batch ends.
      const clip = Number(status?.progress?.current_clip) || 0
      polls += 1
      if (active && (phase !== lastPhase || clip !== lastClip || polls % SAVED_REFRESH_EVERY === 0)) void refreshSaved(pid)
      if (wasActive && !active) {
        void refreshSaved(pid)
        void loadPipelineList()
      }
      lastPhase = phase
      lastClip = clip
      wasActive = active
      if (active) timer = window.setTimeout(tick, LIVE_POLL_MS)
    }
    void tick()
    return () => {
      tokenRef.current += 1
      if (timer != null) window.clearTimeout(timer)
    }
  }, [pid, epoch, savedStatus, fetchOnce, refreshSaved, loadPipelineList])

  const refresh = useCallback(async () => {
    if (!pid) return
    setEpoch(value => value + 1)
    await refreshSaved(pid)
  }, [pid, refreshSaved])

  return { live: state && state.pid === pid ? state.live : null, refresh }
}

/** A run's spend, re-read whenever `version` changes. */
export function useRunCosts(pid: string | null, version: string): PipelineCosts | null {
  const [costs, setCosts] = useState<PipelineCosts | null>(null)
  useEffect(() => {
    if (!pid) return
    let current = true
    fetchPipelineCosts(pid)
      .then(value => { if (current) setCosts(value) })
      .catch(() => { if (current) setCosts(null) })
    return () => { current = false }
  }, [pid, version])
  return pid ? costs : null
}
