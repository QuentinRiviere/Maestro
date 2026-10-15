import { useEffect } from 'react'
import { useStore } from '../../stores/useStore'
import { formatDirectorHash, parseDirectorHash } from '../../lib/directorRoute'

/** Open the Director runs page (the persistent top-bar link uses this). */
export function openDirectorRuns(): void {
  if (parseDirectorHash(window.location.hash)?.view === 'runs') {
    useStore.getState().setDashboardOpen(true)
    useStore.getState().setDashboardView('runs')
    return
  }
  window.location.hash = '#/director'
}

/**
 * Keep the Director page and the URL hash in sync, both ways.
 *
 * `#/director[/<pid>[/<view>]]` opens the page (deep links, reload, and
 * browser back/forward), and opening, switching run or view, or closing the
 * page pushes the matching hash so those steps are in the browser history.
 */
export function useDirectorHashRoute(): void {
  useEffect(() => {
    let applying = false

    const apply = () => {
      const route = parseDirectorHash(window.location.hash)
      const state = useStore.getState()
      applying = true
      try {
        if (!route) {
          if (state.dashboardOpen) state.setDashboardOpen(false)
          return
        }
        const justOpened = !state.dashboardOpen
        // Opening re-reads the run that was selected before, if any.
        if (justOpened) state.setDashboardOpen(true)
        if (route.pid && (justOpened || state.dashboardSelectedPipeline?.pipeline_id !== route.pid)) {
          // The newest load wins, so this supersedes the re-read above.
          void state.loadSavedPipeline(route.pid)
        }
        if (state.dashboardView !== route.view) state.setDashboardView(route.view)
      } finally {
        applying = false
      }
    }

    const sync = () => {
      if (applying) return
      const state = useStore.getState()
      const current = window.location.hash
      const onDirector = parseDirectorHash(current) !== null
      if (!state.dashboardOpen) {
        if (onDirector) {
          window.history.pushState(null, '', window.location.pathname + window.location.search)
        }
        return
      }
      if (state.dashboardLoading && state.dashboardView !== 'runs') return
      const pid = state.dashboardSelectedPipeline?.pipeline_id || null
      const view = pid ? state.dashboardView : 'runs'
      const wanted = formatDirectorHash({ pid, view })
      if (wanted === current) return
      // Auto-selecting a run after opening is not a navigation step.
      const previous = parseDirectorHash(current)
      if (previous && !previous.pid && state.dashboardView !== 'runs') {
        window.history.replaceState(null, '', wanted)
      } else {
        window.history.pushState(null, '', wanted)
      }
    }

    apply()
    sync()
    window.addEventListener('hashchange', apply)
    window.addEventListener('popstate', apply)
    const unsubscribe = useStore.subscribe((state, previous) => {
      if (
        state.dashboardOpen !== previous.dashboardOpen
        || state.dashboardView !== previous.dashboardView
        || state.dashboardLoading !== previous.dashboardLoading
        || state.dashboardSelectedPipeline?.pipeline_id !== previous.dashboardSelectedPipeline?.pipeline_id
      ) sync()
    })
    return () => {
      window.removeEventListener('hashchange', apply)
      window.removeEventListener('popstate', apply)
      unsubscribe()
    }
  }, [])
}
