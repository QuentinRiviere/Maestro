/**
 * Hash routes of the Director runs page. The app has no router, so the page
 * lives in the URL hash and browser back/forward walk through it:
 *
 *   #/director                      runs list
 *   #/director/<pid>                one run, storyboard view
 *   #/director/<pid>/<view>         one run, given view
 */
export type DirectorView = 'runs' | 'storyboard' | 'canvas' | 'details'

export const DIRECTOR_RUN_VIEWS: Exclude<DirectorView, 'runs'>[] = ['storyboard', 'canvas', 'details']

export interface DirectorRoute {
  pid: string | null
  view: DirectorView
}

/** The Director route in a location hash, or null when the hash is elsewhere. */
export function parseDirectorHash(hash: string): DirectorRoute | null {
  const path = hash.replace(/^#/, '').replace(/^\/+/, '').split('?')[0]
  const parts = path.split('/').filter(Boolean)
  if (parts[0] !== 'director') return null
  if (parts.length === 1) return { pid: null, view: 'runs' }
  let pid: string
  try {
    pid = decodeURIComponent(parts[1])
  } catch {
    return { pid: null, view: 'runs' }
  }
  const view = (DIRECTOR_RUN_VIEWS as string[]).includes(parts[2] || '')
    ? parts[2] as DirectorView
    : 'storyboard'
  return { pid, view }
}

export function formatDirectorHash(route: DirectorRoute): string {
  if (!route.pid || route.view === 'runs') return '#/director'
  const base = `#/director/${encodeURIComponent(route.pid)}`
  return route.view === 'storyboard' ? base : `${base}/${route.view}`
}
