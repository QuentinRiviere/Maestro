/**
 * Per-viewer canvas preferences of a Director run (collapsed sections),
 * kept in browser storage. Every access is guarded: storage may be missing,
 * full or blocked, and the canvas must still render with defaults.
 */

type KeyValueStorage = Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>

const PREFIX = 'maestro.director.canvas.collapsed.'

function defaultStorage(): KeyValueStorage | null {
  try {
    return typeof window !== 'undefined' ? window.localStorage : null
  } catch {
    return null
  }
}

export function loadCollapsed(runId: string, storage: KeyValueStorage | null = defaultStorage()): Set<string> {
  if (!storage || !runId) return new Set()
  try {
    const raw = storage.getItem(PREFIX + runId)
    const parsed: unknown = raw ? JSON.parse(raw) : []
    return new Set(Array.isArray(parsed) ? parsed.filter((k): k is string => typeof k === 'string') : [])
  } catch {
    return new Set()
  }
}

export function saveCollapsed(
  runId: string, keys: ReadonlySet<string>, storage: KeyValueStorage | null = defaultStorage(),
): void {
  if (!storage || !runId) return
  try {
    if (keys.size === 0) storage.removeItem(PREFIX + runId)
    else storage.setItem(PREFIX + runId, JSON.stringify([...keys].sort()))
  } catch {
    // Storage full or blocked: the preference just does not persist.
  }
}
