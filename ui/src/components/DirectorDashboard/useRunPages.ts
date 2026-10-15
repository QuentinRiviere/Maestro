import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchPipelinePage } from '../../api/client'
import type { PipelineListItem } from '../../types'
import { mergeRunPages } from '../../lib/directorRunList'

interface PagesState {
  /** Query the rows below belong to. */
  key: string
  items: PipelineListItem[]
  nextCursor: string | null
  total: number | null
  error: string | null
  /** A page request for `key` is in flight. */
  loadingMore: boolean
}

const EMPTY: PagesState = { key: '', items: [], nextCursor: null, total: null, error: null, loadingMore: false }

/**
 * Saved Director runs, fetched page by page from the server, newest first.
 *
 * Changing `query` or `status` starts again from the first page; `loadMore`
 * appends the next one. While a new query's first page loads, the previous
 * rows stay visible and `loading` is true.
 */
export function useRunPages({ query = '', status = '', pageSize = 20, enabled = true }: {
  query?: string
  status?: string
  pageSize?: number
  enabled?: boolean
}) {
  const key = `${status}\u0000${query.trim()}`
  const [state, setState] = useState<PagesState>(EMPTY)
  const [reloadToken, setReloadToken] = useState(0)
  const requestRef = useRef<AbortController | null>(null)

  useEffect(() => {
    if (!enabled) return
    const controller = new AbortController()
    requestRef.current?.abort()
    requestRef.current = controller
    fetchPipelinePage({ limit: pageSize, q: query, status }, controller.signal)
      .then(page => {
        if (controller.signal.aborted) return
        setState({
          key,
          items: mergeRunPages([], page.pipelines),
          nextCursor: page.next_cursor || null,
          total: page.total ?? null,
          error: null,
          loadingMore: false,
        })
      })
      .catch(error => {
        if (controller.signal.aborted) return
        setState(current => ({
          ...current,
          key,
          items: current.key === key ? current.items : [],
          error: error instanceof Error ? error.message : String(error),
          loadingMore: false,
        }))
      })
    return () => controller.abort()
  }, [enabled, key, query, status, pageSize, reloadToken])

  const loadMore = useCallback(() => {
    if (state.key !== key || !state.nextCursor || state.loadingMore) return
    const controller = new AbortController()
    requestRef.current = controller
    const cursor = state.nextCursor
    setState(current => ({ ...current, loadingMore: true }))
    fetchPipelinePage({ limit: pageSize, q: query, status, cursor }, controller.signal)
      .then(page => {
        if (controller.signal.aborted) return
        setState(current => current.key !== key ? current : {
          ...current,
          items: mergeRunPages(current.items, page.pipelines),
          nextCursor: page.next_cursor || null,
          total: page.total ?? current.total,
          error: null,
          loadingMore: false,
        })
      })
      .catch(error => {
        if (controller.signal.aborted) return
        setState(current => current.key !== key ? current : {
          ...current,
          error: error instanceof Error ? error.message : String(error),
          loadingMore: false,
        })
      })
  }, [key, pageSize, query, state.key, state.nextCursor, state.loadingMore, status])

  /** Re-read from the first page (for example after a run finished). */
  const reload = useCallback(() => setReloadToken(token => token + 1), [])

  /** Fold fresher copies of runs (from another list) into the loaded rows. */
  const absorb = useCallback((runs: PipelineListItem[]) => {
    setState(current => {
      if (current.key !== key || runs.length === 0) return current
      const loaded = new Set(current.items.map(run => run.id))
      // Only update rows already shown; new runs arrive through reload so a
      // search or status filter is never bypassed.
      const updates = runs.filter(run => loaded.has(run.id))
      if (updates.length === 0) return current
      return { ...current, items: mergeRunPages(current.items, updates) }
    })
  }, [key])

  const stale = state.key !== key
  return {
    items: state.items,
    loading: enabled && (stale || state.loadingMore),
    initialLoading: enabled && stale,
    error: stale ? null : state.error,
    hasMore: !stale && !!state.nextCursor,
    total: stale ? null : state.total,
    loadMore,
    reload,
    absorb,
  }
}
