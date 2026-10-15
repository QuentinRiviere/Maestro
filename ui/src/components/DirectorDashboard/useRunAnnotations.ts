import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  createRunAnnotation, deleteRunAnnotation, fetchRunAnnotations, updateRunAnnotation,
} from '../../api/client'
import type { AnnotationDraft, AnnotationPatch, RunAnnotation } from '../../lib/directorAnnotations'

export interface AnnotationActions {
  create: (draft: AnnotationDraft) => Promise<void>
  update: (id: string, patch: AnnotationPatch) => Promise<void>
  remove: (id: string) => Promise<void>
}

/**
 * Post-analysis notes of one run. Re-read whenever `pollKey` changes (the
 * live status object, so notes follow the live poll cadence) and after every
 * write, so a note added from another tab shows up too.
 */
export function useRunAnnotations(pid: string, pollKey: unknown): {
  annotations: RunAnnotation[]
  error: string | null
  actions: AnnotationActions
} {
  const [state, setState] = useState<{ pid: string; list: RunAnnotation[]; error: string | null }>(
    { pid, list: [], error: null },
  )
  const tokenRef = useRef(0)

  const load = useCallback(async () => {
    const token = ++tokenRef.current
    try {
      const list = await fetchRunAnnotations(pid)
      if (token === tokenRef.current) setState({ pid, list, error: null })
    } catch (e) {
      if (token === tokenRef.current) {
        setState(prev => ({ pid, list: prev.pid === pid ? prev.list : [], error: e instanceof Error ? e.message : String(e) }))
      }
    }
  }, [pid])

  useEffect(() => { void load() }, [load, pollKey])

  const actions = useMemo<AnnotationActions>(() => ({
    create: async draft => { await createRunAnnotation(pid, draft); await load() },
    update: async (id, patch) => { await updateRunAnnotation(pid, id, patch); await load() },
    remove: async id => { await deleteRunAnnotation(pid, id); await load() },
  }), [pid, load])

  const current = state.pid === pid ? state : { list: [], error: null }
  return { annotations: current.list, error: current.error, actions }
}
