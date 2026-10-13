import { useCallback, useEffect, useState, type ReactNode } from 'react'
import { AlertTriangle } from 'lucide-react'

interface Pending {
  title: string
  message: ReactNode
  confirmLabel: string
  danger: boolean
  run: () => void | Promise<void>
}

/** An in-app confirmation: ``ask`` opens it, ``dialog`` renders it. Enter confirms, Escape cancels. */
export function useConfirm() {
  const [pending, setPending] = useState<Pending | null>(null)
  const [busy, setBusy] = useState(false)

  const ask = useCallback((options: {
    title: string
    message: ReactNode
    confirmLabel?: string
    danger?: boolean
    run: () => void | Promise<void>
  }) => {
    setPending({ confirmLabel: 'Confirm', danger: false, ...options })
  }, [])

  const confirm = useCallback(async () => {
    if (!pending) return
    setBusy(true)
    try { await pending.run() } finally { setBusy(false); setPending(null) }
  }, [pending])

  useEffect(() => {
    if (!pending) return
    const key = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.stopImmediatePropagation(); setPending(null) }
      if (e.key === 'Enter') { e.preventDefault(); void confirm() }
    }
    window.addEventListener('keydown', key, true)
    return () => window.removeEventListener('keydown', key, true)
  }, [pending, confirm])

  const dialog = pending && (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 p-4" role="alertdialog" aria-modal="true"
      aria-label={pending.title} onClick={e => { if (e.target === e.currentTarget && !busy) setPending(null) }}>
      <div className="w-full max-w-sm space-y-3 rounded-xl border border-border bg-bg-secondary p-4">
        <div className="flex items-center gap-2 text-sm font-medium text-text-primary">
          {pending.danger && <AlertTriangle size={14} className="text-chip-red" />} {pending.title}
        </div>
        <div className="text-[12px] leading-relaxed text-text-secondary">{pending.message}</div>
        <div className="flex justify-end gap-2">
          <button onClick={() => setPending(null)} disabled={busy}
            className="rounded border border-border px-3 py-1 text-[11px] text-text-secondary hover:text-text-primary disabled:opacity-50">
            Cancel
          </button>
          <button onClick={() => void confirm()} disabled={busy} autoFocus
            className={`rounded px-3 py-1 text-[11px] text-white disabled:opacity-50 ${pending.danger ? 'bg-red-600 hover:bg-red-500' : 'bg-accent-blue'}`}>
            {busy ? '…' : pending.confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
  return { ask, dialog }
}
