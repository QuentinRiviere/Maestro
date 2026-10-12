import { useState } from 'react'
import { useStore } from '../stores/useStore'

export type LoraStorage = 'local' | 'r2' | 'both'

/** Per-download storage choice, pre-filled with the Settings default. */
export function useLoraStorage(): {
  storage: LoraStorage
  setStorage: (value: LoraStorage) => void
  r2Configured: boolean
} {
  const r2Configured = useStore(s => Boolean(s.servicesConfig?.r2_configured))
  const fallback = useStore(s => s.servicesConfig?.lora_download_storage ?? 'local')
  const [choice, setStorage] = useState<LoraStorage | null>(null)
  const storage = r2Configured ? (choice ?? fallback) : 'local'
  return { storage, setStorage, r2Configured }
}
