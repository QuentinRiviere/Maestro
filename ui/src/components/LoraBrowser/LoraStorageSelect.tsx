import type { LoraStorage } from '../../lib/loraStorage'

const OPTIONS: Array<{ value: LoraStorage; label: string }> = [
  { value: 'local', label: 'This computer' },
  { value: 'r2', label: 'R2 bucket only' },
  { value: 'both', label: 'Both' },
]

const HELP: Record<LoraStorage, string> = {
  local: 'Saved on this computer only.',
  r2: 'Uploaded to your R2 bucket, then removed here. It stays listed; the GPU worker and cloud models read it from R2, and a local run downloads it on demand.',
  both: 'Saved here and copied to your R2 bucket.',
}

export function LoraStorageSelect({ value, onChange }: { value: LoraStorage; onChange: (value: LoraStorage) => void }) {
  return (
    <div>
      <label className="text-[11px] text-text-muted uppercase tracking-wider mb-1 block">Store in</label>
      <select
        value={value}
        onChange={e => onChange(e.target.value as LoraStorage)}
        className="w-full bg-bg-tertiary border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-accent-blue"
      >
        {OPTIONS.map(option => <option key={option.value} value={option.value}>{option.label}</option>)}
      </select>
      <p className="text-[10px] text-text-muted mt-1 leading-snug">{HELP[value]}</p>
    </div>
  )
}

export function StorageBadge({ storage }: { storage?: string | null }) {
  if (storage !== 'r2' && storage !== 'both') return null
  return (
    <span
      title={storage === 'r2' ? 'Stored in your R2 bucket only' : 'Stored here and in your R2 bucket'}
      className="text-[9px] px-1 py-0.5 rounded bg-bg-tertiary text-text-muted leading-none shrink-0"
    >
      {storage === 'r2' ? 'R2' : 'R2 + local'}
    </span>
  )
}
