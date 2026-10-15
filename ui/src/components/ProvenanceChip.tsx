import { useState } from 'react'
import { Cpu } from 'lucide-react'
import {
  hasProvenance, provenanceRows, provenanceSummary, provenanceTitle, type GenerationProvenance,
} from '../lib/provenance'

/** Every known provenance fact as a label/value grid (see lib/provenance). */
export function ProvenanceDetails({ provenance, className = '' }: {
  provenance: GenerationProvenance | null | undefined
  className?: string
}) {
  const rows = provenanceRows(provenance)
  if (!rows.length) return null
  return (
    <dl className={`grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1 text-[11px] ${className}`}>
      {rows.map(row => (
        <div key={row.label} className="contents">
          <dt className="text-text-muted">{row.label}</dt>
          <dd className="min-w-0 break-words text-text-secondary">{row.value}</dd>
        </div>
      ))}
    </dl>
  )
}

/**
 * One-line "model · quantization · acceleration · GPU · time · cost" chip.
 * Hover shows every fact; clicking expands them inline when `expandable`.
 * Renders nothing when the output has no provenance (older generations).
 */
export function ProvenanceChip({ provenance, label, expandable = true, className = '' }: {
  provenance: GenerationProvenance | null | undefined
  /** Prefix such as "Image" or "Clip" when a card shows several records. */
  label?: string
  expandable?: boolean
  className?: string
}) {
  const [open, setOpen] = useState(false)
  if (!hasProvenance(provenance)) return null
  const summary = provenanceSummary(provenance)
  const title = provenanceTitle(provenance)
  const chipClass = 'inline-flex max-w-full items-center gap-1 rounded border border-border bg-bg-tertiary px-1.5 py-0.5 text-[10px] text-text-muted'
  const content = (
    <>
      <Cpu size={10} className="shrink-0" aria-hidden="true" />
      {label && <span className="shrink-0 font-medium text-text-secondary">{label}</span>}
      <span className="truncate">{summary}</span>
    </>
  )
  if (!expandable) {
    return <span className={`${chipClass} ${className}`} title={title}>{content}</span>
  }
  return (
    <div className={`min-w-0 ${className}`}>
      <button
        type="button"
        className={`${chipClass} hover:bg-bg-hover hover:text-text-secondary`}
        title={title}
        aria-expanded={open}
        aria-label={`${label ? `${label} ` : ''}generation details: ${summary}`}
        onClick={event => { event.stopPropagation(); setOpen(value => !value) }}
      >
        {content}
      </button>
      {open && (
        <div
          className="mt-1 rounded border border-border bg-bg-secondary p-2"
          onClick={event => event.stopPropagation()}
        >
          <ProvenanceDetails provenance={provenance} />
        </div>
      )}
    </div>
  )
}
