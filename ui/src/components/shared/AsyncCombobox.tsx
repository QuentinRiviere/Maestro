import { useEffect, useId, useRef, useState, type CSSProperties, type KeyboardEvent, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { ChevronDown, Loader2, Search } from 'lucide-react'

export interface AsyncComboboxProps<T> {
  /** Accessible name of the control. */
  label: string
  /** Trigger content (the current value). */
  value: ReactNode
  /** Rows loaded so far, already in display order. */
  items: T[]
  getKey: (item: T) => string
  renderOption: (item: T, state: { active: boolean; selected: boolean }) => ReactNode
  selectedKey: string | null
  onSelect: (item: T) => void
  /** Debounced search text; called with '' when the popover closes. */
  onQueryChange: (query: string) => void
  onOpenChange?: (open: boolean) => void
  loading?: boolean
  hasMore?: boolean
  onLoadMore?: () => void
  error?: string | null
  emptyText?: string
  searchPlaceholder?: string
  debounceMs?: number
  disabled?: boolean
  className?: string
  triggerClassName?: string
  popoverClassName?: string
  align?: 'left' | 'right'
}

/**
 * A searchable select whose options come from a paged, server-side source.
 *
 * The trigger opens a popover with a search box (the ARIA combobox) and a
 * listbox. Arrow keys, Home/End and Enter pick an option, Escape closes,
 * and typing on the closed trigger opens it and starts a search. Scrolling
 * near the end of the list, or arrowing onto the last row, asks for the
 * next page.
 */
export function AsyncCombobox<T>({
  label, value, items, getKey, renderOption, selectedKey, onSelect, onQueryChange, onOpenChange,
  loading = false, hasMore = false, onLoadMore, error = null, emptyText = 'Nothing found',
  searchPlaceholder = 'Search…', debounceMs = 250, disabled = false,
  className = '', triggerClassName = '', popoverClassName = '', align = 'left',
}: AsyncComboboxProps<T>) {
  const id = useId()
  const listboxId = `${id}-listbox`
  const [open, setOpen] = useState(false)
  const [input, setInput] = useState('')
  const [activeIndex, setActiveIndex] = useState(0)
  const rootRef = useRef<HTMLDivElement | null>(null)
  const triggerRef = useRef<HTMLButtonElement | null>(null)
  const inputRef = useRef<HTMLInputElement | null>(null)
  const listRef = useRef<HTMLUListElement | null>(null)
  const popoverRef = useRef<HTMLDivElement | null>(null)
  const [position, setPosition] = useState<CSSProperties>({})

  // The popover is portalled with fixed positioning so scrolling or
  // overflow-hidden ancestors (sidebars, headers) cannot clip it.
  const measure = () => {
    const rect = triggerRef.current?.getBoundingClientRect()
    if (!rect) return
    const width = Math.min(448, window.innerWidth - 16)
    const preferred = align === 'right' ? rect.right - width : rect.left
    const left = Math.max(8, Math.min(preferred, window.innerWidth - width - 8))
    const below = window.innerHeight - rect.bottom
    const above = rect.top
    const maxHeight = Math.max(160, Math.min(440, (below >= 260 || below >= above ? below : above) - 12))
    setPosition(below >= 260 || below >= above
      ? { left, width, top: rect.bottom + 4, maxHeight }
      : { left, width, bottom: window.innerHeight - rect.top + 4, maxHeight })
  }

  const active = items.length === 0 ? -1 : Math.min(Math.max(activeIndex, 0), items.length - 1)
  const optionId = (index: number) => `${id}-option-${index}`

  const setOpenState = (next: boolean, { restoreFocus = false }: { restoreFocus?: boolean } = {}) => {
    setOpen(next)
    onOpenChange?.(next)
    if (!next) {
      setInput('')
      onQueryChange('')
      if (restoreFocus) triggerRef.current?.focus()
    }
  }

  const openWith = (text: string) => {
    setInput(text)
    if (!text) {
      const selected = items.findIndex(item => getKey(item) === selectedKey)
      setActiveIndex(selected >= 0 ? selected : 0)
    } else {
      setActiveIndex(0)
    }
    measure()
    setOpenState(true)
  }

  // Debounce the search text into the data source.
  useEffect(() => {
    if (!open) return
    const timer = window.setTimeout(() => onQueryChange(input), input ? debounceMs : 0)
    return () => window.clearTimeout(timer)
  }, [input, open, debounceMs, onQueryChange])

  useEffect(() => {
    if (open) inputRef.current?.focus()
  }, [open])

  // Close on a click outside.
  useEffect(() => {
    if (!open) return
    const onPointerDown = (event: PointerEvent) => {
      const target = event.target as Node
      if (rootRef.current?.contains(target) || popoverRef.current?.contains(target)) return
      setOpen(false)
      onOpenChange?.(false)
      setInput('')
      onQueryChange('')
    }
    document.addEventListener('pointerdown', onPointerDown)
    return () => document.removeEventListener('pointerdown', onPointerDown)
  }, [open, onOpenChange, onQueryChange])

  // Follow the trigger while the page scrolls or resizes.
  useEffect(() => {
    if (!open) return
    const onMove = (event: Event) => {
      if (popoverRef.current && event.target instanceof Node && popoverRef.current.contains(event.target)) return
      measure()
    }
    window.addEventListener('scroll', onMove, true)
    window.addEventListener('resize', onMove)
    return () => {
      window.removeEventListener('scroll', onMove, true)
      window.removeEventListener('resize', onMove)
    }
  })

  // Keep the active option visible.
  useEffect(() => {
    if (!open || active < 0) return
    document.getElementById(`${id}-option-${active}`)?.scrollIntoView({ block: 'nearest' })
  }, [open, active, id])

  const maybeLoadMore = () => {
    if (hasMore && !loading) onLoadMore?.()
  }

  const move = (index: number) => {
    if (items.length === 0) return
    const next = Math.min(Math.max(index, 0), items.length - 1)
    setActiveIndex(next)
    if (next >= items.length - 3) maybeLoadMore()
  }

  const choose = (item: T) => {
    onSelect(item)
    setOpenState(false, { restoreFocus: true })
  }

  const onInputKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    switch (event.key) {
      case 'ArrowDown': event.preventDefault(); move(active + 1); break
      case 'ArrowUp': event.preventDefault(); move(active - 1); break
      case 'PageDown': event.preventDefault(); move(active + 8); break
      case 'PageUp': event.preventDefault(); move(active - 8); break
      case 'Home': if (event.ctrlKey || !input) { event.preventDefault(); move(0) } break
      case 'End': if (event.ctrlKey || !input) { event.preventDefault(); move(items.length - 1) } break
      case 'Enter':
        event.preventDefault()
        if (active >= 0 && items[active]) choose(items[active])
        break
      case 'Escape':
        event.preventDefault()
        event.stopPropagation()
        setOpenState(false, { restoreFocus: true })
        break
      case 'Tab':
        setOpenState(false)
        break
    }
  }

  const onTriggerKeyDown = (event: KeyboardEvent<HTMLButtonElement>) => {
    if (disabled) return
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault()
      openWith('')
    } else if (event.key.length === 1 && event.key !== ' ' && !event.ctrlKey && !event.metaKey && !event.altKey) {
      // Type to search: the first character opens the popover with it.
      event.preventDefault()
      openWith(event.key)
    }
  }

  const onListScroll = () => {
    const el = listRef.current
    if (el && el.scrollTop + el.clientHeight >= el.scrollHeight - 48) maybeLoadMore()
  }

  return (
    <div ref={rootRef} className={`relative ${className}`}>
      <button
        ref={triggerRef}
        type="button"
        disabled={disabled}
        aria-label={label}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={open ? listboxId : undefined}
        onClick={() => (open ? setOpenState(false) : openWith(''))}
        onKeyDown={onTriggerKeyDown}
        className={`w-full flex items-center gap-2 text-left bg-bg-tertiary border border-border rounded-lg px-2 py-1 text-xs text-text-primary hover:border-border-light focus:outline-none focus-visible:border-accent-blue disabled:opacity-50 ${open ? 'border-accent-blue' : ''} ${triggerClassName}`}
      >
        <span className="flex-1 min-w-0">{value}</span>
        <ChevronDown size={12} className={`shrink-0 text-text-muted transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>

      {open && createPortal(
        <div
          ref={popoverRef}
          style={position}
          className={`fixed z-[70] rounded-lg border border-border bg-bg-secondary shadow-xl flex flex-col overflow-hidden ${popoverClassName}`}
        >
          <div className="relative border-b border-border p-1.5">
            <Search size={12} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-text-muted pointer-events-none" />
            <input
              ref={inputRef}
              role="combobox"
              aria-label={`${label}: search`}
              aria-expanded
              aria-controls={listboxId}
              aria-autocomplete="list"
              aria-activedescendant={active >= 0 ? optionId(active) : undefined}
              value={input}
              placeholder={searchPlaceholder}
              onChange={event => { setInput(event.target.value); setActiveIndex(0) }}
              onKeyDown={onInputKeyDown}
              className="w-full rounded-md border border-border bg-bg-tertiary py-1.5 pl-7 pr-7 text-xs text-text-primary outline-none placeholder:text-text-muted focus:border-accent-blue/60"
            />
            {loading && <Loader2 size={12} className="absolute right-3.5 top-1/2 -translate-y-1/2 animate-spin text-text-muted" />}
          </div>
          <ul
            ref={listRef}
            id={listboxId}
            role="listbox"
            aria-label={label}
            aria-busy={loading}
            onScroll={onListScroll}
            className="min-h-0 flex-1 overflow-y-auto py-1"
          >
            {items.map((item, index) => {
              const key = getKey(item)
              const selected = key === selectedKey
              return (
                <li
                  key={key}
                  id={optionId(index)}
                  role="option"
                  aria-selected={selected}
                  onPointerMove={() => { if (index !== active) setActiveIndex(index) }}
                  onMouseDown={event => event.preventDefault()}
                  onClick={() => choose(item)}
                  className={`cursor-pointer px-2 py-1.5 ${index === active ? 'bg-bg-hover' : ''} ${selected ? 'text-accent-blue' : ''}`}
                >
                  {renderOption(item, { active: index === active, selected })}
                </li>
              )
            })}
            {items.length === 0 && !loading && (
              <li role="presentation" className="px-3 py-4 text-center text-[11px] text-text-muted">
                {error ? `Could not load: ${error}` : emptyText}
              </li>
            )}
            {items.length === 0 && loading && (
              <li role="presentation" className="px-3 py-4 flex items-center justify-center gap-2 text-[11px] text-text-muted">
                <Loader2 size={12} className="animate-spin" /> Loading…
              </li>
            )}
            {items.length > 0 && hasMore && (
              <li role="presentation" className="px-2 pt-1">
                <button
                  type="button"
                  tabIndex={-1}
                  disabled={loading}
                  onMouseDown={event => event.preventDefault()}
                  onClick={maybeLoadMore}
                  className="w-full flex items-center justify-center gap-1.5 rounded-md py-1 text-[10px] text-text-muted hover:bg-bg-hover hover:text-text-secondary disabled:opacity-60"
                >
                  {loading ? <><Loader2 size={10} className="animate-spin" /> Loading…</> : error ? `Retry (${error})` : 'Load more'}
                </button>
              </li>
            )}
          </ul>
        </div>,
        document.body,
      )}
    </div>
  )
}
