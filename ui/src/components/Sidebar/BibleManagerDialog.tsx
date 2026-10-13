import { useCallback, useEffect, useRef, useState } from 'react'
import { AlertTriangle, Check, ImagePlus, Loader2, Plus, Sparkles, Trash2, X } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import {
  approveBibleImage, BIBLE_PLURAL, createDirectorBible, deleteBibleImage, deleteDirectorBible,
  directorBibleImageUrl, fetchDirectorBible, fetchDirectorBibles, renderBibleImage, updateDirectorBible,
  uploadBibleImage,
  type BibleElement, type BibleKind, type BibleVariant, type DirectorBible, type DirectorBibleSummary,
} from '../../api/directorBibles'

const input = 'w-full bg-bg-tertiary border border-border rounded px-2 py-1 text-[11px] text-text-primary focus:outline-none focus:border-accent-blue'
const label = 'text-[9px] text-text-muted uppercase tracking-wider block mb-0.5'
const TABS: Array<[BibleKind | 'world', string]> = [
  ['character', 'Characters'], ['location', 'Locations'], ['prop', 'Props'], ['world', 'World & style'],
]
const FIELDS: Record<BibleKind, Array<[keyof BibleElement, string, boolean]>> = {
  character: [['role', 'Role', false], ['descriptor', 'How shots name them', false],
    ['description', 'Look', true], ['wardrobe', 'Default wardrobe', false]],
  location: [['description', 'Place', true], ['notes', 'Geometry & physics (doors, scale, light)', false]],
  prop: [['descriptor', 'How shots name it', false], ['description', 'Object', true],
    ['notes', 'Scale & details', false]],
}

const slug = (text: string) => text.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '').slice(0, 32)
const rendering = (bible: DirectorBible | null) => !!bible && (['characters', 'locations', 'props'] as const)
  .some(plural => bible[plural].some(e => e.variants.some(v => v.images.some(i => i.status === 'rendering'))))

function Images({ bible, element, variant, onBible }: {
  bible: DirectorBible
  element: BibleElement
  variant: BibleVariant
  onBible: (bible: DirectorBible) => void
}) {
  const [error, setError] = useState<string | null>(null)
  const act = async (call: () => Promise<DirectorBible>) => {
    setError(null)
    try { onBible(await call()) } catch (e) { setError(e instanceof Error ? e.message : 'Failed') }
  }
  if (!variant.images.length) return <p className="text-[10px] text-text-muted">No image yet: render or upload one.</p>
  return (
    <div className="space-y-1">
      <div className="flex flex-wrap gap-1.5">
        {variant.images.map(image => (
          <div key={image.id} className={`relative w-24 aspect-video rounded overflow-hidden border ${
            image.status === 'approved' ? 'border-emerald-400' : 'border-border'} bg-bg-tertiary`}>
            {image.file
              ? <button className="w-full h-full" title={image.status === 'approved' ? 'Approved reference' : 'Approve this image'}
                  onClick={() => image.status !== 'approved' && void act(() => approveBibleImage(bible.id, element.id, variant.id, image.id))}>
                  <img src={directorBibleImageUrl(bible.id, image.file)} alt={`${element.name} ${variant.label}`} className="w-full h-full object-cover" />
                </button>
              : <div className="w-full h-full flex items-center justify-center" title={image.error || image.status}>
                  {image.status === 'rendering'
                    ? <Loader2 size={14} className="animate-spin text-text-muted" />
                    : <AlertTriangle size={14} className="text-chip-red" />}
                </div>}
            {image.status === 'approved' && <Check size={12} className="absolute left-1 top-1 rounded bg-emerald-500 text-white" />}
            {image.status !== 'rendering' && (
              <button aria-label="Delete image" onClick={() => void act(() => deleteBibleImage(bible.id, element.id, variant.id, image.id))}
                className="absolute right-0.5 top-0.5 rounded bg-black/60 p-0.5 text-white/80 hover:text-white">
                <X size={10} />
              </button>
            )}
          </div>
        ))}
      </div>
      {variant.images.some(i => i.status === 'failed') && (
        <p className="text-[9px] text-chip-red truncate" title={variant.images.find(i => i.status === 'failed')?.error}>
          {variant.images.find(i => i.status === 'failed')?.error}
        </p>
      )}
      {error && <p className="text-[10px] text-chip-red">{error}</p>}
    </div>
  )
}

function ElementCard({ bible, element, saved, onChange, onRemove, onBible }: {
  bible: DirectorBible
  element: BibleElement
  saved: boolean
  onChange: (element: BibleElement) => void
  onRemove: () => void
  onBible: (bible: DirectorBible) => void
}) {
  const imageModel = useStore(s => s.selectedModelPerMode.image || 'flux2_klein_9b')
  const [variantId, setVariantId] = useState('default')
  const [newLook, setNewLook] = useState<{ label: string; description: string } | null>(null)
  const [chain, setChain] = useState<string[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const upload = useRef<HTMLInputElement>(null)
  const variant = element.variants.find(v => v.id === variantId) ?? element.variants[0]
  const others = (element.kind === 'character' ? [] : [...bible.locations, ...bible.props])
    .filter(e => e.id !== element.id && e.image)

  const act = async (call: () => Promise<DirectorBible>) => {
    setBusy(true)
    setError(null)
    try { onBible(await call()) } catch (e) { setError(e instanceof Error ? e.message : 'Failed') } finally { setBusy(false) }
  }
  const addLook = () => {
    if (!newLook?.label.trim()) return
    const id = slug(newLook.label) || `look_${element.variants.length}`
    onChange({ ...element, variants: [...element.variants, { id, label: newLook.label.trim(), description: newLook.description.trim(), images: [] }] })
    setVariantId(id)
    setNewLook(null)
  }

  return (
    <div className="rounded-lg border border-border bg-bg-tertiary/30 p-2.5 space-y-2">
      <div className="flex items-center gap-2">
        <input className={`${input} font-medium`} value={element.name} aria-label="Name"
          onChange={e => onChange({ ...element, name: e.target.value })} />
        <span className="shrink-0 text-[10px] text-text-muted">{element.tag}</span>
        <button onClick={onRemove} aria-label="Remove element" className="shrink-0 text-text-muted hover:text-chip-red"><Trash2 size={12} /></button>
      </div>
      <div className="grid grid-cols-2 gap-1.5">
        {FIELDS[element.kind].map(([field, title, wide]) => (
          <label key={field} className={wide ? 'col-span-2' : ''}>
            <span className={label}>{title}</span>
            {wide
              ? <textarea className={input} rows={3} value={String(element[field] ?? '')}
                  onChange={e => onChange({ ...element, [field]: e.target.value })} />
              : <input className={input} value={String(element[field] ?? '')}
                  onChange={e => onChange({ ...element, [field]: e.target.value })} />}
          </label>
        ))}
      </div>

      <div className="space-y-1.5 border-t border-border/60 pt-2">
        <div className="flex flex-wrap items-center gap-1">
          <span className={label}>Looks</span>
          {element.variants.map(v => (
            <button key={v.id} onClick={() => setVariantId(v.id)}
              className={`rounded px-1.5 py-0.5 text-[10px] border ${v.id === variant?.id
                ? 'border-accent-blue text-text-primary' : 'border-border text-text-secondary hover:text-text-primary'}`}>
              {v.label}{v.images.some(i => i.status === 'approved') ? ' ✓' : ''}
            </button>
          ))}
          <button onClick={() => setNewLook({ label: '', description: '' })} className="text-[10px] text-accent-blue hover:underline">+ look</button>
        </div>
        {newLook && (
          <div className="flex gap-1">
            <input className={`${input} w-28`} placeholder={element.kind === 'character' ? 'Winter' : 'Window open'} value={newLook.label}
              onChange={e => setNewLook({ ...newLook, label: e.target.value })} aria-label="Look name" />
            <input className={input} placeholder={element.kind === 'character' ? 'long camel coat over the uniform' : 'the window is open, curtain moving'}
              value={newLook.description} onChange={e => setNewLook({ ...newLook, description: e.target.value })} aria-label="Look change" />
            <button onClick={addLook} className="px-2 rounded border border-border text-[10px] text-text-secondary hover:text-text-primary">Add</button>
          </div>
        )}
        {variant && variant.id !== 'default' && (
          <input className={input} value={variant.description} placeholder="What changes from the default look"
            aria-label="Look change"
            onChange={e => onChange({ ...element, variants: element.variants.map(v => v.id === variant.id ? { ...v, description: e.target.value } : v) })} />
        )}
        {variant && saved && <Images bible={bible} element={element} variant={variant} onBible={onBible} />}

        {saved && variant && element.variants.some(v => v.id === variant.id) ? (
          <div className="space-y-1">
            {others.length > 0 && (
              <div className="flex flex-wrap items-center gap-1">
                <span className="text-[9px] text-text-muted">Match:</span>
                {others.map(o => (
                  <label key={o.id} className="flex items-center gap-0.5 text-[9px] text-text-secondary">
                    <input type="checkbox" className="w-2.5 h-2.5" checked={chain.includes(o.id)}
                      onChange={e => setChain(e.target.checked ? [...chain, o.id] : chain.filter(id => id !== o.id))} />
                    {o.name}
                  </label>
                ))}
              </div>
            )}
            <div className="flex gap-1.5">
              <button disabled={busy} onClick={() => void act(() => renderBibleImage(bible.id, element.id, variant.id, { image_model: imageModel, chain_from: chain }))}
                title={`Render with ${imageModel}`}
                className="flex-1 py-1 rounded border border-border text-[10px] text-text-secondary hover:text-text-primary hover:bg-bg-hover flex items-center justify-center gap-1 disabled:opacity-50">
                {busy ? <Loader2 size={10} className="animate-spin" /> : <Sparkles size={10} />} Render
              </button>
              <button disabled={busy} onClick={() => upload.current?.click()}
                className="flex-1 py-1 rounded border border-border text-[10px] text-text-secondary hover:text-text-primary hover:bg-bg-hover flex items-center justify-center gap-1 disabled:opacity-50">
                <ImagePlus size={10} /> Upload
              </button>
              <input ref={upload} type="file" accept="image/png,image/jpeg,image/webp" className="hidden"
                onChange={e => {
                  const file = e.target.files?.[0]
                  e.target.value = ''
                  if (file) void act(() => uploadBibleImage(bible.id, element.id, variant.id, file))
                }} />
            </div>
          </div>
        ) : (
          <p className="text-[9px] text-text-muted">Save the bible to render or upload images for this look.</p>
        )}
        {error && <p className="text-[10px] text-chip-red">{error}</p>}
      </div>
    </div>
  )
}

/** The bibles of your series: approved people, places, props and their looks, reused by every episode. */
export function BibleManagerDialog({ initialId, onClose }: { initialId?: string; onClose: () => void }) {
  const [list, setList] = useState<DirectorBibleSummary[]>([])
  const [selected, setSelected] = useState(initialId || '')
  const [bible, setBible] = useState<DirectorBible | null>(null)
  const [draft, setDraft] = useState<DirectorBible | null>(null)
  const [tab, setTab] = useState<BibleKind | 'world'>('character')
  const [newName, setNewName] = useState('')
  const [saving, setSaving] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const dirty = !!draft && !!bible && JSON.stringify(draft) !== JSON.stringify(bible)

  const refreshList = useCallback(() => {
    fetchDirectorBibles().then(setList).catch(() => setList([]))
  }, [])
  useEffect(refreshList, [refreshList])
  useEffect(() => {
    if (!selected) { setBible(null); setDraft(null); return }
    let live = true
    fetchDirectorBible(selected)
      .then(b => { if (live) { setBible(b); setDraft(b) } })
      .catch(e => { if (live) setError(e instanceof Error ? e.message : 'Bible not found') })
    return () => { live = false }
  }, [selected])

  // Server images changed (render, upload, approval): keep the user's unsaved text.
  const applyServer = useCallback((next: DirectorBible) => {
    setBible(next)
    setDraft(current => {
      if (!current || current.id !== next.id) return next
      const merged = { ...current, revision: next.revision, updated_at: next.updated_at }
      for (const plural of ['characters', 'locations', 'props'] as const) {
        const server = new Map(next[plural].map(e => [e.id, e]))
        merged[plural] = current[plural].map(e => {
          const stored = server.get(e.id)
          if (!stored) return e
          const variants = new Map(stored.variants.map(v => [v.id, v]))
          return { ...e, image: stored.image, variants: e.variants.map(v => ({ ...v, images: variants.get(v.id)?.images ?? v.images })) }
        })
      }
      return merged
    })
  }, [])

  const isRendering = rendering(bible)
  useEffect(() => {
    if (!isRendering || !selected) return
    const timer = window.setInterval(() => {
      fetchDirectorBible(selected).then(applyServer).catch(() => undefined)
    }, 3000)
    return () => window.clearInterval(timer)
  }, [isRendering, selected, applyServer])

  const save = async () => {
    if (!draft) return
    setSaving(true)
    setError(null)
    try {
      const next = await updateDirectorBible(draft)
      setBible(next)
      setDraft(next)
      refreshList()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Save failed')
    } finally {
      setSaving(false)
    }
  }
  const create = async () => {
    setError(null)
    try {
      const created = await createDirectorBible(newName.trim())
      setNewName('')
      refreshList()
      setSelected(created.id)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not create the bible')
    }
  }
  const remove = async () => {
    if (!bible) return
    if (!confirmDelete) { setConfirmDelete(true); return }
    await deleteDirectorBible(bible.id).catch(e => setError(e instanceof Error ? e.message : 'Delete failed'))
    setConfirmDelete(false)
    setSelected('')
    refreshList()
  }
  const savedIds = new Set(bible ? [...bible.characters, ...bible.locations, ...bible.props].map(e => e.id) : [])
  const elements = draft && tab !== 'world' ? draft[BIBLE_PLURAL[tab]] : []
  const setElements = (kind: BibleKind, next: BibleElement[]) => draft && setDraft({ ...draft, [BIBLE_PLURAL[kind]]: next })

  return (
    <div className="fixed inset-0 z-[80] flex items-center justify-center bg-black/50 p-4" role="dialog" aria-modal="true"
      aria-label="Bibles" onClick={e => { if (e.target === e.currentTarget) onClose() }}>
      <div className="w-full max-w-5xl h-[90vh] flex bg-bg-secondary border border-border rounded-xl overflow-hidden">
        <aside className="w-48 shrink-0 border-r border-border p-2.5 space-y-2 overflow-y-auto">
          <h2 className="text-sm font-medium text-text-primary">Bibles</h2>
          {list.map(item => (
            <button key={item.id} onClick={() => { setSelected(item.id); setConfirmDelete(false) }}
              className={`w-full text-left rounded px-2 py-1.5 text-[11px] ${item.id === selected ? 'bg-bg-hover text-text-primary' : 'text-text-secondary hover:bg-bg-hover'}`}>
              <span className="block truncate">{item.name}</span>
              <span className="block text-[9px] text-text-muted">
                {item.characters.length} char · {item.locations.length} loc · {item.props.length} props · r{item.revision}
              </span>
            </button>
          ))}
          <div className="space-y-1 border-t border-border/60 pt-2">
            <input className={input} placeholder="New series name" value={newName} onChange={e => setNewName(e.target.value)} aria-label="New bible name" />
            <button onClick={() => void create()} disabled={!newName.trim()}
              className="w-full py-1 rounded border border-border text-[10px] text-text-secondary hover:text-text-primary disabled:opacity-50 flex items-center justify-center gap-1">
              <Plus size={10} /> New bible
            </button>
          </div>
        </aside>

        <section className="flex-1 min-w-0 flex flex-col">
          <header className="flex items-center gap-2 border-b border-border p-2.5">
            {draft
              ? <input className={`${input} max-w-xs text-xs font-medium`} value={draft.name} aria-label="Bible name"
                  onChange={e => setDraft({ ...draft, name: e.target.value })} />
              : <span className="text-xs text-text-muted">Pick or create a bible</span>}
            {draft && <span className="text-[10px] text-text-muted">r{draft.revision}</span>}
            {isRendering && <span className="flex items-center gap-1 text-[10px] text-text-muted"><Loader2 size={10} className="animate-spin" /> rendering</span>}
            <div className="ml-auto flex items-center gap-2">
              {draft && (
                <button onClick={() => void remove()} className={`text-[10px] ${confirmDelete ? 'text-chip-red' : 'text-text-muted hover:text-chip-red'}`}>
                  {confirmDelete ? 'Click again to delete' : 'Delete'}
                </button>
              )}
              {draft && (
                <button onClick={() => void save()} disabled={!dirty || saving}
                  className="px-3 py-1 rounded bg-accent-blue text-white text-[11px] disabled:opacity-40 flex items-center gap-1">
                  {saving ? <Loader2 size={10} className="animate-spin" /> : <Check size={10} />} Save
                </button>
              )}
              <button onClick={onClose} className="text-text-muted hover:text-text-primary" aria-label="Close"><X size={16} /></button>
            </div>
          </header>
          {error && <p className="px-3 pt-2 text-[10px] text-chip-red">{error}</p>}

          {draft && bible && (
            <>
              <nav className="flex gap-1 px-2.5 pt-2">
                {TABS.map(([id, title]) => (
                  <button key={id} onClick={() => setTab(id)}
                    className={`rounded px-2 py-1 text-[11px] ${tab === id ? 'bg-bg-hover text-text-primary' : 'text-text-secondary hover:text-text-primary'}`}>
                    {title}{id !== 'world' ? ` (${draft[BIBLE_PLURAL[id]].length})` : ''}
                  </button>
                ))}
              </nav>
              <div className="flex-1 overflow-y-auto p-2.5">
                {tab === 'world' ? (
                  <div className="space-y-2 max-w-2xl">
                    <label className="block"><span className={label}>Description</span>
                      <textarea className={input} rows={2} value={draft.description}
                        onChange={e => setDraft({ ...draft, description: e.target.value })} /></label>
                    {([['logline', 'Series premise'], ['tone', 'Tone'],
                      ['look', 'Look: camera, lenses, film stock, palette (pasted into every image prompt)'],
                      ['rules', 'World rules (positive form works best)'], ['forbidden', 'Never']] as const).map(([field, title]) => (
                      <label key={field} className="block"><span className={label}>{title}</span>
                        <textarea className={input} rows={field === 'rules' || field === 'look' ? 4 : 2} value={draft.world[field] ?? ''}
                          onChange={e => setDraft({ ...draft, world: { ...draft.world, [field]: e.target.value } })} /></label>
                    ))}
                    <div>
                      <span className={label}>Style contract</span>
                      <pre className="whitespace-pre-wrap rounded bg-bg-tertiary p-2 text-[10px] text-text-secondary">
                        {draft.style ? JSON.stringify(draft.style, null, 1) : 'None yet: the first project saved into this bible sets it.'}
                      </pre>
                    </div>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-2.5">
                    {elements.map((element, index) => (
                      <ElementCard key={element.id || `new-${index}`} bible={bible} element={element}
                        saved={!!element.id && savedIds.has(element.id)} onBible={applyServer}
                        onChange={next => setElements(tab, elements.map((e, i) => (i === index ? next : e)))}
                        onRemove={() => setElements(tab, elements.filter((_, i) => i !== index))} />
                    ))}
                    <button onClick={() => setElements(tab, [...elements, {
                      id: '', kind: tab, name: '', tag: '', description: '', image: null,
                      variants: [{ id: 'default', label: 'Default', description: '', images: [] }],
                    }])}
                      className="min-h-24 rounded-lg border border-dashed border-border text-[11px] text-text-muted hover:text-text-primary flex items-center justify-center gap-1">
                      <Plus size={12} /> Add {tab === 'character' ? 'a character' : tab === 'location' ? 'a location' : 'a prop'}
                    </button>
                  </div>
                )}
              </div>
            </>
          )}
        </section>
      </div>
    </div>
  )
}
