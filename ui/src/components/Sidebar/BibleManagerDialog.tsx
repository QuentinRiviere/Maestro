import { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react'
import { AlertTriangle, Check, ChevronLeft, ChevronRight, Clock, ImagePlus, Loader2, Plus, RefreshCw, Sparkles, Trash2, Undo2, X } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import { DIRECTOR_DEFAULT_IMAGE_MODEL } from '../../lib/directorImageDefaults'
import { AppModeToggle, MaestroBrand } from '../AppModeNavigation'
import { BibleChat } from './BibleChat'
import { BibleStyleReferences } from './BibleStyleReferences'
import { BibleStyleLora } from './BibleStyleLora'
import { renderBiblePanorama } from '../../api/directorBibles'
import { useConfirm } from './ConfirmDialog'
import { fetchRemoteWorkerSettings, updateRemoteWorkerSettings } from '../../api/client'
import type { RemoteWorkerSettings } from '../../types'
import {
  approveBibleImage, BIBLE_PLURAL, renderBibleViews, unapproveBibleImage, createDirectorBible, deleteBibleImage, deleteDirectorBible,
  directorBibleImageUrl, fetchDirectorBible, fetchDirectorBibles, generateBibleAssets, renderBibleImage, updateDirectorBible,
  retryBibleImage, uploadBibleImage,
  type BibleElement, type BibleKind, type BibleVariant, type DirectorBible, type DirectorBibleSummary,
} from '../../api/directorBibles'

const input = 'w-full bg-bg-tertiary border border-border rounded px-2 py-1 text-[11px] text-text-primary focus:outline-none focus:border-accent-blue'
const label = 'text-[9px] text-text-muted uppercase tracking-wider block mb-0.5'
type BibleTab = BibleKind | 'world' | 'style'
const TABS: Array<[BibleTab, string]> = [
  ['character', 'Characters'], ['location', 'Locations'], ['prop', 'Props'], ['world', 'World & style'],
  ['style', 'Style references'],
]
const FIELDS: Record<BibleKind, Array<[keyof BibleElement, string, boolean]>> = {
  character: [['role', 'Role', false], ['descriptor', 'How shots name them', false],
    ['description', 'Look', true], ['wardrobe', 'Default wardrobe', false],
    ['personality', 'Personality', false], ['arc', 'Arc', false]],
  location: [['description', 'Place', true], ['notes', 'Geometry & physics (doors, scale, light)', false]],
  prop: [['descriptor', 'How shots name it', false], ['description', 'Object', true],
    ['notes', 'Scale & details', false]],
}

const pendingImages = (bible: DirectorBible) => (['characters', 'locations', 'props'] as const)
  .reduce((count, plural) => count + bible[plural].reduce((n, e) => n + e.variants
    .filter(v => !v.images.some(i => !i.view && (i.status === 'approved' || i.status === 'queued' || i.status === 'rendering'))).length, 0), 0)
/** Whether a model can edit from a reference image (looks need it to keep the same person or place). */
function useReadsImages(modelType: string): boolean {
  return useStore(s => {
    const model = s.models.find(m => m.model_type === modelType)
    return !model?.cloud_provider || !!model.cloud_reads_images
  })
}
const BIBLE_MODEL_KEY = 'maestro.bibleImageModel'
// Qwen-Image-Edit 2511 keeps faces natural at full sheet size and follows the description closely;
// Nano Banana is close behind; Seedream 4 over-sharpens faces (halos, waxy skin, odd ears).
const PREFERRED_BIBLE_MODELS = [
  'cloud_fal_qwen_image_edit_2511', 'cloud_fal_nano_banana', 'cloud_replicate_nano_banana', 'cloud_fal_seedream_4',
]
const readStoredModel = () => { try { return window.localStorage.getItem(BIBLE_MODEL_KEY) } catch { return null } }

/** The image model bibles render with: picked on the Bibles page, apart from the Director's own model. */
function useBibleImageModel(): [string, (model: string) => void] {
  const directorModel = useStore(s => s.selectedModelPerMode.image || DIRECTOR_DEFAULT_IMAGE_MODEL)
  const models = useStore(s => s.models)
  const [picked, setPicked] = useState<string | null>(readStoredModel)
  const reads = (id: string) => {
    const model = models.find(m => m.model_type === id)
    return !!model && (!model.cloud_provider || !!model.cloud_reads_images)
  }
  const fallback = reads(directorModel) ? directorModel
    : PREFERRED_BIBLE_MODELS.find(id => models.some(m => m.model_type === id)) ?? directorModel
  const pick = useCallback((model: string) => {
    setPicked(model)
    try { window.localStorage.setItem(BIBLE_MODEL_KEY, model) } catch { /* per-viewer convenience only */ }
  }, [])
  return [picked && models.some(m => m.model_type === picked) ? picked : fallback, pick]
}
const BibleModelContext = createContext(DIRECTOR_DEFAULT_IMAGE_MODEL)

function BibleModelPicker({ value, onChange }: { value: string; onChange: (model: string) => void }) {
  const models = useStore(s => s.models)
  const enabledModels = useStore(s => s.enabledModels)
  const nsfwMode = useStore(s => s.servicesConfig?.nsfw_mode ?? false)
  const choices = models.filter(m => m.director?.image.compatible === true
    && (enabledModels.has(m.model_type) || m.model_type === value) && (!m.nsfw_only || nsfwMode))
  const editing = choices.filter(m => !m.cloud_provider || m.cloud_reads_images)
  const textOnly = choices.filter(m => m.cloud_provider && !m.cloud_reads_images)
  const current = models.find(m => m.model_type === value)
  return (
    <select value={value} onChange={e => onChange(e.target.value)} aria-label="Image model for bible renders"
      title="Image model for bible renders. Looks other than Default need a model that reads images."
      className="max-w-[14rem] bg-bg-tertiary border border-border rounded px-2 py-1 text-[11px] text-text-primary focus:outline-none focus:border-accent-blue">
      {!choices.some(m => m.model_type === value) && <option value={value}>{current?.name ?? value}</option>}
      <optgroup label="Reads images (every look)">
        {editing.map(m => <option key={m.model_type} value={m.model_type}>{m.name}</option>)}
      </optgroup>
      {textOnly.length > 0 && (
        <optgroup label="Text only (Default looks only)">
          {textOnly.map(m => <option key={m.model_type} value={m.model_type}>{m.name}</option>)}
        </optgroup>
      )}
    </select>
  )
}
const TEXT_ONLY_NOTE = 'cannot read an image: looks other than Default need a model that reads images to keep the same person or place. Pick one (Nano Banana, Seedream 4, FLUX Kontext, Klein, Qwen-Image-Edit) in the model menu at the top of the page.'

const WORKER_CHOICES = [1, 2, 3]

/** How many remote GPU workers local-model renders spread over (a remote worker setting, shared with every job). */
function WorkerCountPicker({ imageModel }: { imageModel: string }) {
  const [settings, setSettings] = useState<RemoteWorkerSettings | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => { fetchRemoteWorkerSettings().then(setSettings).catch(() => setSettings(null)) }, [])
  if (!settings?.configured || !settings.route_generations || imageModel.startsWith('cloud_')) return null
  const pick = async (count: number) => {
    setError(null)
    try { setSettings(await updateRemoteWorkerSettings({ worker_count: count })) }
    catch (e) { setError(e instanceof Error ? e.message : 'Could not change the worker count') }
  }
  return (
    <div className="flex items-center gap-1 text-[10px] text-text-muted"
      title={'Remote GPU workers the renders spread over. Each render takes the first worker free; '
        + 'extra workers start only while renders are waiting, and each one is billed while it runs.'}>
      <span>Workers</span>
      <div className="flex overflow-hidden rounded border border-border">
        {WORKER_CHOICES.map(count => (
          <button key={count} onClick={() => void pick(count)} aria-pressed={settings.worker_count === count}
            className={`px-1.5 py-0.5 ${settings.worker_count === count ? 'bg-accent-blue/30 text-text-primary' : 'hover:bg-bg-hover'}`}>
            {count}
          </button>
        ))}
      </div>
      {error && <span className="text-chip-red">{error}</span>}
    </div>
  )
}

/** Render a failed image again, or a new candidate next to an image: pick the model (fal or the GPU worker)
 * and, for the worker, how many pods. */
function RetryDialog({ regenerate = false, needsImages, onRetry, onClose }: {
  regenerate?: boolean
  needsImages: boolean
  onRetry: (model: string) => Promise<void>
  onClose: () => void
}) {
  const [model, setModel] = useState(useContext(BibleModelContext))
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const readsImages = useReadsImages(model)
  const blocked = needsImages && !readsImages
  const retry = async () => {
    setBusy(true)
    setError(null)
    try { await onRetry(model); onClose() }
    catch (e) { setError(e instanceof Error ? e.message : `Could not ${regenerate ? 'start' : 'retry'} the render`) }
    finally { setBusy(false) }
  }
  return (
    <div className="fixed inset-0 z-[95] flex items-center justify-center bg-black/60" role="dialog" aria-modal="true"
      aria-label={regenerate ? 'Regenerate the image' : 'Retry the render'} onClick={e => { if (e.target === e.currentTarget) onClose() }}>
      <div className="w-80 space-y-3 rounded border border-border bg-bg-secondary p-3 text-[11px] text-text-secondary">
        <h3 className="text-xs font-medium text-text-primary">{regenerate ? 'Regenerate this image' : 'Retry the render'}</h3>
        {regenerate && <p className="text-[10px] text-text-muted">A new candidate joins this one; nothing is replaced until you approve it.</p>}
        <label className="block space-y-1">
          <span className="block text-[10px] text-text-muted">Model: a cloud model (fal, Replicate) or a local one on the GPU worker</span>
          <BibleModelPicker value={model} onChange={setModel} />
        </label>
        <WorkerCountPicker imageModel={model} />
        {blocked && <p className="text-amber-300">{model} {TEXT_ONLY_NOTE}</p>}
        {error && <p className="text-chip-red">{error}</p>}
        <div className="flex justify-end gap-2">
          <button onClick={onClose} className="rounded border border-border px-2 py-1 hover:text-text-primary">Cancel</button>
          <button disabled={busy || blocked} onClick={() => void retry()}
            className="flex items-center gap-1 rounded border border-accent-blue px-2 py-1 text-text-primary disabled:opacity-40">
            {busy ? <Loader2 size={10} className="animate-spin" /> : <RefreshCw size={10} />} {regenerate ? 'Regenerate' : 'Retry'}
          </button>
        </div>
      </div>
    </div>
  )
}

const lookCount = (bible: DirectorBible) => (['characters', 'locations', 'props'] as const)
  .reduce((count, plural) => count + bible[plural].reduce((n, e) => n + e.variants.length, 0), 0)
const slug = (text: string) => text.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '').slice(0, 32)
const rendering = (bible: DirectorBible | null) => !!bible && (['characters', 'locations', 'props'] as const)
  .some(plural => bible[plural].some(e => e.variants.some(v => v.images.some(i => i.status === 'queued' || i.status === 'rendering'))))

type Ask = ReturnType<typeof useConfirm>['ask']
const AskContext = createContext<Ask>(({ run }) => void run())
const DELETE_IMAGE_NOTE = 'The image leaves this look. Projects that already used it keep their own copy.'

const inProgress = (status: string) => status === 'queued' || status === 'rendering'

/** One image of a look, full size: approve it, delete it, or step through the look's images. */
function ImageViewer({ bible, element, variant, view, index, onIndex, onClose, onBible, onRegenerate }: {
  bible: DirectorBible
  element: BibleElement
  variant: BibleVariant
  view: string
  index: number
  onIndex: (index: number) => void
  onClose: () => void
  onBible: (bible: DirectorBible) => void
  onRegenerate: () => void
}) {
  const images = variant.images.filter(i => i.file && (i.view || '') === view)
  const image = images[Math.min(index, images.length - 1)]
  const [busy, setBusy] = useState(false)
  const ask = useContext(AskContext)
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
      if (e.key === 'ArrowRight') onIndex(Math.min(images.length - 1, index + 1))
      if (e.key === 'ArrowLeft') onIndex(Math.max(0, index - 1))
    }
    window.addEventListener('keydown', key)
    return () => window.removeEventListener('keydown', key)
  }, [images.length, index, onClose, onIndex])
  if (!image?.file) return null
  const act = async (call: () => Promise<DirectorBible>, close = false) => {
    setBusy(true)
    try { onBible(await call()); if (close) onClose() } finally { setBusy(false) }
  }
  return (
    <div className="fixed inset-0 z-[90] flex flex-col bg-black/90 p-4" role="dialog" aria-modal="true"
      aria-label={`${element.name}, ${variant.label}`} onClick={e => { if (e.target === e.currentTarget) onClose() }}>
      <div className="flex items-center gap-3 pb-3 text-[12px] text-white/90">
        <span className="font-medium">{element.name}</span>
        <span className="text-white/60">{variant.label} · {index + 1}/{images.length}</span>
        {image.status === 'approved' && <span className="flex items-center gap-1 text-emerald-300"><Check size={12} /> Approved reference</span>}
        <div className="ml-auto flex items-center gap-2">
          {image.status === 'approved'
            ? <button disabled={busy} onClick={() => void act(() => unapproveBibleImage(bible.id, element.id, variant.id, image.id))}
                title="Back to candidate: this look has no reference until you approve another image"
                className="flex items-center gap-1 rounded border border-emerald-300/60 px-2.5 py-1 text-[11px] text-emerald-200 hover:text-white disabled:opacity-50">
                <Undo2 size={12} /> Unapprove
              </button>
            : <button disabled={busy} onClick={() => void act(() => approveBibleImage(bible.id, element.id, variant.id, image.id))}
                className="flex items-center gap-1 rounded bg-emerald-600 px-2.5 py-1 text-[11px] text-white disabled:opacity-50">
                <Check size={12} /> Approve
              </button>}
          <button disabled={busy} onClick={onRegenerate} title="Render a new candidate of this image"
            className="flex items-center gap-1 rounded border border-white/30 px-2.5 py-1 text-[11px] text-white/80 hover:text-white disabled:opacity-50">
            <RefreshCw size={12} /> Regenerate
          </button>
          <button disabled={busy} onClick={() => ask({
            title: 'Delete this image?', message: DELETE_IMAGE_NOTE, confirmLabel: 'Delete', danger: true,
            run: () => act(() => deleteBibleImage(bible.id, element.id, variant.id, image.id), true),
          })}
            className="flex items-center gap-1 rounded border border-white/30 px-2.5 py-1 text-[11px] text-white/80 hover:text-white disabled:opacity-50">
            <Trash2 size={12} /> Delete
          </button>
          <button onClick={onClose} aria-label="Close" className="text-white/70 hover:text-white"><X size={18} /></button>
        </div>
      </div>
      <div className="relative flex min-h-0 flex-1 items-center justify-center">
        {index > 0 && (
          <button onClick={() => onIndex(index - 1)} aria-label="Previous image"
            className="absolute left-0 rounded-full bg-white/10 p-2 text-white hover:bg-white/20"><ChevronLeft size={20} /></button>
        )}
        <img src={directorBibleImageUrl(bible.id, image.file)} alt={`${element.name} ${variant.label}`}
          className="max-h-full max-w-full object-contain" />
        {index < images.length - 1 && (
          <button onClick={() => onIndex(index + 1)} aria-label="Next image"
            className="absolute right-0 rounded-full bg-white/10 p-2 text-white hover:bg-white/20"><ChevronRight size={20} /></button>
        )}
      </div>
    </div>
  )
}

/** The images of a look, or of one of its angles (``view``). */
function Images({ bible, element, variant, view = '', onBible }: {
  bible: DirectorBible
  element: BibleElement
  variant: BibleVariant
  view?: string
  onBible: (bible: DirectorBible) => void
}) {
  const [error, setError] = useState<string | null>(null)
  const [viewing, setViewing] = useState<number | null>(null)
  const ask = useContext(AskContext)
  // A failed slot to retry in place, or "new" for a new candidate of this look or angle.
  const [retrying, setRetrying] = useState<string | null>(null)
  const act = async (call: () => Promise<DirectorBible>) => {
    setError(null)
    try { onBible(await call()) } catch (e) { setError(e instanceof Error ? e.message : 'Failed') }
  }
  const shown = variant.images.filter(i => (i.view || '') === view)
  if (!shown.length) return view ? null : <p className="text-[10px] text-text-muted">No image yet: render or upload one.</p>
  const withFiles = shown.filter(i => i.file)
  return (
    <div className="space-y-1">
      <div className="flex flex-wrap gap-1.5">
        {shown.map(image => (
          <div key={image.id} className={`relative w-28 aspect-video rounded overflow-hidden border ${
            image.status === 'approved' ? 'border-emerald-400' : 'border-border'} bg-bg-tertiary`}>
            {image.file
              ? <button className="w-full h-full" title="View full size"
                  onClick={() => setViewing(withFiles.findIndex(i => i.id === image.id))}>
                  <img src={directorBibleImageUrl(bible.id, image.file)} alt={`${element.name} ${variant.label}`} className="w-full h-full object-contain" />
                </button>
              : <div className="w-full h-full flex flex-col items-center justify-center gap-0.5 text-[9px] text-text-muted" title={image.error || image.status}>
                  {image.status === 'rendering'
                    ? <><Loader2 size={14} className="animate-spin" /> rendering</>
                    : image.status === 'queued'
                      ? <><Clock size={14} /> queued</>
                      : <>
                          <AlertTriangle size={14} className="text-chip-red" />
                          <button onClick={() => setRetrying(image.id)} title="Render it again"
                            className="flex items-center gap-0.5 rounded border border-border px-1 text-text-secondary hover:text-text-primary">
                            <RefreshCw size={9} /> Retry
                          </button>
                        </>}
                </div>}
            {image.status === 'approved' && <Check size={12} className="absolute left-1 top-1 rounded bg-emerald-500 text-white" />}
            {image.file && !inProgress(image.status) && (
              <button aria-label="Regenerate image" title="Render a new candidate of this image" onClick={() => setRetrying('new')}
                className="absolute right-0.5 bottom-0.5 rounded bg-black/60 p-0.5 text-white/80 hover:text-white">
                <RefreshCw size={10} />
              </button>
            )}
            {!inProgress(image.status) && (
              <button aria-label="Delete image" onClick={() => ask({
                title: 'Delete this image?', message: DELETE_IMAGE_NOTE, confirmLabel: 'Delete', danger: true,
                run: () => act(() => deleteBibleImage(bible.id, element.id, variant.id, image.id)),
              })}
                className="absolute right-0.5 top-0.5 rounded bg-black/60 p-0.5 text-white/80 hover:text-white">
                <X size={10} />
              </button>
            )}
          </div>
        ))}
      </div>
      {shown.some(i => i.status === 'failed') && (
        <p className="text-[9px] text-chip-red truncate" title={shown.find(i => i.status === 'failed')?.error}>
          {shown.find(i => i.status === 'failed')?.error}
        </p>
      )}
      {error && <p className="text-[10px] text-chip-red">{error}</p>}
      {viewing !== null && withFiles.length > 0 && (
        <ImageViewer bible={bible} element={element} variant={variant} view={view} index={viewing} onIndex={setViewing}
          onClose={() => setViewing(null)} onBible={onBible} onRegenerate={() => setRetrying('new')} />
      )}
      {retrying && (
        <RetryDialog regenerate={retrying === 'new'} needsImages={!!view || variant.id !== 'default'}
          onClose={() => setRetrying(null)}
          onRetry={async model => onBible(await (retrying !== 'new'
            ? retryBibleImage(bible.id, element.id, variant.id, retrying, model)
            : view
              ? renderBibleViews(bible.id, element.id, variant.id, model, [view])
              : renderBibleImage(bible.id, element.id, variant.id, { image_model: model })))} />
      )}
    </div>
  )
}

const VIEW_LABELS: Record<string, string> = {
  face_front: 'Face', face_three_quarter: 'Face 3/4', face_profile: 'Profile', back: 'Back',
}
/** A place's angle: the same empty room from the opposite wall, for shots that look back. */
const LOCATION_VIEW_LABELS: Record<string, string> = { reverse: 'Reverse', left: 'Left wall', right: 'Right wall' }

/**
 * The angles of a look, edited from its approved image: a character's approved ones make its turnaround
 * sheet, a place's angles (reverse, left wall, right wall) are the sets of the shots whose camera looks at that wall.
 */
function Angles({ bible, element, variant, readsImages, imageModel, onBible }: {
  bible: DirectorBible
  element: BibleElement
  variant: BibleVariant
  readsImages: boolean
  imageModel: string
  onBible: (bible: DirectorBible) => void
}) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const place = element.kind === 'location'
  const labels = place ? LOCATION_VIEW_LABELS : VIEW_LABELS
  const total = Object.keys(labels).length
  const hasMain = variant.images.some(i => !i.view && i.status === 'approved')
  const angles = variant.images.filter(i => i.view)
  const running = angles.some(i => i.status === 'queued' || i.status === 'rendering')
  const approved = new Set(angles.filter(i => i.status === 'approved').map(i => i.view))
  const render = async () => {
    setBusy(true)
    setError(null)
    try { onBible(await renderBibleViews(bible.id, element.id, variant.id, imageModel)) }
    catch (e) { setError(e instanceof Error ? e.message : 'Could not render the angles') }
    finally { setBusy(false) }
  }
  const panorama = async () => {
    setBusy(true)
    setError(null)
    try { onBible(await renderBiblePanorama(bible.id, element.id, variant.id)) }
    catch (e) { setError(e instanceof Error ? e.message : 'Could not render the panorama') }
    finally { setBusy(false) }
  }
  return (
    <div className="space-y-1 rounded border border-border/60 p-1.5">
      <div className="flex items-center gap-2">
        <span className={label + ' !mb-0'}>Angles</span>
        <span className="text-[9px] text-text-muted">
          {place
            ? (approved.size ? `${approved.size}/${total} walls: used by shots whose camera looks at that wall` : 'the same empty place looking at its other walls (reverse, left, right), one per camera direction')
            : approved.size ? `${approved.size}/${total} in the turnaround sheet projects use` : 'face, 3/4, profile and back, for a turnaround sheet'}
        </span>
        <button disabled={busy || running || !hasMain || !readsImages} onClick={() => void render()}
          title={!hasMain ? 'Approve an image of this look first' : !readsImages ? `${imageModel} cannot read an image`
            : `Render the ${total} angles with ${imageModel}`}
          className="ml-auto flex items-center gap-1 rounded border border-border px-2 py-0.5 text-[10px] text-text-secondary hover:text-text-primary disabled:opacity-40">
          {busy || running ? <Loader2 size={10} className="animate-spin" /> : <Sparkles size={10} />} {angles.length ? 'Re-render angles' : 'Render angles'}
        </button>
        {place && (
          <button disabled={busy || running || !hasMain} onClick={() => void panorama()}
            title="One 360° panorama painted around the approved plate (FLUX.2 Klein + 360 outpaint LoRA; from text with the trained panorama LoRA otherwise), then every angle cut from it: the geometry holds between angles"
            className="flex items-center gap-1 rounded border border-border px-2 py-0.5 text-[10px] text-text-secondary hover:text-text-primary disabled:opacity-40">
            <Sparkles size={10} /> Panorama
          </button>
        )}
      </div>
      {Object.entries(labels).map(([view, name]) => angles.some(i => i.view === view) && (
        <div key={view} className="flex items-start gap-1.5">
          <span className="w-12 shrink-0 pt-1 text-[9px] text-text-muted">{name}</span>
          <Images bible={bible} element={element} variant={variant} view={view} onBible={onBible} />
        </div>
      ))}
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
  const imageModel = useContext(BibleModelContext)
  const readsImages = useReadsImages(imageModel)
  const [variantId, setVariantId] = useState('default')
  const [newLook, setNewLook] = useState<{ label: string; description: string } | null>(null)
  const [chain, setChain] = useState<string[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const upload = useRef<HTMLInputElement>(null)
  const variant = element.variants.find(v => v.id === variantId) ?? element.variants[0]
  const ask = useContext(AskContext)
  const lookRendering = element.variants.some(v => v.images.some(i => !i.view && inProgress(i.status)))
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
        {saved && (
          <button disabled={busy || lookRendering}
            title={lookRendering ? 'Wait for the renders under way' : `Render a new candidate of every look of ${element.name} with ${imageModel}`}
            onClick={() => ask({
              title: `Regenerate every look of ${element.name}?`, confirmLabel: 'Regenerate all',
              message: <>A new candidate for each of its {element.variants.length} look{element.variants.length > 1 ? 's' : ''},
                rendered with <b>{imageModel}</b>. Approved images stay the references until you approve a new one.
                {element.kind !== 'prop' && ' Angles are not re-rendered: use Re-render angles once a new look is approved.'}
                {!readsImages && <span className="mt-2 block text-amber-300">{imageModel} {TEXT_ONLY_NOTE} Only the Default look will render.</span>}</>,
              run: () => act(() => generateBibleAssets(bible.id, imageModel, true, element.id)),
            })}
            className="shrink-0 flex items-center gap-1 rounded border border-border px-1.5 py-0.5 text-[10px] text-text-secondary hover:text-text-primary disabled:opacity-40">
            <RefreshCw size={10} /> Regenerate all
          </button>
        )}
        <button aria-label="Remove element" className="shrink-0 text-text-muted hover:text-chip-red"
          onClick={() => ask({
            title: `Remove ${element.name || 'this element'}?`, confirmLabel: 'Remove', danger: true,
            message: 'It leaves the draft with its looks and images; Save makes it final.', run: onRemove,
          })}><Trash2 size={12} /></button>
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
        {variant && saved && (element.kind === 'character' || element.kind === 'location') && (
          <Angles bible={bible} element={element} variant={variant} readsImages={readsImages} imageModel={imageModel} onBible={onBible} />
        )}

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
            {!readsImages && variant.id !== 'default' && (
              <p className="text-[9px] text-amber-300">{imageModel} {TEXT_ONLY_NOTE}</p>
            )}
            <div className="flex gap-1.5">
              <button disabled={busy || (!readsImages && variant.id !== 'default')}
                onClick={() => void act(() => renderBibleImage(bible.id, element.id, variant.id, { image_model: imageModel, chain_from: chain }))}
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
function BibleManagerPanel({ initialId, onClose }: { initialId?: string; onClose?: () => void }) {
  const [list, setList] = useState<DirectorBibleSummary[]>([])
  const [selected, setSelected] = useState(initialId || '')
  const [bible, setBible] = useState<DirectorBible | null>(null)
  const [draft, setDraft] = useState<DirectorBible | null>(null)
  const [tab, setTab] = useState<BibleTab>('character')
  const [newName, setNewName] = useState('')
  const [saving, setSaving] = useState(false)
  const { ask, dialog: confirmDialog } = useConfirm()
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
      // Style references are server-owned (their own endpoints), like images.
      const merged = { ...current, revision: next.revision, updated_at: next.updated_at, style_references: next.style_references }
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

  const [imageModel, setImageModel] = useBibleImageModel()
  const readsImages = useReadsImages(imageModel)
  // Generate saves the draft first: the server renders the stored bible.
  const generateAssets = async () => {
    if (!draft) return
    const saved = await updateDirectorBible(draft)
    const next = await generateBibleAssets(saved.id, imageModel)
    setBible(next)
    setDraft(next)
    refreshList()
  }

  const regenerateAll = async () => {
    if (!draft) return
    setError(null)
    try {
      const saved = dirty ? await updateDirectorBible(draft) : draft
      const next = await generateBibleAssets(saved.id, imageModel, true)
      setBible(next)
      setDraft(next)
      refreshList()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not start the generation')
    }
  }

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
    await deleteDirectorBible(bible.id).catch(e => setError(e instanceof Error ? e.message : 'Delete failed'))
    setSelected('')
    refreshList()
  }
  const savedIds = new Set(bible ? [...bible.characters, ...bible.locations, ...bible.props].map(e => e.id) : [])
  const elements = draft && tab !== 'world' && tab !== 'style' ? draft[BIBLE_PLURAL[tab]] : []
  const setElements = (kind: BibleKind, next: BibleElement[]) => draft && setDraft({ ...draft, [BIBLE_PLURAL[kind]]: next })

  return (
    <AskContext.Provider value={ask}>
    <BibleModelContext.Provider value={imageModel}>
      <div className="w-full h-full flex bg-bg-secondary overflow-hidden">
        <aside className="w-48 shrink-0 border-r border-border p-2.5 space-y-2 overflow-y-auto">
          <h2 className="text-sm font-medium text-text-primary">Bibles</h2>
          {list.map(item => (
            <button key={item.id} onClick={() => setSelected(item.id)}
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
              {draft && <WorkerCountPicker imageModel={imageModel} />}
              {draft && <BibleModelPicker value={imageModel} onChange={setImageModel} />}
              {draft && (
                <button className="text-[10px] text-text-muted hover:text-chip-red"
                  onClick={() => ask({
                    title: `Delete the bible "${draft.name}"?`, confirmLabel: 'Delete bible', danger: true,
                    message: 'Its characters, places, props, looks and images are deleted. Projects already made from it keep their copies.',
                    run: remove,
                  })}>
                  Delete
                </button>
              )}
              {draft && (
                <button disabled={isRendering || !lookCount(draft)}
                  title={isRendering ? 'Wait for the renders under way' : `Render a new candidate of every look with ${imageModel}`}
                  onClick={() => ask({
                    title: 'Regenerate every look?', confirmLabel: 'Regenerate all',
                    message: <>A new candidate for each of the {lookCount(draft)} looks, rendered with <b>{imageModel}</b>
                      {dirty ? ', after saving your changes' : ''}. Approved images stay the references until you approve a new one.
                      {!readsImages && <span className="mt-2 block text-amber-300">{imageModel} {TEXT_ONLY_NOTE} Only Default looks will render.</span>}</>,
                    run: regenerateAll,
                  })}
                  className="flex items-center gap-1 rounded border border-border px-2.5 py-1 text-[11px] text-text-secondary hover:text-text-primary disabled:opacity-40">
                  <RefreshCw size={10} /> Regenerate all
                </button>
              )}
              {draft && (
                <button onClick={() => void save()} disabled={!dirty || saving}
                  className="px-3 py-1 rounded bg-accent-blue text-white text-[11px] disabled:opacity-40 flex items-center gap-1">
                  {saving ? <Loader2 size={10} className="animate-spin" /> : <Check size={10} />} Save
                </button>
              )}
              {onClose && <button onClick={onClose} className="text-text-muted hover:text-text-primary" aria-label="Close"><X size={16} /></button>}
            </div>
          </header>
          {error && <p className="px-3 pt-2 text-[10px] text-chip-red">{error}</p>}

          {draft && bible && (
            <>
              <nav className="flex gap-1 px-2.5 pt-2">
                {TABS.map(([id, title]) => (
                  <button key={id} onClick={() => setTab(id)}
                    className={`rounded px-2 py-1 text-[11px] ${tab === id ? 'bg-bg-hover text-text-primary' : 'text-text-secondary hover:text-text-primary'}`}>
                    {title}{id === 'style'
                      ? (bible.style_references?.length ? ` (${bible.style_references.length})` : '')
                      : id !== 'world' ? ` (${draft[BIBLE_PLURAL[id]].length})` : ''}
                  </button>
                ))}
              </nav>
              <div className="flex-1 overflow-y-auto p-2.5">
                {tab === 'style' ? (
                  <div className="space-y-4">
                    <BibleStyleReferences key={draft.id} bibleId={draft.id} ask={ask} />
                    <BibleStyleLora key={`lora-${draft.id}`} bibleId={draft.id} ask={ask} />
                  </div>
                ) : tab === 'world' ? (
                  <div className="space-y-2 max-w-2xl">
                    <label className="block"><span className={label}>Description</span>
                      <textarea className={input} rows={2} value={draft.description}
                        onChange={e => setDraft({ ...draft, description: e.target.value })} /></label>
                    {([['logline', 'Series premise'], ['story', 'Story'], ['stakes', 'Stakes'],
                      ['twists', 'Possible twists'], ['tone', 'Tone'], ['narrative_style', 'Narrative style'],
                      ['look', 'Look: camera, lenses, film stock, palette (pasted into every image prompt)'],
                      ['rules', 'World rules (positive form works best)'], ['forbidden', 'Never']] as const).map(([field, title]) => (
                      <label key={field} className="block"><span className={label}>{title}</span>
                        <textarea className={input} rows={['story', 'twists', 'rules', 'look'].includes(field) ? 4 : 2} value={draft.world[field] ?? ''}
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
        {draft && (
          <BibleChat key={draft.id} bibleId={draft.id} draft={draft} onChange={setDraft} onGenerate={generateAssets}
            pendingImages={pendingImages(draft)} imageModel={imageModel} rendering={isRendering} />
        )}
      </div>
      {confirmDialog}
    </BibleModelContext.Provider>
    </AskContext.Provider>
  )
}

export function BibleManagerDialog({ initialId, onClose }: { initialId?: string; onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-[80] flex items-center justify-center bg-black/50 p-4" role="dialog" aria-modal="true"
      aria-label="Bibles" onClick={e => { if (e.target === e.currentTarget) onClose() }}>
      <div className="w-full max-w-6xl h-[90vh] border border-border rounded-xl overflow-hidden">
        <BibleManagerPanel initialId={initialId} onClose={onClose} />
      </div>
    </div>
  )
}

/** The Bibles tab: the manager on the whole canvas, with the app navigation. */
export function BiblesWorkspace() {
  const bibleId = useStore(s => s.directorBibleId)
  return (
    <div className="flex h-full w-full flex-col">
      <header className="flex h-12 shrink-0 items-center gap-3 border-b border-border bg-bg-secondary px-3">
        <MaestroBrand />
        <AppModeToggle size="sm" />
      </header>
      <div className="min-h-0 flex-1">
        <BibleManagerPanel initialId={bibleId || undefined} />
      </div>
    </div>
  )
}
