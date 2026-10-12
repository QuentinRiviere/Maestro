import { ChevronDown, Check, Plus, Globe } from 'lucide-react'
import { useState, useRef, useEffect, type ReactNode } from 'react'
import {
  useStore,
  getFamiliesForMode,
  getModelsForFamily,
  modelSupportsImageWorkflow,
  modelSupportsStudioVideoMediaIntent,
} from '../../stores/useStore'
import { InfoTooltip } from './InfoTooltip'
import { SidebarDialog } from './SidebarPanels'
import { isLongCatAvatarModel } from '../../lib/avatarWorkflow'

type Placement = 'above' | 'below' | 'footer'
type SourceFilter = 'all' | 'local' | 'fal' | 'replicate'

const SOURCE_FILTERS: Array<{ id: SourceFilter; label: string; title: string }> = [
  { id: 'all', label: 'All', title: 'Every model' },
  { id: 'local', label: 'Open weights', title: 'Models that run on your GPU or the remote GPU worker' },
  { id: 'fal', label: 'fal.ai', title: 'Hosted models billed by fal.ai' },
  { id: 'replicate', label: 'Replicate', title: 'Hosted models billed by Replicate' },
]
const CLOUD_LABELS: Record<string, string> = { fal: 'fal.ai', replicate: 'Replicate' }
const FILTER_STORAGE_KEY = 'maestro_model_selector_filters'

function readStoredFilters(): { source: SourceFilter; loraOnly: boolean } {
  try {
    const stored = JSON.parse(localStorage.getItem(FILTER_STORAGE_KEY) || '{}')
    const source = SOURCE_FILTERS.some(f => f.id === stored.source) ? stored.source : 'all'
    return { source, loraOnly: stored.loraOnly === true }
  } catch {
    return { source: 'all', loraOnly: false }
  }
}

function matchesSource(model: { cloud_provider?: string | null }, source: SourceFilter) {
  if (source === 'all') return true
  if (source === 'local') return !model.cloud_provider
  return model.cloud_provider === source
}

function ModelChoicesPanel({open, placement, onClose, children}: {open: boolean; placement: Placement; onClose: () => void; children: ReactNode}) {
  if (placement === 'footer') return <SidebarDialog open={open} title="Choose a model" variant="settings" onClose={onClose}>{open && children}</SidebarDialog>
  return open ? <div className={`${placement === 'below' ? 'relative mt-2 w-full' : 'absolute bottom-full left-0 mb-1 w-[360px] max-w-[90vw]'} bg-bg-secondary border border-border rounded-xl shadow-xl overflow-hidden z-50`}>{children}</div> : null
}

export function ModelSelector({ placement = 'above' }: { placement?: Placement }) {
  const models = useStore(s => s.models)
  const families = useStore(s => s.families)
  const enabledModels = useStore(s => s.enabledModels)
  const generationMode = useStore(s => s.generationMode)
  const editSubMode = useStore(s => s.editSubMode)
  const imageWorkflow = useStore(s => s.studioImageWorkflow)
  const hasImageReferences = useStore(s => s.imageRefs.length > 0)
  const studioVideoWorkflow = useStore(s => s.studioVideoWorkflow)
  const videoImageMode = useStore(s => Number(s.params.image_mode || 0))
  const hasFrameGuidance = useStore(s => Boolean(
    s.startImage
    || s.endImage
    || s.params.image_start
    || s.params.image_end
    || s.imageRefs.length
    || (
      Array.isArray(s.params.image_refs)
      && s.params.image_refs.length
      && s.params.frames_positions
    ),
  ))
  const hasOmniReferences = useStore(s => (
    s.params.minimax_h3_references?.some(reference => !(
      reference.type === 'audio' && reference.audio_intent === 'drive'
    )) === true
  ))
  const hasFrameAudioDrive = useStore(s => Boolean(s.params.audio_guide))
  const hasReferenceAudioDrive = useStore(s => Boolean(
    s.params.minimax_h3_references?.some(reference => (
      reference.type === 'audio' && reference.audio_intent === 'drive'
    )),
  ))
  const currentModelType = useStore(s => s.params.model_type)
  const selectModel = useStore(s => s.selectModel)
  const selectStudioVideoModel = useStore(s => s.selectStudioVideoModel)
  const openModelVisibility = useStore(s => s.openModelVisibility)
  // Mature Mode gate: models with nsfw_only flag are hidden from the
  // selector unless servicesConfig.nsfw_mode is enabled. Backend always
  // ships the entry (so the toggle can show/hide without a model reload)
  // but the UI clamps visibility here.
  const nsfwMode = useStore(s => s.servicesConfig?.nsfw_mode ?? false)

  const [open, setOpen] = useState(false)
  const [sourceFilter, setSourceFilter] = useState<SourceFilter>(() => readStoredFilters().source)
  const [loraOnly, setLoraOnly] = useState(() => readStoredFilters().loraOnly)
  useEffect(() => {
    try {
      localStorage.setItem(FILTER_STORAGE_KEY, JSON.stringify({ source: sourceFilter, loraOnly }))
    } catch { /* storage unavailable: filters reset next session */ }
  }, [sourceFilter, loraOnly])
  const containerRef = useRef<HTMLDivElement>(null)

  // Close on click outside
  useEffect(() => {
    if (!open || placement === 'footer') return
    function handleClick(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [open, placement])

  const audioSubMode = useStore(s => s.audioSubMode)

  const currentModel = models.find(m => m.model_type === currentModelType)
  const createWorkflow = studioVideoWorkflow === 'references' ? 'references'
    : studioVideoWorkflow === 'avatar' ? 'avatar' : 'frames'
  const isPrimaryStudioCreate = generationMode === 'video'
    && ['frames', 'references', 'avatar'].includes(studioVideoWorkflow)
    && videoImageMode === 0
  const studioMediaIntent = {
    workflow: createWorkflow,
    hasFrameGuidance: createWorkflow === 'frames' && hasFrameGuidance,
    hasOmniReferences: createWorkflow === 'references' && hasOmniReferences,
    hasAudioDrive: createWorkflow === 'references' ? hasReferenceAudioDrive : hasFrameAudioDrive,
  } as const
  const currentModelCompatible = (
    generationMode !== 'video' || studioVideoWorkflow === 'avatar' || !isLongCatAvatarModel(currentModel)
  ) && (
    generationMode !== 'image'
    || modelSupportsImageWorkflow(currentModel, imageWorkflow, hasImageReferences)
  ) && (
    !isPrimaryStudioCreate
    || modelSupportsStudioVideoMediaIntent(currentModel, studioMediaIntent)
  )
  const compatibilityDescription = generationMode === 'image'
    ? imageWorkflow === 'generate' && hasImageReferences
      ? 'Add or enable an image edit model for the supplied source or reference images.'
      : `Add or enable an image model compatible with ${imageWorkflow}.`
    : createWorkflow === 'avatar'
      ? 'Add or enable a LongCat Avatar Single or Multi model for image-and-voice animation.'
      : createWorkflow === 'references'
      ? 'Add or enable an H3 Omni model for these references or characters.'
      : hasFrameGuidance && hasFrameAudioDrive
        ? 'Add or enable a frame-capable model that accepts an exact audio timeline.'
        : hasFrameGuidance
          ? 'Add or enable an image-to-video model for these frame inputs.'
          : hasFrameAudioDrive
            ? 'Add or enable a video model that accepts an audio timeline.'
            : 'Add or enable a text-to-video or image-to-video model.'
  const effectiveSubMode = generationMode === 'avatar' ? editSubMode : undefined
  const effectiveAudioSubMode = generationMode === 'audio' ? audioSubMode : undefined
  const modeFamilies = getFamiliesForMode(generationMode, families, effectiveSubMode, effectiveAudioSubMode)

  // Build grouped model list, filtered by:
  //   1. enabledModels (Settings → System → Model Visibility),
  //   2. nsfw_only gate (Mature Mode must be on for those to appear).
  const workflowFilter = (model: typeof models[number]) => (
    (generationMode !== 'video' || (studioVideoWorkflow === 'animate'
      ? model.model_type === 'viggle_animate' : model.model_type !== 'viggle_animate'))
    && (generationMode !== 'video' || studioVideoWorkflow === 'avatar' || !isLongCatAvatarModel(model))
    &&
    (generationMode !== 'image' || modelSupportsImageWorkflow(model, imageWorkflow, hasImageReferences))
    && (
      !isPrimaryStudioCreate
      || modelSupportsStudioVideoMediaIntent(model, studioMediaIntent)
    )
  )
  const visibleGroups = modeFamilies.map(family => ({
    family,
    models: getModelsForFamily(family.id, models, generationMode, effectiveSubMode)
      .filter(workflowFilter)
      .filter(m => enabledModels.has(m.model_type))
      .filter(m => !m.nsfw_only || nsfwMode),
  })).filter(g => g.models.length > 0)
  const visibleModels = visibleGroups.flatMap(g => g.models)
  // Filters only appear when they can split the list in this mode.
  const availableSources = SOURCE_FILTERS.filter(f => f.id === 'all' || visibleModels.some(m => matchesSource(m, f.id)))
  const showSourceFilter = availableSources.length > 2
  const showLoraFilter = visibleModels.some(m => !m.loras_disabled) && visibleModels.some(m => m.loras_disabled)
  const activeSource = showSourceFilter ? sourceFilter : 'all'
  const activeLoraOnly = showLoraFilter && loraOnly
  const groups = visibleGroups.map(g => ({
    family: g.family,
    models: g.models
      .filter(m => matchesSource(m, activeSource))
      .filter(m => !activeLoraOnly || !m.loras_disabled),
  })).filter(g => g.models.length > 0)
  const filtersHideEverything = groups.length === 0 && visibleGroups.length > 0

  // How many models are available for this mode but NOT enabled — powers the
  // "+N" hint that nudges users toward Settings → Enabled Models.
  const disabledCount = modeFamilies.reduce((n, family) => {
    const avail = getModelsForFamily(family.id, models, generationMode, effectiveSubMode)
      .filter(workflowFilter)
      .filter(m => !m.nsfw_only || nsfwMode)
    return n + avail.filter(m => !enabledModels.has(m.model_type)).length
  }, 0)

  return (
    <div className={`relative min-w-0 ${placement !== 'below' ? 'flex-1' : ''}`} ref={containerRef}>
      {/* Trigger button */}
      <button
        aria-label="Choose model" aria-expanded={open}
        onClick={() => setOpen(!open)}
        title={currentModelCompatible
          ? `${currentModel?.name || 'Select model'}\n${currentModel?.selector_help || currentModel?.description || ''}`
          : compatibilityDescription}
        className={`w-full flex items-center gap-1.5 bg-bg-tertiary border border-border rounded-lg px-2 py-2 text-left hover:border-border-light transition-colors ${placement === 'footer' ? 'min-h-11' : ''}`}
      >
        <span className={`flex-1 min-w-0 text-text-primary ${placement === 'footer' ? 'line-clamp-2 text-[11px] leading-tight' : 'truncate text-xs'}`}>
          {currentModelCompatible ? (currentModel?.name ?? 'Select model') : 'No compatible model'}
        </span>
        <ChevronDown size={14} className={`shrink-0 text-text-muted transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>

      <ModelChoicesPanel open={open} placement={placement} onClose={() => setOpen(false)}>
          {placement === 'footer' && <button type="button" onClick={() => { setOpen(false); useStore.getState().setLoraBrowserOpen(true, currentModelType) }}
            className="mb-2 flex min-h-10 w-full items-center gap-2 rounded-lg border border-border px-3 text-xs text-text-secondary hover:bg-bg-hover">
            <Globe size={14}/> Browse models, LoRAs & characters
          </button>}
          {/* Enable-more entry — sits above the enabled model list; opens
              Settings → Enabled Models expanded to this mode. */}
          {disabledCount > 0 && (
            <button
              onClick={() => { openModelVisibility(generationMode); setOpen(false) }}
              className="w-full flex items-center gap-2 px-3 py-2 text-left border-b border-border text-text-secondary hover:bg-bg-hover hover:text-accent-blue transition-colors"
            >
              <Plus size={13} className="shrink-0" />
              <span className="flex-1 text-xs">Enable more models</span>
              <span className="text-[10px] text-text-muted shrink-0">{disabledCount} available</span>
            </button>
          )}
          {(showSourceFilter || showLoraFilter) && (
            <div className="flex flex-wrap items-center gap-1 px-3 py-2 border-b border-border" role="group" aria-label="Filter models">
              {showSourceFilter && availableSources.map(f => (
                <FilterChip key={f.id} active={activeSource === f.id} title={f.title} onClick={() => setSourceFilter(f.id)}>
                  {f.label}
                </FilterChip>
              ))}
              {showLoraFilter && (
                <FilterChip active={activeLoraOnly} title="Only models that accept LoRAs" onClick={() => setLoraOnly(!loraOnly)}>
                  LoRA
                </FilterChip>
              )}
            </div>
          )}
          <div className={placement === 'footer' ? 'py-1' : 'max-h-[360px] overflow-y-auto py-1'}>
            {filtersHideEverything && (
              <p className="px-3 py-3 text-[10px] leading-relaxed text-text-muted">
                No enabled model matches these filters.{' '}
                <button type="button" className="text-accent-blue hover:underline"
                  onClick={() => { setSourceFilter('all'); setLoraOnly(false) }}>
                  Clear filters
                </button>
              </p>
            )}
            {groups.length === 0 && !filtersHideEverything && (
              <p className="px-3 py-3 text-[10px] leading-relaxed text-text-muted">
                {compatibilityDescription}
              </p>
            )}
            {groups.map(({ family, models: famModels }) => (
              <div key={family.id}>
                {/* Family header */}
                <div className="px-3 pt-2 pb-1 text-[10px] text-text-muted uppercase tracking-wider font-medium">
                  {family.label}
                </div>
                {/* Models in family */}
                {famModels.map(model => {
                  const isSelected = model.model_type === currentModelType
                  const help = model.selector_help
                  return (
                    <div
                      key={model.model_type}
                      className={`group w-full flex items-center transition-colors ${
                        isSelected
                          ? 'bg-accent-blue/10 text-text-primary'
                          : 'hover:bg-bg-hover text-text-secondary hover:text-text-primary'
                      }`}
                    >
                      <button
                        onClick={() => {
                          if (
                            generationMode === 'video'
                            && ['frames', 'references', 'avatar'].includes(studioVideoWorkflow)
                            && videoImageMode === 0
                          ) selectStudioVideoModel(model.model_type)
                          else selectModel(model.model_type)
                          setOpen(false)
                        }}
                        className="min-w-0 flex-1 px-3 py-1.5 flex items-center gap-2 text-left"
                      >
                        <span className="flex-1 min-w-0 text-xs truncate">{model.name}</span>
                        <ModelBadges model={model} />
                        {isSelected && <Check size={12} className="shrink-0 text-accent-blue" />}
                      </button>
                      {help && (
                        <span className="pr-2">
                          <InfoTooltip
                            text={help}
                            label={`About ${model.name}`}
                          />
                        </span>
                      )}
                    </div>
                  )
                })}
              </div>
            ))}
          </div>
      </ModelChoicesPanel>
    </div>
  )
}

function FilterChip({ active, title, onClick, children }: {
  active: boolean
  title: string
  onClick: () => void
  children: ReactNode
}) {
  return (
    <button
      type="button"
      aria-pressed={active}
      title={title}
      onClick={onClick}
      className={`min-h-6 rounded-full border px-2 text-[10px] leading-none transition-colors ${
        active
          ? 'border-accent-blue bg-accent-blue/15 text-text-primary'
          : 'border-border text-text-muted hover:border-border-light hover:text-text-secondary'
      }`}
    >
      {children}
    </button>
  )
}

function ModelBadges({ model }: {
  model: {
    model_type: string
    cloud_provider?: string | null
    is_i2v: boolean
    is_t2v: boolean
    supports_end_frame?: boolean
    supports_audio?: boolean
    supports_audio_input?: boolean
    generates_audio?: boolean
    supports_ref_images?: boolean
  }
}) {
  const badges: Array<{ label: string; title: string }> = []
  if (model.cloud_provider) {
    const provider = CLOUD_LABELS[model.cloud_provider] ?? model.cloud_provider
    badges.push({ label: provider, title: `Runs on ${provider}; billed per image by the provider` })
  }
  const workflowIsAlreadyInName = model.model_type.startsWith('minimax_h3')
  if (!workflowIsAlreadyInName && model.is_i2v && model.supports_end_frame) {
    badges.push({ label: 'First / Last', title: 'Accepts a first frame, a last frame, or both' })
  } else if (!workflowIsAlreadyInName && model.is_i2v) {
    badges.push({ label: 'I2V', title: 'Accepts an input image' })
  }
  if (model.generates_audio) {
    badges.push({ label: 'Audio Out', title: 'Generates synchronized audio with the video' })
  }
  if (model.supports_audio_input) {
    badges.push({ label: 'Audio In', title: 'Accepts audio input; H3 Omni uses it as an ordered audio reference' })
  }
  if (model.supports_ref_images) {
    badges.push({ label: 'Refs', title: 'Accepts one or more reference images' })
  }
  if (badges.length === 0) return null
  return (
    <span className="flex gap-0.5 shrink-0">
      {badges.map(b => (
        <span
          key={b.label}
          title={b.title}
          className="text-[9px] px-1 py-0.5 rounded bg-bg-tertiary text-text-muted leading-none"
        >
          {b.label}
        </span>
      ))}
    </span>
  )
}
