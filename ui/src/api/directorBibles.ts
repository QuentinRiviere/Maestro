const BASE = ''

export type BibleKind = 'character' | 'location' | 'prop'
export const BIBLE_PLURAL: Record<BibleKind, 'characters' | 'locations' | 'props'> = {
  character: 'characters', location: 'locations', prop: 'props',
}

export interface DirectorBibleSummary {
  id: string
  name: string
  description: string
  revision: number
  updated_at: string
  characters: string[]
  locations: string[]
  props: string[]
  /** How many reference videos the bible keeps. */
  style_references?: number
}

export interface BibleImage {
  id: string
  file: string | null
  status: 'queued' | 'rendering' | 'candidate' | 'approved' | 'failed'
  error: string
  created_at: string
  /** An angle of the look (a character: face_front, face_three_quarter, face_profile, back; a place: reverse); empty for the look's own image. */
  view?: string
}

export interface BibleVariant {
  id: string
  label: string
  description: string
  images: BibleImage[]
}

export interface BibleElement {
  id: string
  kind: BibleKind
  name: string
  tag: string
  role?: string
  descriptor?: string
  description: string
  wardrobe?: string
  personality?: string
  arc?: string
  notes?: string
  voice_character_id?: string
  /** A location's rough 3D layout for the Director blockout guide (see docs/Director-blockout.md). */
  layout?: Record<string, unknown> | null
  variants: BibleVariant[]
  image: string | null
}

export interface BibleWorld {
  logline: string
  story: string
  stakes: string
  twists: string
  tone: string
  narrative_style: string
  look: string
  rules: string
  forbidden: string
}

export interface DirectorBible {
  id: string
  name: string
  description: string
  revision: number
  style: Record<string, unknown> | null
  world: BibleWorld
  characters: BibleElement[]
  locations: BibleElement[]
  props: BibleElement[]
  /** Reference videos (staging, editing, rhythm); edited through their own endpoints, never by a bible save. */
  style_references?: StyleReference[]
  created_at: string
  updated_at: string
}

export type StyleReferenceStatus = 'queued' | 'analysing' | 'done' | 'failed'
/** Whether the analyst reads captions, logos and other on-screen graphics as part of the staging. */
export type StyleReferenceGraphics = 'ignore' | 'include'

export interface StyleReferenceStats {
  duration_sec: number
  fps?: number | null
  width?: number
  height?: number
  has_audio?: boolean
  shot_count: number
  cut_count?: number
  asl_sec: number | null
  median_sec: number | null
  min_sec?: number | null
  max_sec?: number | null
  cuts_per_minute: number | null
  distribution?: { bucket: string; count: number; percent: number }[]
  /** Average shot length per 10 s window. */
  pacing?: { start: number; end: number; asl_sec: number; cuts: number }[]
  /** Percent of the described shots per shot size, largest first. */
  shot_mix?: Record<string, number>
  described_shots?: number
}

export interface StyleGrammar {
  rhythm: { target_asl_sec: number | null; pace_with_emotion: string; notes: string }
  shot_mix: Record<string, number>
  patterns: string[]
  scene_openings: string
  scene_closings: string
  dialogue_coverage: string
  camera_movement: string
  lighting_color: string
  music_silence: string
  rules: string[]
  rules_edited?: boolean
}

export interface StyleReference {
  id: string
  label: string
  file: string
  source_filename: string
  created_at: string
  updated_at: string
  status: StyleReferenceStatus
  error: string
  progress: string
  enabled: boolean
  scene_tags: string[]
  graphics: StyleReferenceGraphics
  /** The analysis named the reference / chose its tags (until the user types their own). */
  label_auto?: boolean
  tags_auto?: boolean
  frames_status?: StyleFramesStatus
  frames_count?: number
  frames_progress?: string
  frames_error?: string
  camera_profile?: StyleCameraProfile | null
  stats: StyleReferenceStats | null
  described_shots: number
  grammar: StyleGrammar | null
  summary: string
  cost_usd: number | null
  model: string
}

/** One shot of a reference's breakdown (the sidecar the analysis fills excerpt by excerpt). */
export interface StyleReferenceShot {
  index: number
  start: number
  end: number
  duration: number
  shot_size?: string
  angle?: string
  movement?: string
  subject?: string
  action?: string
  cut_reason?: string
  transition?: string
  lighting?: string
  audio?: string
  graphics?: string
  false_cut?: boolean
  credits?: boolean
}

/** A frame of a shot written as the image prompt that would regenerate it. */
export interface StyleReferenceFrame {
  index: number
  position: 'first' | 'last'
  t: number
  prompt: string
  shot_size?: string
  measured_shot_size?: string
  lighting?: string
  palette?: string[]
  mood?: string
  camera?: { lens_mm?: number | null; aperture?: string; format?: string; height?: string; angle?: string; look?: string }
  lighting_setup?: string
  grade?: string
  aspect_ratio?: string
}

export interface StyleCameraProfile {
  frames: number
  lens_mm_median?: number | null
  lenses_mm?: number[]
  aperture?: string
  format?: string
  height?: string
  look?: string
  grade?: string
  lighting_setup?: string
  aspect_ratio?: string
}

export type StyleFramesStatus = 'idle' | 'queued' | 'running' | 'done' | 'failed'

export interface StyleReferencePatch {
  label?: string
  enabled?: boolean
  scene_tags?: string[]
  graphics?: StyleReferenceGraphics
  rules?: string[]
}

async function json<T>(res: Response, fallback: string): Promise<T> {
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: fallback }))
    throw new Error(err.detail || fallback)
  }
  return res.json()
}

async function bibleCall(url: string, init: RequestInit, fallback: string): Promise<DirectorBible> {
  return (await json<{ bible: DirectorBible }>(await fetch(`${BASE}${url}`, init), fallback)).bible
}

const jsonBody = (body: unknown): RequestInit => ({
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
})

export async function fetchDirectorBibles(): Promise<DirectorBibleSummary[]> {
  return (await json<{ bibles: DirectorBibleSummary[] }>(
    await fetch(`${BASE}/api/v1/director/bibles`), 'Could not load bibles',
  )).bibles
}

export function fetchDirectorBible(id: string): Promise<DirectorBible> {
  return bibleCall(`/api/v1/director/bibles/${encodeURIComponent(id)}`, {}, 'Bible not found')
}

export function createDirectorBible(name: string): Promise<DirectorBible> {
  return bibleCall('/api/v1/director/bibles', jsonBody({ name }), 'Could not create the bible')
}

/** Save the text of a bible; images are kept server-side. */
export function updateDirectorBible(bible: DirectorBible): Promise<DirectorBible> {
  return bibleCall(`/api/v1/director/bibles/${encodeURIComponent(bible.id)}`,
    { ...jsonBody(bible), method: 'PUT' }, 'Could not save the bible')
}

export async function deleteDirectorBible(id: string): Promise<void> {
  await json(await fetch(`${BASE}/api/v1/director/bibles/${encodeURIComponent(id)}`, { method: 'DELETE' }),
    'Could not delete the bible')
}

const variantUrl = (bibleId: string, elementId: string, variantId: string) =>
  `/api/v1/director/bibles/${encodeURIComponent(bibleId)}/elements/${encodeURIComponent(elementId)}`
  + `/variants/${encodeURIComponent(variantId)}`

/** Render a candidate image of one look; it renders in the background. */
export function renderBibleImage(bibleId: string, elementId: string, variantId: string,
  body: { image_model?: string; chain_from?: string[] }): Promise<DirectorBible> {
  return bibleCall(`${variantUrl(bibleId, elementId, variantId)}/render`, jsonBody(body), 'Could not start the render')
}

/** Render the angles (``views``: some of them) of a look, edited from its approved image, in the background: a character's four, a place's reverse. */
export function renderBibleViews(bibleId: string, elementId: string, variantId: string, imageModel: string,
  views?: string[]): Promise<DirectorBible> {
  return bibleCall(`${variantUrl(bibleId, elementId, variantId)}/views`, jsonBody({ image_model: imageModel, views }),
    'Could not render the angles')
}

export function uploadBibleImage(bibleId: string, elementId: string, variantId: string, file: File): Promise<DirectorBible> {
  const form = new FormData()
  form.append('file', file)
  return bibleCall(`${variantUrl(bibleId, elementId, variantId)}/upload`, { method: 'POST', body: form },
    'Could not upload the image')
}

export function approveBibleImage(bibleId: string, elementId: string, variantId: string, imageId: string): Promise<DirectorBible> {
  return bibleCall(`${variantUrl(bibleId, elementId, variantId)}/images/${encodeURIComponent(imageId)}/approve`,
    { method: 'POST' }, 'Could not approve the image')
}

/** Turn an approved image back into a candidate. */
export function unapproveBibleImage(bibleId: string, elementId: string, variantId: string, imageId: string): Promise<DirectorBible> {
  return bibleCall(`${variantUrl(bibleId, elementId, variantId)}/images/${encodeURIComponent(imageId)}/unapprove`,
    { method: 'POST' }, 'Could not unapprove the image')
}

/** Render a failed image (a look or one of its angles) again, in the background. */
export function retryBibleImage(bibleId: string, elementId: string, variantId: string, imageId: string,
  imageModel: string): Promise<DirectorBible> {
  return bibleCall(`${variantUrl(bibleId, elementId, variantId)}/images/${encodeURIComponent(imageId)}/retry`,
    jsonBody({ image_model: imageModel }), 'Could not retry the render')
}

export function deleteBibleImage(bibleId: string, elementId: string, variantId: string, imageId: string): Promise<DirectorBible> {
  return bibleCall(`${variantUrl(bibleId, elementId, variantId)}/images/${encodeURIComponent(imageId)}`,
    { method: 'DELETE' }, 'Could not delete the image')
}

/** Save a project's cast and style into a new bible (``name``) or merge it into one (``bible_id``). */
export function saveCastToBible(pid: string, target: { bible_id?: string; name?: string }): Promise<DirectorBible> {
  return bibleCall(`/api/v1/director/pipeline/${encodeURIComponent(pid)}/bible`, jsonBody(target),
    'Could not save the cast to a bible')
}

export function directorBibleImageUrl(bibleId: string, image: string): string {
  return `${BASE}/api/v1/director/bibles/${encodeURIComponent(bibleId)}/images/${encodeURIComponent(image)}`
}

export interface BibleChatMessage { role: 'user' | 'assistant'; content: string }

export interface BibleAssistantAnswer {
  reply: string
  changes: string[]
  /** The proposed bible (not saved): it goes to the draft, saved by Save or Generate. */
  bible: DirectorBible
  /** True once the bible can be rendered; ``missing`` says what is still needed otherwise. */
  ready: boolean
  missing: string[]
}

/** Render every look that has no approved image yet (``everyLook``: a new candidate for every look), in the background. */
/** Render the bible's looks (``elementId``: one element's looks); ``everyLook`` adds a candidate to approved ones too. */
export function generateBibleAssets(bibleId: string, imageModel: string, everyLook = false,
  elementId = ''): Promise<DirectorBible> {
  return bibleCall(`/api/v1/director/bibles/${encodeURIComponent(bibleId)}/generate`,
    jsonBody({ image_model: imageModel, every_look: everyLook, element_id: elementId }), 'Could not start the generation')
}

/** One assistant turn on the bible being edited. */
export async function askBibleAssistant(bibleId: string, draft: DirectorBible,
  messages: BibleChatMessage[]): Promise<BibleAssistantAnswer> {
  return json(await fetch(`${BASE}/api/v1/director/bibles/${encodeURIComponent(bibleId)}/assistant`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ bible: draft, messages }),
  }), 'The assistant did not answer')
}

export interface BibleRender {
  bible_id: string
  bible_name: string
  kind: BibleKind
  element: string
  look: string
  slot: string
  status: 'queued' | 'rendering'
  created_at: string
}

/** Queued and running bible renders, for the generation queue. */
export async function fetchBibleRenders(): Promise<BibleRender[]> {
  return (await json<{ renders: BibleRender[] }>(await fetch(`${BASE}/api/v1/director/bibles/renders`),
    'Could not load bible renders')).renders
}

// ── Style references: reference videos analysed into a staging grammar ──

const styleRefUrl = (bibleId: string, refId = '') =>
  `${BASE}/api/v1/director/bibles/${encodeURIComponent(bibleId)}/style-references`
  + (refId ? `/${encodeURIComponent(refId)}` : '')

export const STYLE_REFERENCE_MAX_BYTES = 4 * 1024 * 1024 * 1024
export const STYLE_REFERENCE_ACCEPT = '.mp4,.mov,.webm,.mkv,video/mp4,video/quicktime,video/webm,video/x-matroska'

/** One reference with its shots (described ones fill in while the analysis runs). */
export async function fetchStyleReference(bibleId: string, refId: string):
  Promise<StyleReference & { shots: StyleReferenceShot[]; frames: StyleReferenceFrame[] }> {
  return (await json<{ reference: StyleReference & { shots: StyleReferenceShot[]; frames: StyleReferenceFrame[] } }>(
    await fetch(styleRefUrl(bibleId, refId)), 'Could not load the reference')).reference
}

/** Start (again) the reverse-prompt pass over the shots' first and last frames. */
export async function startStyleReferenceFrames(bibleId: string, refId: string): Promise<StyleReference> {
  return (await json<{ reference: StyleReference }>(
    await fetch(`${styleRefUrl(bibleId, refId)}/frames`, { method: 'POST' }),
    'Could not start the frame prompts')).reference
}

export const styleReferenceStillUrl = (bibleId: string, refId: string, seconds: number) =>
  `${styleRefUrl(bibleId, refId)}/still?t=${Math.max(0, seconds).toFixed(1)}`

export async function fetchStyleReferences(bibleId: string): Promise<StyleReference[]> {
  return (await json<{ references: StyleReference[] }>(await fetch(styleRefUrl(bibleId)),
    'Could not load the style references')).references
}

/** Upload a reference video; its analysis starts in the background. */
export async function uploadStyleReference(bibleId: string, file: File, label: string,
  sceneTags: string[], graphics: StyleReferenceGraphics = 'ignore'): Promise<StyleReference> {
  const form = new FormData()
  form.append('file', file)
  form.append('label', label)
  form.append('scene_tags', sceneTags.join(','))
  form.append('graphics', graphics)
  return (await json<{ reference: StyleReference }>(await fetch(styleRefUrl(bibleId), { method: 'POST', body: form }),
    'Could not upload the video')).reference
}

export type StyleReferenceFetchStatus = 'queued' | 'downloading' | 'done' | 'failed'

/** A link being turned into a reference by the video fetch service (docs/Spark-video-fetch.md). */
export interface StyleReferenceFetch {
  id: string
  url: string
  status: StyleReferenceFetchStatus
  progress: string
  error: string
  reference_id: string
  label: string
  created_at: string
  updated_at: string
}

/** Queue links (one per line): each is downloaded at 480p on the fetch service, then analysed like an upload.
 *  The label applies only when there is a single link; the others take the video's title. */
export async function fetchStyleReferencesFromUrls(bibleId: string, urls: string[], label: string,
  sceneTags: string[], graphics: StyleReferenceGraphics = 'ignore'): Promise<StyleReferenceFetch[]> {
  return (await json<{ fetches: StyleReferenceFetch[] }>(await fetch(`${styleRefUrl(bibleId)}/from-url`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ urls, label, scene_tags: sceneTags, graphics }),
  }), 'Could not fetch the links')).fetches
}

export const parseVideoLinks = (text: string): string[] =>
  Array.from(new Set(text.split(/[\s,]+/).map(s => s.trim()).filter(s => /^https?:\/\/\S+$/i.test(s))))

export async function fetchStyleReferenceFetches(bibleId: string): Promise<StyleReferenceFetch[]> {
  return (await json<{ fetches: StyleReferenceFetch[] }>(await fetch(`${styleRefUrl(bibleId)}/fetches`),
    'Could not load the fetches')).fetches
}

export async function updateStyleReference(bibleId: string, refId: string,
  patch: StyleReferencePatch): Promise<StyleReference> {
  return (await json<{ reference: StyleReference }>(await fetch(styleRefUrl(bibleId, refId), {
    method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(patch),
  }), 'Could not save the reference')).reference
}

export async function analyseStyleReference(bibleId: string, refId: string): Promise<StyleReference> {
  return (await json<{ reference: StyleReference }>(
    await fetch(`${styleRefUrl(bibleId, refId)}/analyse`, { method: 'POST' }),
    'Could not start the analysis')).reference
}

export async function deleteStyleReference(bibleId: string, refId: string): Promise<StyleReference[]> {
  return (await json<{ references: StyleReference[] }>(
    await fetch(styleRefUrl(bibleId, refId), { method: 'DELETE' }), 'Could not delete the reference')).references
}

export function styleReferenceVideoUrl(bibleId: string, refId: string): string {
  return `${styleRefUrl(bibleId, refId)}/video`
}

// ── Style LoRAs: adapters trained on the worker from the references' frames ──

export type StyleLoraStatus = 'queued' | 'preparing' | 'training' | 'sampling' | 'done' | 'failed'

export interface StyleLoraConfig {
  steps: number
  rank: number
  learning_rate: number
  resolution: number
  exclude_close_ups: boolean
  caption_mode: 'content' | 'full'
  targets: 'attention' | 'all'
  max_images: number
  trigger: string
  references: string[]
}

export interface StyleLoraDataset {
  images: number
  references: { id: string; label: string; frames: number; used: number }[]
  excluded_close_ups: number
  excluded_credits: number
  by_shot_size?: Record<string, number>
}

export interface StyleLora {
  id: string
  name: string
  file: string
  trigger: string
  status: StyleLoraStatus
  progress: string
  step: number
  total_steps: number
  error: string
  enabled: boolean
  scale: number
  config: Partial<StyleLoraConfig>
  dataset: StyleLoraDataset | null
  samples: string[]
  job_id: string
  model: string
  training: { final_loss?: number; seconds?: number; items?: number; losses?: number[] } | null
  created_at: string
  updated_at: string
}

/** What a training would use, before anything is extracted. */
export interface StyleLoraDatasetPreview {
  available: number
  minimum: number
  trigger: string
  references: { id: string; label: string; frames: number; used: number }[]
  excluded_close_ups: number
  excluded_credits: number
  captions: string[]
}

export interface StyleLoraRequest {
  name?: string
  trigger?: string
  steps?: number
  rank?: number
  learning_rate?: number
  resolution?: number
  exclude_close_ups?: boolean
  caption_mode?: 'content' | 'full'
  targets?: 'attention' | 'all'
  references?: string[]
  train?: boolean
}

const styleLoraUrl = (bibleId: string, loraId = '') =>
  `${BASE}/api/v1/director/bibles/${encodeURIComponent(bibleId)}/style-loras`
  + (loraId ? `/${encodeURIComponent(loraId)}` : '')

export async function fetchStyleLoras(bibleId: string): Promise<StyleLora[]> {
  return (await json<{ loras: StyleLora[] }>(await fetch(styleLoraUrl(bibleId)),
    'Could not load the style LoRAs')).loras
}

export async function fetchStyleLoraDataset(bibleId: string, options: { exclude_close_ups?: boolean; caption_mode?: string; trigger?: string } = {}):
  Promise<StyleLoraDatasetPreview> {
  const query = new URLSearchParams()
  if (options.exclude_close_ups !== undefined) query.set('exclude_close_ups', String(options.exclude_close_ups))
  if (options.caption_mode) query.set('caption_mode', options.caption_mode)
  if (options.trigger) query.set('trigger', options.trigger)
  return json<StyleLoraDatasetPreview>(await fetch(`${styleLoraUrl(bibleId)}/dataset?${query}`),
    'Could not read the frames')
}

/** Create a style LoRA record and start its training (``train: false`` only creates it). */
export async function createStyleLora(bibleId: string, body: StyleLoraRequest): Promise<StyleLora> {
  return (await json<{ lora: StyleLora }>(await fetch(styleLoraUrl(bibleId), {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  }), 'Could not start the training')).lora
}

export async function updateStyleLora(bibleId: string, loraId: string, body: { enabled?: boolean; scale?: number; name?: string }):
  Promise<StyleLora> {
  return (await json<{ lora: StyleLora }>(await fetch(styleLoraUrl(bibleId, loraId), {
    method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  }), 'Could not update the LoRA')).lora
}

export async function deleteStyleLora(bibleId: string, loraId: string): Promise<StyleLora[]> {
  return (await json<{ loras: StyleLora[] }>(await fetch(styleLoraUrl(bibleId, loraId), { method: 'DELETE' }),
    'Could not delete the LoRA')).loras
}

export async function trainStyleLora(bibleId: string, loraId: string): Promise<StyleLora> {
  return (await json<{ lora: StyleLora }>(await fetch(`${styleLoraUrl(bibleId, loraId)}/train`, { method: 'POST' }),
    'Could not start the training')).lora
}

export async function cancelStyleLora(bibleId: string, loraId: string): Promise<StyleLora> {
  return (await json<{ lora: StyleLora }>(await fetch(`${styleLoraUrl(bibleId, loraId)}/cancel`, { method: 'POST' }),
    'Could not cancel the training')).lora
}

/** Render the sample images again at the LoRA's current weight. */
export async function resampleStyleLora(bibleId: string, loraId: string): Promise<StyleLora> {
  return (await json<{ lora: StyleLora }>(await fetch(`${styleLoraUrl(bibleId, loraId)}/samples`, { method: 'POST' }),
    'Could not render the samples')).lora
}

export const styleLoraSampleUrl = (bibleId: string, loraId: string, number: number) =>
  `${styleLoraUrl(bibleId, loraId)}/samples/${number}`
