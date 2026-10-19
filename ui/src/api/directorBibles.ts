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
  stats: StyleReferenceStats | null
  described_shots: number
  grammar: StyleGrammar | null
  summary: string
  cost_usd: number | null
  model: string
}

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
