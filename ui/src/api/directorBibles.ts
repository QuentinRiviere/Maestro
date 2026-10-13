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
}

export interface BibleImage {
  id: string
  file: string | null
  status: 'queued' | 'rendering' | 'candidate' | 'approved' | 'failed'
  error: string
  created_at: string
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
  created_at: string
  updated_at: string
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
export function generateBibleAssets(bibleId: string, imageModel: string, everyLook = false): Promise<DirectorBible> {
  return bibleCall(`/api/v1/director/bibles/${encodeURIComponent(bibleId)}/generate`,
    jsonBody({ image_model: imageModel, every_look: everyLook }), 'Could not start the generation')
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
