/** Character voices, song covers (audio then video) and persona clips: the Persona studio of a bible. */
const BASE = ''

async function json<T>(res: Response, fallback: string): Promise<T> {
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: fallback }))
    throw new Error(err.detail || fallback)
  }
  return res.json()
}

function form(fields: Record<string, string | number | boolean | undefined | null>, files: Array<[string, File | null | undefined]> = []): FormData {
  const data = new FormData()
  for (const [key, value] of Object.entries(fields)) {
    if (value === undefined || value === null || value === '') continue
    data.append(key, String(value))
  }
  for (const [key, file] of files) if (file) data.append(key, file)
  return data
}

export type RecordStatus = 'queued' | 'running' | 'done' | 'failed'
export type VoiceMode = 'sing' | 'rap'
export const isLive = (status: string | undefined) => status === 'queued' || status === 'running'

// ── voices ────────────────────────────────────────────────────────────────

export interface VoiceReference { file: string; name: string; weight: number }
export interface VoiceTraining {
  status: RecordStatus
  progress: string
  error: string
  config: { epochs: number; batch_size: number; sample_rate: number }
  dataset: number
  model: string
  index: string
  seconds: number | null
}
export type CoverEngine = 'auto' | 'seedvc' | 'rvc'
export interface CharacterVoice {
  id: string
  name: string
  mode: VoiceMode
  references: VoiceReference[]
  profile: string
  rvc?: VoiceTraining
  covers: number
  bible_id: string
  character_id: string
  created_at: string
  updated_at: string
}

const voicesUrl = (id?: string) => `${BASE}/api/v1/director/voices${id ? `/${encodeURIComponent(id)}` : ''}`

export async function fetchVoices(bibleId?: string): Promise<CharacterVoice[]> {
  const voices = (await json<{ voices: CharacterVoice[] }>(await fetch(voicesUrl()), 'Could not load the voices')).voices
  return bibleId ? voices.filter(v => !v.bible_id || v.bible_id === bibleId) : voices
}

export async function createVoice(request: {
  name: string; references: File[]; weights: number[]; mode: VoiceMode; bible_id?: string; character_id?: string
}): Promise<CharacterVoice> {
  const data = form({ name: request.name, weights: request.weights.join(','), mode: request.mode,
    bible_id: request.bible_id, character_id: request.character_id }, request.references.map(f => ['references', f]))
  return (await json<{ voice: CharacterVoice }>(await fetch(voicesUrl(), { method: 'POST', body: data }),
    'Could not create the voice')).voice
}

export async function deleteVoice(id: string): Promise<CharacterVoice[]> {
  return (await json<{ voices: CharacterVoice[] }>(await fetch(voicesUrl(id), { method: 'DELETE' }),
    'Could not delete the voice')).voices
}

export async function trainVoice(id: string, request: { epochs?: number; batch_size?: number; sample_rate?: number } = {}): Promise<CharacterVoice> {
  return (await json<{ voice: CharacterVoice }>(await fetch(`${voicesUrl(id)}/train`,
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(request) }), 'Could not start the training')).voice
}

export async function cancelVoiceTraining(id: string): Promise<CharacterVoice> {
  return (await json<{ voice: CharacterVoice }>(await fetch(`${voicesUrl(id)}/train/cancel`, { method: 'POST' }), 'Could not stop the training')).voice
}

export const voiceReferenceUrl = (id: string, index: number) => `${voicesUrl(id)}/audio/${index}`

// ── covers ────────────────────────────────────────────────────────────────

export interface CoverVideoSettings {
  character_id: string
  location_id: string
  orientation: 'portrait' | 'landscape'
  start_sec: number
  frames: number
  seconds: number
  seed: number
  model_type: string
  lyrics: string
}
export interface CoverVideo {
  status: RecordStatus
  progress: string
  error: string
  settings: CoverVideoSettings
  outputs: Partial<Record<'start' | 'video' | 'clip_mix' | 'clip_vocals', string>>
  job_id: string
  seconds: number | null
  prompt?: string
}
export interface CoverScores {
  timbre: number
  timbre_source: number
  pitch_corr: number
  pitch_cents: number | null
  voiced_agreement: number
  singmos?: number
  utmos22?: number
}
export interface SongCover {
  id: string
  name: string
  status: RecordStatus
  progress: string
  error: string
  config: { mode: VoiceMode; engine: 'seedvc' | 'rvc'; steps: number; cfg_rate: number; semitone_shift: number; auto_f0: boolean; vocal_gain_db: number }
  song: string
  reference: string
  voice_id: string
  bible_id: string
  character_id: string
  outputs: Partial<Record<'cover' | 'vocals' | 'source_vocals' | 'instrumental' | 'profile' | 'scores', string>>
  job_id: string
  seconds: number | null
  /** The worker's judge: timbre match to the voice (0-1), pitch fidelity to the source, a MOS when available. */
  scores?: CoverScores | null
  video?: CoverVideo
  created_at: string
  updated_at: string
}

const coversUrl = (id?: string) => `${BASE}/api/v1/director/covers${id ? `/${encodeURIComponent(id)}` : ''}`

export async function fetchCovers(bibleId?: string): Promise<SongCover[]> {
  const covers = (await json<{ covers: SongCover[] }>(await fetch(coversUrl()), 'Could not load the covers')).covers
  return bibleId ? covers.filter(c => !c.bible_id || c.bible_id === bibleId) : covers
}

export interface CoverRequest {
  name: string
  song?: File | null
  /** A song rendered by the generate-music route, by its file name in the outputs folder. */
  song_output?: string
  workspace?: string
  voice_id: string
  mode?: VoiceMode | ''
  engine?: CoverEngine
  steps?: number
  cfg_rate?: number
  semitone_shift?: number
  auto_f0?: boolean
  vocal_gain_db?: number
  bible_id?: string
  character_id?: string
}

export async function createCover(request: CoverRequest): Promise<SongCover> {
  const { song, ...fields } = request
  const data = form({ ...fields, run: true }, [['song', song]])
  return (await json<{ cover: SongCover }>(await fetch(coversUrl(), { method: 'POST', body: data }),
    'Could not start the cover')).cover
}

export async function runCover(id: string): Promise<SongCover> {
  return (await json<{ cover: SongCover }>(await fetch(`${coversUrl(id)}/run`, { method: 'POST' }), 'Could not start the cover')).cover
}

export async function cancelCover(id: string): Promise<SongCover> {
  return (await json<{ cover: SongCover }>(await fetch(`${coversUrl(id)}/cancel`, { method: 'POST' }), 'Could not stop the cover')).cover
}

export async function deleteCover(id: string): Promise<SongCover[]> {
  return (await json<{ covers: SongCover[] }>(await fetch(coversUrl(id), { method: 'DELETE' }), 'Could not delete the cover')).covers
}

export const coverAudioUrl = (id: string, which: 'cover' | 'vocals' | 'source_vocals' | 'instrumental' | 'song' | 'reference') =>
  `${coversUrl(id)}/audio/${which}`

export interface CoverVideoRequest {
  character_id?: string
  location_id?: string
  orientation?: 'portrait' | 'landscape'
  start_sec?: number
  seconds?: number
  seed?: number
  lyrics?: string
  prompt?: string
  start_image?: File | null
}

export async function startCoverVideo(id: string, request: CoverVideoRequest): Promise<SongCover> {
  const { start_image, ...fields } = request
  const data = form(fields, [['start_image', start_image]])
  return (await json<{ cover: SongCover }>(await fetch(`${coversUrl(id)}/video`, { method: 'POST', body: data }),
    'Could not start the video')).cover
}

export async function cancelCoverVideo(id: string): Promise<SongCover> {
  return (await json<{ cover: SongCover }>(await fetch(`${coversUrl(id)}/video/cancel`, { method: 'POST' }),
    'Could not stop the video')).cover
}

export const coverVideoUrl = (id: string, which: 'video' | 'start' | 'clip_mix' | 'clip_vocals') => `${coversUrl(id)}/video/${which}`

// ── persona clips ─────────────────────────────────────────────────────────

export interface ClipEvaluation {
  identity?: number | null
  move?: number | null
  place?: number | null
  artefacts?: number | null
  hook?: number | null
  overall?: number | null
  verdict?: string
  fix?: string
  model?: string
  evaluated_at?: string
  error?: string
}
export interface PersonaClipSettings {
  persona_id?: string
  character_id: string
  location_id: string
  gesture: string
  place: string
  orientation: 'portrait' | 'landscape'
  frames: number
  seconds: number
  /** Rendered ``speed`` times longer and sped up in post (LTX moves at about two thirds of real pace). */
  speed?: number
  rendered_seconds?: number
  /** Shots cut together: single (one full-body take), trend (wide, side, close), street, reveal. */
  template?: ClipTemplate
  shots?: Array<{ framing: string; seconds: number; frames: number }>
  /** ltx (LTX-2.5, default) or h3 (MiniMax H3 image-to-video, 480p on the Spark). */
  engine?: ClipEngine
  seed: number
  model_type: string
  audio: 'silent' | 'model'
}
export type ClipTemplate = 'single' | 'trend' | 'street' | 'reveal'
export type ClipEngine = 'ltx' | 'h3'
export interface PersonaClip {
  id: string
  name: string
  status: RecordStatus
  progress: string
  error: string
  settings: PersonaClipSettings
  bible_id: string
  outputs: Partial<Record<'start' | 'clip', string>>
  job_id: string
  prompt: string
  seconds: number | null
  evaluation?: ClipEvaluation | null
  created_at: string
  updated_at: string
}

const clipsUrl = (id?: string) => `${BASE}/api/v1/director/persona-clips${id ? `/${encodeURIComponent(id)}` : ''}`

export async function fetchPersonaClips(bibleId: string): Promise<PersonaClip[]> {
  return (await json<{ clips: PersonaClip[] }>(await fetch(`${clipsUrl()}?bible_id=${encodeURIComponent(bibleId)}`),
    'Could not load the clips')).clips
}

// ── personas ──────────────────────────────────────────────────────────────

export type PersonaFormat = 'silent_move' | 'diary'
export interface Persona {
  id: string
  name: string
  handle: string
  look: string
  /** Must-have visual traits (face, props, garments) the editor keeps and the judge checks, derived from the look. */
  traits: string[]
  move: string
  format: PersonaFormat
  world: string
  music: string
  places: string[]
  orientation: 'portrait' | 'landscape'
  seconds: number
  speed: number
  template: ClipTemplate
  engine?: ClipEngine
  audio: 'silent' | 'model'
  bible_id: string
  character_id: string
  status: RecordStatus
  progress: string
  error: string
  portrait: string
  clips: number
  created_at: string
  updated_at: string
}

const personasUrl = (id?: string) => `${BASE}/api/v1/director/personas${id ? `/${encodeURIComponent(id)}` : ''}`

export async function fetchPersonas(): Promise<{ personas: Persona[]; formats: Record<string, string> }> {
  return json(await fetch(personasUrl()), 'Could not load the personas')
}

export async function fetchPersona(id: string): Promise<{ persona: Persona; clips: PersonaClip[] }> {
  return json(await fetch(personasUrl(id)), 'Could not load the persona')
}

export interface PersonaRequest {
  name: string
  handle?: string
  look?: string
  traits?: string
  move: string
  format?: PersonaFormat
  world?: string
  music?: string
  places?: string
  orientation?: 'portrait' | 'landscape'
  seconds?: number
  speed?: number
  template?: ClipTemplate
  engine?: ClipEngine
  audio?: 'silent' | 'model'
  bible_id?: string
  character_id?: string
  portrait?: File | null
}

export async function createPersona(request: PersonaRequest): Promise<Persona> {
  const { portrait, ...fields } = request
  return (await json<{ persona: Persona }>(await fetch(personasUrl(), { method: 'POST', body: form(fields, [['portrait', portrait]]) }),
    'Could not create the persona')).persona
}

export async function updatePersona(id: string, fields: Partial<Omit<PersonaRequest, 'portrait'>>): Promise<Persona> {
  return (await json<{ persona: Persona }>(await fetch(personasUrl(id),
    { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(fields) }), 'Could not update the persona')).persona
}

export async function repaintPersona(id: string): Promise<Persona> {
  return (await json<{ persona: Persona }>(await fetch(`${personasUrl(id)}/portrait`, { method: 'POST' }), 'Could not repaint the portrait')).persona
}

/** Queue one clip per place: the given ones, or the persona's own list (``count`` of them). */
export async function startPersonaSeries(id: string, request: { places?: string[]; count?: number; seconds?: number; template?: ClipTemplate; engine?: ClipEngine }): Promise<PersonaClip[]> {
  return (await json<{ clips: PersonaClip[] }>(await fetch(`${personasUrl(id)}/series`,
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(request) }), 'Could not start the series')).clips
}

/** The analyst scores every finished, unjudged clip (identity, move, place, artefacts, hook). */
export async function evaluatePersona(id: string): Promise<PersonaClip[]> {
  return (await json<{ clips: PersonaClip[] }>(await fetch(`${personasUrl(id)}/evaluate`, { method: 'POST' }), 'Could not evaluate the clips')).clips
}

export async function addPersonaToBible(id: string, bibleId: string): Promise<Persona> {
  return (await json<{ persona: Persona }>(await fetch(`${personasUrl(id)}/bible`,
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ bible_id: bibleId }) }), 'Could not add the persona to the bible')).persona
}

export async function deletePersona(id: string): Promise<Persona[]> {
  return (await json<{ personas: Persona[] }>(await fetch(personasUrl(id), { method: 'DELETE' }), 'Could not delete the persona')).personas
}

export const personaPortraitUrl = (id: string) => `${personasUrl(id)}/portrait`

export interface PersonaClipRequest {
  bible_id?: string
  character_id?: string
  persona_id?: string
  gesture: string
  location_id?: string
  place?: string
  name?: string
  orientation?: 'portrait' | 'landscape'
  seconds?: number
  speed?: number
  template?: ClipTemplate
  engine?: ClipEngine
  seed?: number
  audio?: 'silent' | 'model'
  prompt?: string
  start_image?: File | null
}

export async function createPersonaClip(request: PersonaClipRequest): Promise<PersonaClip> {
  const { start_image, ...fields } = request
  const data = form({ ...fields, run: true }, [['start_image', start_image]])
  return (await json<{ clip: PersonaClip }>(await fetch(clipsUrl(), { method: 'POST', body: data }),
    'Could not start the clip')).clip
}

export async function runPersonaClip(id: string): Promise<PersonaClip> {
  return (await json<{ clip: PersonaClip }>(await fetch(`${clipsUrl(id)}/run`, { method: 'POST' }), 'Could not start the clip')).clip
}

export async function cancelPersonaClip(id: string): Promise<PersonaClip> {
  return (await json<{ clip: PersonaClip }>(await fetch(`${clipsUrl(id)}/cancel`, { method: 'POST' }), 'Could not stop the clip')).clip
}

export async function deletePersonaClip(id: string): Promise<PersonaClip[]> {
  return (await json<{ clips: PersonaClip[] }>(await fetch(clipsUrl(id), { method: 'DELETE' }), 'Could not delete the clip')).clips
}

export const personaClipFileUrl = (id: string, which: 'clip' | 'start') => `${clipsUrl(id)}/file/${which}`
