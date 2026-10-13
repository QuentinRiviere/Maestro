const BASE = ''

export interface DirectorBibleSummary {
  id: string
  name: string
  description: string
  revision: number
  updated_at: string
  characters: string[]
  locations: string[]
}

export interface DirectorBibleEntry {
  name: string
  role?: string
  descriptor?: string
  description: string
  wardrobe?: string
  image: string | null
}

export interface DirectorBible {
  id: string
  name: string
  description: string
  revision: number
  style: Record<string, unknown> | null
  world: { logline: string; tone: string; rules: string; forbidden: string }
  characters: DirectorBibleEntry[]
  locations: DirectorBibleEntry[]
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

export async function fetchDirectorBibles(): Promise<DirectorBibleSummary[]> {
  return (await json<{ bibles: DirectorBibleSummary[] }>(
    await fetch(`${BASE}/api/v1/director/bibles`), 'Could not load bibles',
  )).bibles
}

/** Save a project's cast and style into a new bible (``name``) or merge it into one (``bible_id``). */
export async function saveCastToBible(pid: string, target: { bible_id?: string; name?: string }): Promise<DirectorBible> {
  return (await json<{ bible: DirectorBible }>(await fetch(
    `${BASE}/api/v1/director/pipeline/${encodeURIComponent(pid)}/bible`,
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(target) },
  ), 'Could not save the cast to a bible')).bible
}

export function directorBibleImageUrl(bibleId: string, image: string): string {
  return `${BASE}/api/v1/director/bibles/${encodeURIComponent(bibleId)}/images/${encodeURIComponent(image)}`
}
