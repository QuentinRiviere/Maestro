const BASE = ''

export interface SkillOption {
  key: string
  label: string
  values: string[]
  default: string
}

export interface DirectorSkillEntry {
  id: string
  name: string
  description: string
  icon: string
  workflow: 'story' | 'audio' | 'both'
  builtin: boolean
  options: SkillOption[]
  default_duration?: number
  version?: number
}

export interface SkillRules {
  goal: string
  structure: string
  pacing: string
  camera: string
  dialogue: string
  sound: string
  do: string[]
  dont: string[]
}

export interface SkillSpec {
  id?: string
  name: string
  description: string
  icon: string
  workflow: 'story' | 'audio' | 'both'
  default_duration: number
  shot_duration: [number, number]
  max_shots: number
  style_hint: string
  rules: SkillRules
  options: SkillOption[]
  references: Array<{ kind: string; name: string; notes: string }>
  based_on?: string | null
  version?: number
  builtin?: boolean
}

export interface ReferenceAnalysis {
  name: string
  kind: string
  notes?: string
  error?: string
}

async function json<T>(res: Response, fallback: string): Promise<T> {
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: fallback }))
    throw new Error(err.detail || fallback)
  }
  return res.json()
}

export async function fetchDirectorSkills(): Promise<DirectorSkillEntry[]> {
  return (await json<{ skills: DirectorSkillEntry[] }>(await fetch(`${BASE}/api/v1/director/skills`), 'Could not load skills')).skills
}

export async function fetchDirectorSkill(id: string): Promise<SkillSpec> {
  return json(await fetch(`${BASE}/api/v1/director/skills/${encodeURIComponent(id)}`), 'Skill not found')
}

export async function draftDirectorSkill(body: {
  description: string
  base_skill_id?: string
  edit_skill_id?: string
  reference_paths: string[]
}): Promise<{ spec: SkillSpec; analyses: ReferenceAnalysis[] }> {
  return json(await fetch(`${BASE}/api/v1/director/skills/draft`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  }), 'Could not draft the skill')
}

export async function saveDirectorSkill(spec: SkillSpec, overwrite: boolean): Promise<SkillSpec> {
  return json(await fetch(`${BASE}/api/v1/director/skills`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ spec, overwrite }),
  }), 'Could not save the skill')
}

export async function deleteDirectorSkill(id: string): Promise<void> {
  await json(await fetch(`${BASE}/api/v1/director/skills/${encodeURIComponent(id)}`, { method: 'DELETE' }), 'Could not delete the skill')
}

export async function testDirectorSkill(spec: SkillSpec, brief: string, options: Record<string, string>): Promise<{
  shots: Array<{ duration_sec: number; goal: string; type: string; actions: string[] }>
  total_duration_sec: number
}> {
  return json(await fetch(`${BASE}/api/v1/director/skills/test`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ spec, brief, options }),
  }), 'Test planning failed')
}
