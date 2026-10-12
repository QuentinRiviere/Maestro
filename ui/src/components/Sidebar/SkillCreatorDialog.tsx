import { useEffect, useState } from 'react'
import { Loader2, Paperclip, Trash2, X } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import { uploadImage } from '../../api/client'
import {
  deleteDirectorSkill, draftDirectorSkill, fetchDirectorSkill, saveDirectorSkill, testDirectorSkill,
  type ReferenceAnalysis, type SkillOption, type SkillSpec,
} from '../../api/directorSkills'
import { SKILL_ICONS } from '../../lib/skillIcons'

const RULES: Array<[keyof SkillSpec['rules'], string]> = [
  ['goal', 'Purpose'], ['structure', 'Structure'], ['pacing', 'Pacing'],
  ['camera', 'Camera'], ['dialogue', 'Dialogue'], ['sound', 'Sound'],
]
const input = 'w-full bg-bg-tertiary border border-border rounded-lg px-2.5 py-1.5 text-xs text-text-primary focus:outline-none focus:border-accent-blue'
const label = 'text-[10px] text-text-muted uppercase tracking-wider block mb-1'

interface Reference { name: string; path: string; kind: string }

function kindOf(file: File) {
  if (file.type.startsWith('image/')) return 'image'
  if (file.type.startsWith('audio/')) return 'audio'
  return 'video'
}

function OptionsEditor({ options, onChange }: { options: SkillOption[]; onChange: (options: SkillOption[]) => void }) {
  const update = (index: number, patch: Partial<SkillOption>) =>
    onChange(options.map((option, i) => (i === index ? { ...option, ...patch } : option)))
  return (
    <div className="space-y-1.5">
      {options.map((option, index) => (
        <div key={index} className="flex gap-1.5">
          <input className={`${input} w-28`} value={option.label} placeholder="Label"
            onChange={e => update(index, { label: e.target.value, key: e.target.value.toLowerCase().replace(/[^a-z0-9]+/g, '_') })} />
          <input className={input} value={option.values.join(', ')} placeholder="value, value, value"
            onChange={e => update(index, { values: e.target.value.split(',').map(v => v.trim()).filter(Boolean) })} />
          <button onClick={() => onChange(options.filter((_, i) => i !== index))} aria-label="Remove option"
            className="px-1.5 text-text-muted hover:text-chip-red"><X size={12} /></button>
        </div>
      ))}
      {options.length < 4 && (
        <button onClick={() => onChange([...options, { key: '', label: '', values: [], default: '' }])}
          className="text-[10px] text-accent-blue hover:underline">+ Add an option</button>
      )}
    </div>
  )
}

/** Create or edit a declarative Director skill. */
export function SkillCreatorDialog({ editId, onClose, onSaved }: {
  editId?: string
  onClose: () => void
  onSaved: () => void
}) {
  const skills = useStore(s => s.directorSkills)
  const [mode, setMode] = useState<string>(editId ? `edit:${editId}` : 'new')
  const [baseId, setBaseId] = useState('')
  const [description, setDescription] = useState('')
  const [references, setReferences] = useState<Reference[]>([])
  const [uploading, setUploading] = useState(false)
  const [spec, setSpec] = useState<SkillSpec | null>(null)
  const [analyses, setAnalyses] = useState<ReferenceAnalysis[]>([])
  const [busy, setBusy] = useState<'' | 'draft' | 'test' | 'save' | 'delete'>('')
  const [error, setError] = useState<string | null>(null)
  const [brief, setBrief] = useState('')
  const [testResult, setTestResult] = useState<Awaited<ReturnType<typeof testDirectorSkill>> | null>(null)
  const editing = mode.startsWith('edit:') ? mode.slice(5) : ''

  useEffect(() => {
    setTestResult(null)
    if (!editing) { setSpec(null); return }
    fetchDirectorSkill(editing).then(setSpec).catch(e => setError(e instanceof Error ? e.message : String(e)))
  }, [editing])

  const run = async (step: typeof busy, action: () => Promise<void>) => {
    setBusy(step); setError(null)
    try { await action() } catch (e) { setError(e instanceof Error ? e.message : String(e)) } finally { setBusy('') }
  }

  const addFiles = async (files: FileList | null) => {
    if (!files?.length) return
    setUploading(true); setError(null)
    try {
      const added: Reference[] = []
      for (const file of Array.from(files).slice(0, 8 - references.length)) {
        const uploaded = await uploadImage(file)
        added.push({ name: file.name, path: uploaded.path, kind: kindOf(file) })
      }
      setReferences(current => [...current, ...added])
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setUploading(false)
    }
  }

  const draft = () => run('draft', async () => {
    const result = await draftDirectorSkill({
      description, reference_paths: references.map(r => r.path),
      ...(editing ? { edit_skill_id: editing } : baseId ? { base_skill_id: baseId } : {}),
    })
    setSpec(result.spec); setAnalyses(result.analyses); setTestResult(null)
  })

  const patch = (changes: Partial<SkillSpec>) => setSpec(current => (current ? { ...current, ...changes } : current))
  const patchRules = (changes: Partial<SkillSpec['rules']>) =>
    setSpec(current => (current ? { ...current, rules: { ...current.rules, ...changes } } : current))
  const customSkills = skills.filter(s => !s.builtin)

  return (
    <div className="fixed inset-0 z-[80] flex items-center justify-center bg-black/50 p-4" role="dialog" aria-modal="true"
      aria-label="Skill Creator" onClick={e => { if (e.target === e.currentTarget) onClose() }}>
      <div className="w-full max-w-2xl max-h-[90vh] overflow-y-auto bg-bg-secondary border border-border rounded-xl p-4 space-y-3">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-medium text-text-primary">Skill Creator</h2>
          <button onClick={onClose} className="ml-auto text-text-muted hover:text-text-primary" aria-label="Close"><X size={16} /></button>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          <label>
            <span className={label}>Skill</span>
            <select className={input} value={mode} onChange={e => setMode(e.target.value)}>
              <option value="new">Create a new skill</option>
              {customSkills.map(s => <option key={s.id} value={`edit:${s.id}`}>Edit: {s.name}</option>)}
            </select>
          </label>
          {!editing && (
            <label>
              <span className={label}>Start from (inspiration)</span>
              <select className={input} value={baseId} onChange={e => setBaseId(e.target.value)}>
                <option value="">Nothing — start fresh</option>
                {skills.map(s => <option key={s.id} value={s.id}>{s.name}{s.builtin ? ' (built-in)' : ''}</option>)}
              </select>
            </label>
          )}
        </div>

        <label className="block">
          <span className={label}>{editing ? 'What should change?' : 'Describe the skill'}</span>
          <textarea className={`${input} min-h-[72px]`} value={description} onChange={e => setDescription(e.target.value)}
            placeholder="e.g. 30-second recipe reels: overhead shots, hands only, one step per shot, satisfying final reveal…" />
        </label>

        <div>
          <span className={label}>References (images, videos, audio)</span>
          <div className="flex flex-wrap gap-1.5">
            {references.map(ref => (
              <span key={ref.path} className="flex items-center gap-1 text-[10px] bg-bg-tertiary border border-border rounded px-1.5 py-0.5">
                {ref.kind}: {ref.name}
                <button onClick={() => setReferences(rs => rs.filter(r => r.path !== ref.path))} aria-label={`Remove ${ref.name}`}>
                  <X size={10} />
                </button>
              </span>
            ))}
            {references.length < 8 && (
              <label className="flex items-center gap-1 text-[10px] text-accent-blue cursor-pointer hover:underline">
                {uploading ? <Loader2 size={10} className="animate-spin" /> : <Paperclip size={10} />} Add files
                <input type="file" multiple accept="image/*,video/*,audio/*" className="hidden"
                  onChange={e => { void addFiles(e.target.files); e.target.value = '' }} />
              </label>
            )}
          </div>
          <p className="text-[10px] text-text-muted mt-1">
            Analysed by a video-capable LLM through fal.ai for their craft (rhythm, camera, look, sound), never copied.
          </p>
        </div>

        <button onClick={draft} disabled={!!busy || uploading || (!description.trim() && !editing && !baseId)}
          className="px-3 py-1.5 text-xs rounded-lg bg-accent-blue text-white disabled:opacity-50 flex items-center gap-1.5">
          {busy === 'draft' && <Loader2 size={12} className="animate-spin" />}
          {busy === 'draft' ? 'Analysing and drafting…' : spec ? 'Redraft' : 'Analyse & draft'}
        </button>

        {error && <p className="text-[11px] text-chip-red">{error}</p>}

        {analyses.length > 0 && (
          <details className="text-[10px] text-text-muted">
            <summary className="cursor-pointer">Reference analysis ({analyses.length})</summary>
            <ul className="mt-1 space-y-1">
              {analyses.map((a, i) => (
                <li key={i}><span className="text-text-secondary">{a.kind} · {a.name}:</span> {a.error ? <span className="text-chip-red">{a.error}</span> : a.notes}</li>
              ))}
            </ul>
          </details>
        )}

        {spec && (
          <div className="space-y-2 border-t border-border pt-3">
            <div className="grid grid-cols-1 sm:grid-cols-[1fr_auto_auto] gap-2">
              <label><span className={label}>Name</span>
                <input className={input} value={spec.name} onChange={e => patch({ name: e.target.value })} /></label>
              <label><span className={label}>Icon</span>
                <select className={input} value={spec.icon} onChange={e => patch({ icon: e.target.value })}>
                  {Object.keys(SKILL_ICONS).map(icon => <option key={icon} value={icon}>{icon}</option>)}
                </select></label>
              <label><span className={label}>Works from</span>
                <select className={input} value={spec.workflow} onChange={e => patch({ workflow: e.target.value as SkillSpec['workflow'] })}>
                  <option value="story">A brief</option><option value="audio">Uploaded audio</option><option value="both">Either</option>
                </select></label>
            </div>
            <label className="block"><span className={label}>Description</span>
              <input className={input} value={spec.description} onChange={e => patch({ description: e.target.value })} /></label>
            <div className="grid grid-cols-3 gap-2">
              <label><span className={label}>Default length (s)</span>
                <input type="number" className={input} value={spec.default_duration}
                  onChange={e => patch({ default_duration: Number(e.target.value) })} /></label>
              <label><span className={label}>Shot min (s)</span>
                <input type="number" className={input} value={spec.shot_duration[0]}
                  onChange={e => patch({ shot_duration: [Number(e.target.value), spec.shot_duration[1]] })} /></label>
              <label><span className={label}>Shot max (s)</span>
                <input type="number" className={input} value={spec.shot_duration[1]}
                  onChange={e => patch({ shot_duration: [spec.shot_duration[0], Number(e.target.value)] })} /></label>
            </div>
            <label className="block"><span className={label}>Default look (used when a brief gives none)</span>
              <input className={input} value={spec.style_hint} onChange={e => patch({ style_hint: e.target.value })} /></label>
            {RULES.map(([key, title]) => (
              <label key={key} className="block"><span className={label}>{title}</span>
                <textarea className={`${input} min-h-[44px]`} value={spec.rules[key] as string}
                  onChange={e => patchRules({ [key]: e.target.value })} /></label>
            ))}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              <label><span className={label}>Always (one per line)</span>
                <textarea className={`${input} min-h-[60px]`} value={spec.rules.do.join('\n')}
                  onChange={e => patchRules({ do: e.target.value.split('\n') })} /></label>
              <label><span className={label}>Never (one per line)</span>
                <textarea className={`${input} min-h-[60px]`} value={spec.rules.dont.join('\n')}
                  onChange={e => patchRules({ dont: e.target.value.split('\n') })} /></label>
            </div>
            <div><span className={label}>Options offered to users</span>
              <OptionsEditor options={spec.options} onChange={options => patch({ options })} /></div>

            <div className="border-t border-border pt-2 space-y-1.5">
              <span className={label}>Test (plans only, nothing is rendered)</span>
              <div className="flex gap-1.5">
                <input className={input} value={brief} onChange={e => setBrief(e.target.value)} placeholder="A sample brief" />
                <button disabled={!brief.trim() || !!busy} onClick={() => run('test', async () => {
                  setTestResult(await testDirectorSkill({ ...spec, rules: { ...spec.rules, do: spec.rules.do.filter(Boolean), dont: spec.rules.dont.filter(Boolean) } }, brief, {}))
                })} className="px-2.5 text-xs rounded-lg border border-border text-text-secondary disabled:opacity-50 flex items-center gap-1">
                  {busy === 'test' && <Loader2 size={11} className="animate-spin" />}Test
                </button>
              </div>
              {testResult && (
                <ol className="text-[10px] text-text-secondary space-y-0.5 list-decimal pl-4">
                  {testResult.shots.map((shot, i) => (
                    <li key={i}><span className="text-text-muted">{shot.duration_sec.toFixed(1)}s · {shot.type}</span> — {shot.goal}</li>
                  ))}
                </ol>
              )}
            </div>

            <div className="flex items-center gap-2 pt-1">
              <button disabled={!!busy || !spec.name.trim()} onClick={() => run('save', async () => {
                await saveDirectorSkill({ ...spec, rules: { ...spec.rules, do: spec.rules.do.filter(Boolean), dont: spec.rules.dont.filter(Boolean) } }, Boolean(editing))
                onSaved(); onClose()
              })} className="px-3 py-1.5 text-xs rounded-lg bg-accent-blue text-white disabled:opacity-50 flex items-center gap-1.5">
                {busy === 'save' && <Loader2 size={12} className="animate-spin" />}{editing ? 'Save changes' : 'Create skill'}
              </button>
              {editing && (
                <button disabled={!!busy} onClick={() => run('delete', async () => {
                  await deleteDirectorSkill(editing); onSaved(); onClose()
                })} className="ml-auto flex items-center gap-1 text-[11px] text-chip-red disabled:opacity-50">
                  <Trash2 size={12} /> Delete skill
                </button>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
