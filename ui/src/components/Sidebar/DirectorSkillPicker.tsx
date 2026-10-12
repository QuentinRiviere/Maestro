import { useEffect, useMemo, useState } from 'react'
import { Pencil, Search, Wand2 } from 'lucide-react'
import { skillIcon } from '../../lib/skillIcons'
import { useStore } from '../../stores/useStore'
import type { DirectorSkill } from '../../types'
import type { DirectorSkillEntry } from '../../api/directorSkills'
import { SkillCreatorDialog } from './SkillCreatorDialog'

const WORKFLOW_LABEL: Record<string, string> = { story: 'From a brief', audio: 'From audio', both: 'Brief or audio' }

const COMING_SOON = [{ id: 'podcast', name: 'Video Podcast', description: 'Coming soon', icon: 'mic' }]

/** Searchable list of built-in and custom Director skills, plus the Skill Creator. */
export function DirectorSkillPicker({ onSelect }: { onSelect: (skill: DirectorSkill) => void }) {
  const skills = useStore(s => s.directorSkills)
  const loadSkills = useStore(s => s.loadDirectorSkills)
  const [query, setQuery] = useState('')
  const [creator, setCreator] = useState<{ open: boolean; editId?: string }>({ open: false })

  useEffect(() => { void loadSkills() }, [loadSkills])

  const filtered = useMemo(() => {
    const words = query.toLowerCase().split(/\s+/).filter(Boolean)
    const matches = (entry: { name: string; description: string }) =>
      words.every(word => `${entry.name} ${entry.description}`.toLowerCase().includes(word))
    return { skills: skills.filter(matches), soon: COMING_SOON.filter(matches) }
  }, [skills, query])

  const row = (entry: DirectorSkillEntry) => {
    const Icon = skillIcon(entry.icon)
    return (
      <div key={entry.id} className="group flex items-center gap-2 rounded-lg hover:bg-bg-hover">
        <button
          onClick={() => onSelect(entry.id as DirectorSkill)}
          className="flex-1 min-w-0 flex items-start gap-2 px-2 py-2 text-left"
        >
          <Icon size={15} className="text-accent-blue mt-0.5 shrink-0" />
          <span className="min-w-0">
            <span className="flex items-center gap-1.5">
              <span className="text-xs font-medium text-text-primary truncate">{entry.name}</span>
              <span className="text-[9px] px-1 py-0.5 rounded bg-bg-tertiary text-text-muted shrink-0">
                {entry.builtin ? 'Built-in' : `Custom${entry.version ? ` v${entry.version}` : ''}`}
              </span>
            </span>
            <span className="block text-[10px] text-text-muted leading-snug">{entry.description}</span>
            <span className="block text-[9px] text-text-muted mt-0.5">{WORKFLOW_LABEL[entry.workflow] ?? entry.workflow}</span>
          </span>
        </button>
        {!entry.builtin && (
          <button
            onClick={() => setCreator({ open: true, editId: entry.id })}
            title="Edit this skill in the Skill Creator"
            aria-label={`Edit ${entry.name}`}
            className="p-2 text-text-muted hover:text-text-primary opacity-60 group-hover:opacity-100"
          >
            <Pencil size={12} />
          </button>
        )}
      </div>
    )
  }

  return (
    <div className="space-y-2">
      <div className="relative">
        <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-text-muted" />
        <input
          value={query}
          onChange={e => setQuery(e.target.value)}
          placeholder="Search skills…"
          aria-label="Search Director skills"
          className="w-full bg-bg-tertiary border border-border rounded-lg pl-7 pr-3 py-1.5 text-xs text-text-primary focus:outline-none focus:border-accent-blue"
        />
      </div>
      <div className="max-h-[320px] overflow-y-auto space-y-0.5">
        <button
          onClick={() => setCreator({ open: true })}
          className="w-full flex items-start gap-2 px-2 py-2 rounded-lg border border-dashed border-accent-blue/40 hover:bg-accent-blue/5 text-left"
        >
          <Wand2 size={15} className="text-accent-blue mt-0.5 shrink-0" />
          <span>
            <span className="block text-xs font-medium text-text-primary">Skill Creator</span>
            <span className="block text-[10px] text-text-muted">Create or edit a skill from a description and reference media</span>
          </span>
        </button>
        {filtered.skills.map(row)}
        {filtered.soon.map(entry => {
          const Icon = skillIcon(entry.icon)
          return (
            <div key={entry.id} className="flex items-start gap-2 px-2 py-2 opacity-50 cursor-not-allowed">
              <Icon size={15} className="text-text-muted mt-0.5 shrink-0" />
              <span>
                <span className="block text-xs text-text-primary">{entry.name}</span>
                <span className="block text-[10px] text-text-muted">{entry.description}</span>
              </span>
            </div>
          )
        })}
        {!filtered.skills.length && !filtered.soon.length && (
          <p className="px-2 py-3 text-[10px] text-text-muted">No skill matches “{query}”.</p>
        )}
      </div>
      {creator.open && (
        <SkillCreatorDialog
          editId={creator.editId}
          onClose={() => setCreator({ open: false })}
          onSaved={() => { void loadSkills() }}
        />
      )}
    </div>
  )
}
