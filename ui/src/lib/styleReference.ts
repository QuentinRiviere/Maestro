/** Pure helpers of the bible's style references (reference videos analysed into a staging grammar). */

export const SHOT_SIZE_LABELS: Record<string, string> = {
  extreme_wide: 'Extreme wide', wide: 'Wide', medium: 'Medium', medium_close_up: 'Medium close-up',
  close_up: 'Close-up', extreme_close_up: 'Extreme close-up', insert: 'Insert', ots: 'Over-the-shoulder',
  pov: 'POV', two_shot: 'Two-shot',
}

/** Scene tags as the server keeps them: lower-case words joined by underscores, unique, at most 8. */
export function parseSceneTags(text: string): string[] {
  const tags = text.split(/[,;\n]+/)
    .map(tag => tag.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 24))
    .filter(Boolean)
  return [...new Set(tags)].slice(0, 8)
}

/** Rules edited as text, one per line (list markers dropped). */
export function rulesFromText(text: string): string[] {
  return text.split('\n')
    .map(line => line.replace(/^\s*(?:[-*•]|\d+[.)])\s*/, '').trim())
    .filter(Boolean)
    .slice(0, 12)
}

export function rulesToText(rules: string[] | undefined | null): string {
  return (rules ?? []).join('\n')
}

export const isAnalysing = (status: string) => status === 'queued' || status === 'analysing'

/** "Close-up 40% · Medium 35%": the largest shares first. */
export function formatShotMix(mix: Record<string, number> | undefined | null, limit = 5): string {
  return Object.entries(mix ?? {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([size, percent]) => `${SHOT_SIZE_LABELS[size] ?? size} ${Math.round(percent)}%`)
    .join(' · ')
}

export function formatSeconds(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return '–'
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)} s`
  const minutes = Math.floor(seconds / 60)
  return `${minutes}:${String(Math.round(seconds % 60)).padStart(2, '0')}`
}

/** Bar heights (0-1) of the pacing curve: short shots (fast pace) give tall bars. */
export function pacingBars(pacing: { asl_sec: number }[] | undefined | null): number[] {
  const values = (pacing ?? []).map(point => point.asl_sec).filter(v => Number.isFinite(v) && v > 0)
  if (!values.length) return []
  const fastest = Math.min(...values)
  return values.map(v => Math.max(0.08, fastest / v))
}
