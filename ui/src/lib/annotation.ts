import type { OutputAnnotation } from '../types'

/** Tags typed as "hand morph, keep": trimmed, inner spaces collapsed, case-insensitive duplicates dropped. */
export function parseTags(text: string): string[] {
  const seen = new Set<string>()
  const tags: string[] = []
  for (const raw of text.split(',')) {
    const tag = raw.split(/\s+/).filter(Boolean).join(' ')
    if (!tag || seen.has(tag.toLowerCase())) continue
    seen.add(tag.toLowerCase())
    tags.push(tag)
  }
  return tags
}

/** Whether an annotation carries anything worth showing. */
export function hasAnnotation(annotation: OutputAnnotation | null | undefined): annotation is OutputAnnotation {
  return !!annotation && (!!annotation.note?.trim() || (annotation.tags?.length ?? 0) > 0)
}
