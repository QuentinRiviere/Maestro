/** Pure helpers of the Persona studio (voices, covers, clips), kept free of React so they can be unit-tested. */

/** Even weights for ``count`` references, or the given ones normalised to sum 1 (bad or empty input: even). */
export function normaliseWeights(count: number, raw: Array<number | string | null | undefined>): number[] {
  if (count <= 0) return []
  const values = raw.slice(0, count).map(v => (typeof v === 'string' ? Number(v) : v ?? NaN))
  const usable = values.map(v => (typeof v === 'number' && Number.isFinite(v) && v > 0 ? v : 0))
  const total = usable.reduce((a, b) => a + b, 0)
  if (usable.length !== count || total <= 0) return Array.from({ length: count }, () => Number((1 / count).toFixed(4)))
  return usable.map(v => Number((v / total).toFixed(4)))
}

/** LTX-2.5 clip length: 24 fps, 8k+1 frames, 17 to 345 frames in one window. */
export function clipFrames(seconds: number): number {
  const wanted = Math.ceil(Math.max(0, seconds) * 24)
  const frames = Math.ceil(Math.max(wanted - 1, 0) / 8) * 8 + 1
  return Math.max(17, Math.min(345, frames))
}

export const clipSeconds = (frames: number) => (frames - 1) / 24

/** The longest clip one LTX window holds, in seconds. */
export const MAX_CLIP_SECONDS = clipSeconds(345)

/** "mm:ss" for a position in a song. */
export function formatSeconds(seconds: number): string {
  const whole = Math.max(0, Math.floor(seconds))
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`
}
