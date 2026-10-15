/**
 * Play only the part of a clip the edit keeps: a short shot renders at the
 * H3 minimum and the model often invents a cut after the planned action.
 */
export function trimProps(keep: [number, number] | null, loop: boolean) {
  if (!keep) return {}
  const [start, end] = keep
  return {
    onLoadedMetadata: (event: { currentTarget: HTMLVideoElement }) => { event.currentTarget.currentTime = start },
    onTimeUpdate: (event: { currentTarget: HTMLVideoElement }) => {
      const el = event.currentTarget
      if (el.currentTime < start - 0.05) {
        el.currentTime = start
      } else if (el.currentTime >= end) {
        if (!loop) el.pause()
        el.currentTime = start
      }
    },
  }
}
