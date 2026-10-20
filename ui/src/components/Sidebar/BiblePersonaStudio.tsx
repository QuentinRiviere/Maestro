import { useCallback, useEffect, useState, type ReactNode } from 'react'
import { Clapperboard, Dumbbell, Loader2, Mic, Music, Plus, Trash2, Video, XCircle } from 'lucide-react'
import {
  cancelCover, cancelCoverVideo, cancelPersonaClip, cancelVoiceTraining, coverAudioUrl, coverVideoUrl, createCover, createPersonaClip,
  createVoice, deleteCover, deletePersonaClip, deleteVoice, fetchCovers, fetchPersonaClips, fetchVoices, isLive, personaClipFileUrl,
  startCoverVideo, trainVoice, voiceReferenceUrl,
  type CharacterVoice, type CoverEngine, type PersonaClip, type SongCover, type VoiceMode,
} from '../../api/directorPersona'
import { generateMusic, getFileUrl } from '../../api/client'
import type { DirectorBible } from '../../api/directorBibles'
import { clipFrames, clipSeconds, formatSeconds, MAX_CLIP_SECONDS, MAX_VIDEO_SECONDS, normaliseWeights, videoFrames, videoWindows } from '../../lib/persona'

const input = 'w-full bg-bg-tertiary border border-border rounded px-2 py-1 text-[11px] text-text-primary focus:outline-none focus:border-accent-blue'
const label = 'text-[9px] text-text-muted uppercase tracking-wider block mb-0.5'
const button = 'flex items-center gap-1 rounded border border-border px-2 py-0.5 text-[10px] text-text-secondary hover:text-text-primary disabled:opacity-40'
const primary = 'flex items-center gap-1 rounded bg-accent-blue px-2 py-1 text-[10px] text-white disabled:opacity-40'
const STATUS_CLASS: Record<string, string> = {
  queued: 'text-text-muted', running: 'text-accent-blue', done: 'text-emerald-300', failed: 'text-chip-red',
}
const DEFAULT_LYRICS = '[Verse]\nCity lights are falling on the window pane\nI keep your picture in the pouring rain\n[Chorus]\nStay with me tonight, don\'t let the morning in\nHold the silence close, let the song begin'
const DEFAULT_STYLE = 'Melancholic Korean indie pop ballad, clear lead vocals, piano and soft drums, 90 bpm.'

type Ask = (request: { title: string; message: ReactNode; confirmLabel?: string; danger?: boolean; run: () => void | Promise<void> }) => void

function Status({ status, progress, error }: { status: string; progress?: string; error?: string }) {
  return (
    <span className={`shrink-0 flex items-center gap-1 text-[10px] ${STATUS_CLASS[status] ?? ''}`} title={error || progress}>
      {isLive(status) && <Loader2 size={10} className="animate-spin" />} {status}
    </span>
  )
}

function useAct() {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const act = async (call: () => Promise<void>) => {
    setBusy(true)
    setError(null)
    try { await call() } catch (e) { setError(e instanceof Error ? e.message : 'Failed') } finally { setBusy(false) }
  }
  return { busy, error, act, setError }
}

function ElementSelect({ bible, kind, value, onChange, allowNone, ariaLabel }: {
  bible: DirectorBible
  kind: 'characters' | 'locations'
  value: string
  onChange: (id: string) => void
  allowNone?: string
  ariaLabel: string
}) {
  const elements = bible[kind].filter(e => e.id && e.image)
  return (
    <select className={input} value={value} onChange={e => onChange(e.target.value)} aria-label={ariaLabel}>
      {allowNone !== undefined && <option value="">{allowNone}</option>}
      {elements.map(e => <option key={e.id} value={e.id}>{e.name}</option>)}
    </select>
  )
}

// ── voices ────────────────────────────────────────────────────────────────

function VoiceForm({ bible, onCreated }: { bible: DirectorBible; onCreated: (voice: CharacterVoice) => void }) {
  const [name, setName] = useState('')
  const [files, setFiles] = useState<File[]>([])
  const [weights, setWeights] = useState<string[]>([])
  const [mode, setMode] = useState<VoiceMode>('sing')
  const [characterId, setCharacterId] = useState(bible.characters.find(c => c.id)?.id ?? '')
  const { busy, error, act } = useAct()
  const normalised = normaliseWeights(files.length, weights)
  return (
    <div className="rounded-lg border border-border bg-bg-tertiary/30 p-2.5 space-y-2">
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
        <label className="block"><span className={label}>Voice name</span>
          <input className={input} value={name} onChange={e => setName(e.target.value)} placeholder="Hae-won's voice" /></label>
        <label className="block"><span className={label}>Character</span>
          <ElementSelect bible={bible} kind="characters" value={characterId} onChange={setCharacterId} allowNone="No character" ariaLabel="Voice character" /></label>
        <label className="block"><span className={label}>Mode</span>
          <select className={input} value={mode} onChange={e => setMode(e.target.value as VoiceMode)} aria-label="Voice mode">
            <option value="sing">Sing (pitch follows the song)</option>
            <option value="rap">Rap (pitch from the reference)</option>
          </select></label>
      </div>
      <label className="block"><span className={label}>Reference voices (1 to 4 files, 5 to 30 s each, sung when possible; several blend into one voice)</span>
        <input type="file" accept="audio/*" multiple className="text-[10px] text-text-secondary" aria-label="Reference voice files"
          onChange={e => { const list = Array.from(e.target.files ?? []).slice(0, 4); setFiles(list); setWeights(list.map(() => '')) }} /></label>
      {files.length > 1 && (
        <div className="flex flex-wrap gap-2">
          {files.map((file, index) => (
            <label key={file.name + index} className="flex items-center gap-1 text-[10px] text-text-secondary">
              <span className="max-w-[140px] truncate" title={file.name}>{file.name}</span>
              <input className={`${input} w-14`} type="number" min={0} step={0.1} placeholder="1" value={weights[index] ?? ''}
                aria-label={`Weight of ${file.name}`}
                onChange={e => setWeights(weights.map((w, i) => (i === index ? e.target.value : w)))} />
              <span className="text-text-muted">{Math.round(normalised[index] * 100)}%</span>
            </label>
          ))}
        </div>
      )}
      {error && <p role="alert" className="text-[10px] text-chip-red">{error}</p>}
      <button className={primary} disabled={busy || files.length === 0}
        onClick={() => void act(async () => {
          onCreated(await createVoice({ name: name || files[0].name.replace(/\.[^.]+$/, ''), references: files, weights: normalised, mode,
            bible_id: bible.id, character_id: characterId }))
          setFiles([]); setName('')
        })}>
        {busy ? <Loader2 size={10} className="animate-spin" /> : <Plus size={10} />} Create the voice
      </button>
    </div>
  )
}

function VoiceCard({ bible, voice, ask, onChange, onDeleted }: {
  bible: DirectorBible; voice: CharacterVoice; ask: Ask; onChange: (voice: CharacterVoice) => void; onDeleted: (voices: CharacterVoice[]) => void
}) {
  const { busy, error, act } = useAct()
  const character = bible.characters.find(c => c.id === voice.character_id)
  const rvc = voice.rvc
  const training = isLive(rvc?.status)
  return (
    <div className="rounded-lg border border-border bg-bg-tertiary/30 p-2.5 space-y-1.5">
      <div className="flex items-center gap-2">
        <Mic size={12} className="shrink-0 text-text-muted" />
        <span className="min-w-0 flex-1 truncate text-[11px] font-medium text-text-primary">{voice.name}</span>
        <span className="text-[10px] text-text-muted">{voice.mode}{character ? ` · ${character.name}` : ''}{voice.covers ? ` · ${voice.covers} cover${voice.covers === 1 ? '' : 's'}` : ''}{voice.profile ? ' · profile ready' : ''}</span>
        {rvc && <Status status={rvc.status} progress={rvc.progress} error={rvc.error} />}
        {training ? (
          <button className={button} disabled={busy} title="Stop the training"
            onClick={() => void act(async () => onChange(await cancelVoiceTraining(voice.id)))}><XCircle size={10} /> Stop</button>
        ) : (
          <button className={button} disabled={busy}
            title="Train a permanent RVC model of this voice on its references and the covers it already sang (Applio, a few hundred epochs). Covers then use it by default: steadier timbre from song to song."
            onClick={() => ask({ title: `Train "${voice.name}" as an RVC model?`, confirmLabel: 'Train',
              message: `The worker trains for 300 epochs on the references and every sung cover of this voice${rvc?.status === 'done' ? '; the current model is replaced' : ''}.`,
              run: () => act(async () => onChange(await trainVoice(voice.id, { epochs: 300 }))) })}>
            <Dumbbell size={10} /> {rvc?.status === 'done' ? 'Retrain' : 'Train RVC'}
          </button>
        )}
        <button className={button} disabled={busy || training} title="Delete the voice and its references"
          onClick={() => ask({ title: `Delete "${voice.name}"?`, confirmLabel: 'Delete', danger: true,
            message: 'Covers already made keep their audio.', run: () => act(async () => onDeleted(await deleteVoice(voice.id))) })}>
          <Trash2 size={10} />
        </button>
      </div>
      <div className="flex flex-wrap gap-2">
        {voice.references.map((reference, index) => (
          <div key={reference.file} className="min-w-[200px] flex-1">
            <span className="text-[10px] text-text-muted">{reference.name} · {Math.round(reference.weight * 100)}%</span>
            <audio controls preload="none" src={voiceReferenceUrl(voice.id, index + 1)} className="w-full h-7" aria-label={`Reference ${index + 1} of ${voice.name}`} />
          </div>
        ))}
      </div>
      {rvc && (
        <p className="text-[10px] text-text-muted">
          RVC model: {rvc.status}{training && rvc.progress ? ` · ${rvc.progress}` : ''}{rvc.dataset ? ` · ${rvc.dataset} recording${rvc.dataset === 1 ? '' : 's'}` : ''}
          {rvc.config?.epochs ? ` · ${rvc.config.epochs} epochs` : ''}{rvc.seconds ? ` · ${Math.round(rvc.seconds / 60)} min` : ''}
          {rvc.error && <span className="text-chip-red"> · {rvc.error}</span>}
        </p>
      )}
      {error && <p role="alert" className="text-[10px] text-chip-red">{error}</p>}
    </div>
  )
}

// ── covers ────────────────────────────────────────────────────────────────

function CoverForm({ bible, voices, onCreated }: { bible: DirectorBible; voices: CharacterVoice[]; onCreated: (cover: SongCover) => void }) {
  const [name, setName] = useState('')
  const [song, setSong] = useState<File | null>(null)
  const [voiceId, setVoiceId] = useState(voices[0]?.id ?? '')
  const [mode, setMode] = useState<VoiceMode | ''>('')
  const [engine, setEngine] = useState<CoverEngine>('auto')
  const [source, setSource] = useState<'upload' | 'generate'>('upload')
  const [lyrics, setLyrics] = useState(DEFAULT_LYRICS)
  const [style, setStyle] = useState(DEFAULT_STYLE)
  const [duration, setDuration] = useState(40)
  const [rendered, setRendered] = useState<{ filename: string; workspace: string } | null>(null)
  const [rendering, setRendering] = useState(false)
  const { busy, error, act, setError } = useAct()
  useEffect(() => { if (!voiceId && voices[0]) setVoiceId(voices[0].id) }, [voices, voiceId])

  const render = async () => {
    setRendering(true)
    setError(null)
    try {
      const result = await generateMusic({ lyrics, style, duration_seconds: duration, model_type: 'ace_step_v1_5_turbo_lm_1_7b', workspace: 'default' })
      setRendered({ filename: result.filename, workspace: 'default' })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'The song could not be rendered')
    } finally {
      setRendering(false)
    }
  }
  const ready = !!voiceId && (source === 'upload' ? !!song : !!rendered)
  return (
    <div className="rounded-lg border border-border bg-bg-tertiary/30 p-2.5 space-y-2">
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-2">
        <label className="block"><span className={label}>Cover name</span>
          <input className={input} value={name} onChange={e => setName(e.target.value)} placeholder="Night song" /></label>
        <label className="block"><span className={label}>Voice</span>
          <select className={input} value={voiceId} onChange={e => setVoiceId(e.target.value)} aria-label="Cover voice">
            {voices.length === 0 && <option value="">Create a voice first</option>}
            {voices.map(v => <option key={v.id} value={v.id}>{v.name} ({v.mode})</option>)}
          </select></label>
        <label className="block"><span className={label}>Mode</span>
          <select className={input} value={mode} onChange={e => setMode(e.target.value as VoiceMode | '')} aria-label="Cover mode">
            <option value="">The voice's own</option>
            <option value="sing">Sing</option>
            <option value="rap">Rap</option>
          </select></label>
        <label className="block"><span className={label}>Engine</span>
          <select className={input} value={engine} onChange={e => setEngine(e.target.value as CoverEngine)} aria-label="Cover engine">
            <option value="auto">Auto (RVC model when trained)</option>
            <option value="seedvc">Seed-VC (zero-shot)</option>
            <option value="rvc">RVC (trained model)</option>
          </select></label>
      </div>
      <div className="flex gap-1">
        {(['upload', 'generate'] as const).map(option => (
          <button key={option} className={`rounded px-2 py-0.5 text-[10px] ${source === option ? 'bg-bg-hover text-text-primary' : 'text-text-secondary hover:text-text-primary'}`}
            onClick={() => setSource(option)}>
            {option === 'upload' ? 'Upload a song' : 'Write and render a song'}
          </button>
        ))}
      </div>
      {source === 'upload' ? (
        <label className="block"><span className={label}>Song with vocals (rights cleared)</span>
          <input type="file" accept="audio/*" className="text-[10px] text-text-secondary" aria-label="Song file"
            onChange={e => setSong(e.target.files?.[0] ?? null)} /></label>
      ) : (
        <div className="space-y-2">
          <label className="block"><span className={label}>Lyrics ([Verse] / [Chorus] tags, one line per lyric)</span>
            <textarea className={input} rows={5} value={lyrics} onChange={e => setLyrics(e.target.value)} /></label>
          <div className="grid grid-cols-1 sm:grid-cols-[1fr_90px_auto] gap-2 items-end">
            <label className="block"><span className={label}>Style</span>
              <input className={input} value={style} onChange={e => setStyle(e.target.value)} /></label>
            <label className="block"><span className={label}>Seconds</span>
              <input className={input} type="number" min={10} max={240} value={duration} onChange={e => setDuration(Number(e.target.value) || 40)} /></label>
            <button className={button} disabled={rendering || !lyrics.trim()} onClick={() => void render()} title="ACE-Step 1.5 with its lyrics model, on the worker (a minute or more)">
              {rendering ? <Loader2 size={10} className="animate-spin" /> : <Music size={10} />} Render
            </button>
          </div>
          {rendered && (
            <div>
              <span className={label}>Rendered song</span>
              <audio controls preload="metadata" src={getFileUrl(rendered.filename, rendered.workspace)} className="w-full h-7" aria-label="Rendered song" />
            </div>
          )}
        </div>
      )}
      {error && <p role="alert" className="text-[10px] text-chip-red">{error}</p>}
      <button className={primary} disabled={busy || !ready}
        onClick={() => void act(async () => {
          onCreated(await createCover({ name: name || (song ? song.name.replace(/\.[^.]+$/, '') : 'Rendered song'), song: source === 'upload' ? song : null,
            song_output: source === 'generate' ? rendered?.filename : undefined, workspace: source === 'generate' ? rendered?.workspace : undefined,
            voice_id: voiceId, mode, engine, bible_id: bible.id }))
          setSong(null); setName(''); setRendered(null)
        })}>
        {busy ? <Loader2 size={10} className="animate-spin" /> : <Plus size={10} />} Sing it
      </button>
    </div>
  )
}

function CoverVideoForm({ bible, cover, onChange }: { bible: DirectorBible; cover: SongCover; onChange: (cover: SongCover) => void }) {
  const [characterId, setCharacterId] = useState(cover.character_id || bible.characters.find(c => c.id && c.image)?.id || '')
  const [locationId, setLocationId] = useState('')
  const [seconds, setSeconds] = useState(12)
  const [start, setStart] = useState(0)
  const [lyrics, setLyrics] = useState('')
  const { busy, error, act } = useAct()
  const frames = videoFrames(seconds)
  const windows = videoWindows(frames)
  return (
    <div className="space-y-2 rounded border border-border/60 p-2">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        <label className="block"><span className={label}>Singer</span>
          <ElementSelect bible={bible} kind="characters" value={characterId} onChange={setCharacterId} ariaLabel="Video singer" /></label>
        <label className="block"><span className={label}>Place</span>
          <ElementSelect bible={bible} kind="locations" value={locationId} onChange={setLocationId} allowNone="A home studio" ariaLabel="Video place" /></label>
        <label className="block"><span className={label}>From (s)</span>
          <input className={input} type="number" min={0} step={0.5} value={start} onChange={e => setStart(Number(e.target.value) || 0)} /></label>
        <label className="block"><span className={label}>Length (s, ≤ {MAX_VIDEO_SECONDS}; beyond {MAX_CLIP_SECONDS.toFixed(0)} s LTX chains windows)</span>
          <input className={input} type="number" min={1} max={MAX_VIDEO_SECONDS} step={1} value={seconds} onChange={e => setSeconds(Number(e.target.value) || 12)} /></label>
      </div>
      <label className="block"><span className={label}>Lyrics of that part (quoted in the prompt, optional)</span>
        <input className={input} value={lyrics} onChange={e => setLyrics(e.target.value)} placeholder="City lights are falling on the window pane / …" /></label>
      {error && <p role="alert" className="text-[10px] text-chip-red">{error}</p>}
      <button className={primary} disabled={busy || !characterId}
        onClick={() => void act(async () => onChange(await startCoverVideo(cover.id, { character_id: characterId, location_id: locationId, start_sec: start, seconds, lyrics })))}>
        {busy ? <Loader2 size={10} className="animate-spin" /> : <Video size={10} />} Render {formatSeconds(start)} to {formatSeconds(start + clipSeconds(frames))} ({frames} frames, {windows} window{windows === 1 ? '' : 's'}, 9:16)
      </button>
    </div>
  )
}

function CoverCard({ bible, cover, ask, onChange, onDeleted }: {
  bible: DirectorBible; cover: SongCover; ask: Ask; onChange: (cover: SongCover) => void; onDeleted: (covers: SongCover[]) => void
}) {
  const { busy, error, act } = useAct()
  const [videoForm, setVideoForm] = useState(false)
  const live = isLive(cover.status)
  const video = cover.video
  const videoLive = isLive(video?.status)
  const stamp = encodeURIComponent(cover.updated_at)
  return (
    <div className="rounded-lg border border-border bg-bg-tertiary/30 p-2.5 space-y-2">
      <div className="flex items-center gap-2">
        <Music size={12} className="shrink-0 text-text-muted" />
        <span className="min-w-0 flex-1 truncate text-[11px] font-medium text-text-primary">{cover.name}</span>
        <span className="text-[10px] text-text-muted">{cover.config.mode}{cover.config.engine === 'rvc' ? ' · RVC' : ''}{cover.seconds ? ` · ${cover.seconds} s` : ''}</span>
        <Status status={cover.status} progress={cover.progress} error={cover.error} />
        {live && <button className={button} disabled={busy} title="Stop the conversion"
          onClick={() => void act(async () => onChange(await cancelCover(cover.id)))}><XCircle size={10} /> Stop</button>}
        {cover.status === 'done' && !videoLive && (
          <button className={button} disabled={busy} title="A singing video: the character at a microphone, lips driven by the converted vocals (LTX-2.5)"
            onClick={() => setVideoForm(v => !v)}><Video size={10} /> Video</button>
        )}
        <button className={button} disabled={busy || live || videoLive} title="Delete the cover and its files"
          onClick={() => ask({ title: `Delete "${cover.name}"?`, confirmLabel: 'Delete', danger: true,
            message: 'The song, the stems, the cover and its video are removed.', run: () => act(async () => onDeleted(await deleteCover(cover.id))) })}>
          <Trash2 size={10} />
        </button>
      </div>
      {live && <p className="text-[10px] text-text-secondary">{cover.progress || cover.status}</p>}
      {cover.error && <p role="alert" className="text-[10px] text-chip-red">{cover.error}</p>}
      {error && <p role="alert" className="text-[10px] text-chip-red">{error}</p>}
      {cover.status === 'done' && (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
          {([['cover', 'Cover (mix)'], ['vocals', 'Converted vocals'], ['song', 'Original song']] as const).map(([which, title]) => (
            <div key={which}>
              <span className={label}>{title}</span>
              <audio controls preload="none" src={`${coverAudioUrl(cover.id, which)}?v=${stamp}`} className="w-full h-7" aria-label={`${title} of ${cover.name}`} />
            </div>
          ))}
        </div>
      )}
      {videoForm && cover.status === 'done' && !videoLive && (
        <CoverVideoForm bible={bible} cover={cover} onChange={next => { setVideoForm(false); onChange(next) }} />
      )}
      {video && (
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Video size={12} className="shrink-0 text-text-muted" />
            <span className="text-[10px] text-text-secondary">
              Video {formatSeconds(video.settings.start_sec)} to {formatSeconds(video.settings.start_sec + video.settings.seconds)}
              {video.seconds ? ` · ${Math.round(video.seconds / 60)} min` : ''}
            </span>
            <Status status={video.status} progress={video.progress} error={video.error} />
            {videoLive && <button className={button} disabled={busy} title="Stop the video"
              onClick={() => void act(async () => onChange(await cancelCoverVideo(cover.id)))}><XCircle size={10} /> Stop</button>}
          </div>
          {videoLive && <p className="text-[10px] text-text-secondary">{video.progress || video.status}</p>}
          {video.error && <p role="alert" className="text-[10px] text-chip-red">{video.error}</p>}
          <div className="flex gap-2">
            {video.outputs.start && (
              <img alt={`Start image of ${cover.name}`} className="h-[180px] w-auto rounded bg-black object-contain"
                src={`${coverVideoUrl(cover.id, 'start')}?v=${stamp}`} />
            )}
            {video.outputs.video && video.status === 'done' && (
              <video controls playsInline preload="metadata" className="h-[180px] w-auto rounded bg-black"
                src={`${coverVideoUrl(cover.id, 'video')}?v=${stamp}`} aria-label={`Video of ${cover.name}`} />
            )}
          </div>
        </div>
      )}
    </div>
  )
}

// ── persona clips ─────────────────────────────────────────────────────────

function ClipForm({ bible, onCreated }: { bible: DirectorBible; onCreated: (clip: PersonaClip) => void }) {
  const [characterId, setCharacterId] = useState(bible.characters.find(c => c.id && c.image)?.id ?? '')
  const [locationId, setLocationId] = useState('')
  const [place, setPlace] = useState('')
  const [gesture, setGesture] = useState('')
  const [seconds, setSeconds] = useState(6)
  const [audio, setAudio] = useState<'silent' | 'model'>('silent')
  const { busy, error, act } = useAct()
  const frames = clipFrames(seconds)
  return (
    <div className="rounded-lg border border-border bg-bg-tertiary/30 p-2.5 space-y-2">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        <label className="block"><span className={label}>Persona</span>
          <ElementSelect bible={bible} kind="characters" value={characterId} onChange={setCharacterId} ariaLabel="Clip persona" /></label>
        <label className="block"><span className={label}>Place (bible)</span>
          <ElementSelect bible={bible} kind="locations" value={locationId} onChange={setLocationId} allowNone="In words below" ariaLabel="Clip place" /></label>
        <label className="block"><span className={label}>Length (s)</span>
          <input className={input} type="number" min={1} max={MAX_CLIP_SECONDS} step={1} value={seconds} onChange={e => setSeconds(Number(e.target.value) || 6)} /></label>
        <label className="block"><span className={label}>Sound</span>
          <select className={input} value={audio} onChange={e => setAudio(e.target.value as 'silent' | 'model')} aria-label="Clip sound">
            <option value="silent">Silent (mouth shut)</option>
            <option value="model">Let LTX invent a soundtrack</option>
          </select></label>
      </div>
      {!locationId && (
        <label className="block"><span className={label}>Place in words (when not a bible place)</span>
          <input className={input} value={place} onChange={e => setPlace(e.target.value)} placeholder="a rooftop at dusk, Seoul skyline behind" /></label>
      )}
      <label className="block"><span className={label}>Signature gesture (the same every clip)</span>
        <input className={input} value={gesture} onChange={e => setGesture(e.target.value)}
          placeholder="straightens the trench collar, taps the temple twice, points at the camera with a slow nod" /></label>
      {error && <p role="alert" className="text-[10px] text-chip-red">{error}</p>}
      <button className={primary} disabled={busy || !characterId || !gesture.trim()}
        onClick={() => void act(async () => {
          onCreated(await createPersonaClip({ bible_id: bible.id, character_id: characterId, location_id: locationId, place: locationId ? '' : place,
            gesture, seconds, audio }))
        })}>
        {busy ? <Loader2 size={10} className="animate-spin" /> : <Clapperboard size={10} />} Render the clip ({frames} frames, 9:16)
      </button>
    </div>
  )
}

function ClipCard({ bible, clip, ask, onChange, onDeleted }: {
  bible: DirectorBible; clip: PersonaClip; ask: Ask; onChange: (clip: PersonaClip) => void; onDeleted: (clips: PersonaClip[]) => void
}) {
  const { busy, error, act } = useAct()
  const live = isLive(clip.status)
  const character = bible.characters.find(c => c.id === clip.settings.character_id)
  const location = bible.locations.find(l => l.id === clip.settings.location_id)
  const stamp = encodeURIComponent(clip.updated_at)
  return (
    <div className="rounded-lg border border-border bg-bg-tertiary/30 p-2.5 space-y-1.5">
      <div className="flex items-center gap-2">
        <Clapperboard size={12} className="shrink-0 text-text-muted" />
        <span className="min-w-0 flex-1 truncate text-[11px] font-medium text-text-primary" title={clip.settings.gesture}>
          {character?.name ?? clip.settings.character_id} · {location?.name ?? clip.settings.place ?? 'no place'}
        </span>
        <span className="text-[10px] text-text-muted">{clip.settings.seconds} s · {clip.settings.audio}{clip.seconds ? ` · ${Math.round(clip.seconds / 60)} min` : ''}</span>
        <Status status={clip.status} progress={clip.progress} error={clip.error} />
        {live && <button className={button} disabled={busy} title="Stop the clip"
          onClick={() => void act(async () => onChange(await cancelPersonaClip(clip.id)))}><XCircle size={10} /> Stop</button>}
        <button className={button} disabled={busy || live} title="Delete the clip"
          onClick={() => ask({ title: `Delete this clip?`, confirmLabel: 'Delete', danger: true,
            message: 'The start image and the video are removed.', run: () => act(async () => onDeleted(await deletePersonaClip(clip.id))) })}>
          <Trash2 size={10} />
        </button>
      </div>
      <p className="text-[10px] text-text-muted">{clip.settings.gesture}</p>
      {live && <p className="text-[10px] text-text-secondary">{clip.progress || clip.status}</p>}
      {clip.error && <p role="alert" className="text-[10px] text-chip-red">{clip.error}</p>}
      {error && <p role="alert" className="text-[10px] text-chip-red">{error}</p>}
      <div className="flex gap-2">
        {clip.outputs.start && (
          <img alt={`Start image of the clip`} className="h-[180px] w-auto rounded bg-black object-contain" src={`${personaClipFileUrl(clip.id, 'start')}?v=${stamp}`} />
        )}
        {clip.outputs.clip && clip.status === 'done' && (
          <video controls playsInline preload="metadata" className="h-[180px] w-auto rounded bg-black" src={`${personaClipFileUrl(clip.id, 'clip')}?v=${stamp}`} aria-label="Persona clip" />
        )}
      </div>
    </div>
  )
}

// ── the studio ────────────────────────────────────────────────────────────

/** Voices, covers and persona clips of a bible: the short-form persona formats, all on the worker. */
export function BiblePersonaStudio({ bible, ask }: { bible: DirectorBible; ask: Ask }) {
  const [voices, setVoices] = useState<CharacterVoice[]>([])
  const [covers, setCovers] = useState<SongCover[]>([])
  const [clips, setClips] = useState<PersonaClip[]>([])
  const [error, setError] = useState<string | null>(null)
  const [forms, setForms] = useState({ voice: false, cover: false, clip: false })
  const bibleId = bible.id

  const refresh = useCallback(() => Promise.all([fetchVoices(bibleId), fetchCovers(bibleId), fetchPersonaClips(bibleId)])
    .then(([v, c, k]) => { setVoices(v); setCovers(c); setClips(k); setError(null) })
    .catch(e => setError(e instanceof Error ? e.message : 'Could not load the studio')), [bibleId])
  useEffect(() => { void refresh() }, [refresh])

  const running = covers.some(c => isLive(c.status) || isLive(c.video?.status)) || clips.some(c => isLive(c.status))
    || voices.some(v => isLive(v.rvc?.status))
  useEffect(() => {
    if (!running) return
    const timer = window.setInterval(() => { void refresh() }, 4000)
    return () => window.clearInterval(timer)
  }, [running, refresh])

  const replaceCover = (next: SongCover) => setCovers(current => current.map(c => (c.id === next.id ? next : c)))
  const replaceClip = (next: PersonaClip) => setClips(current => current.map(c => (c.id === next.id ? next : c)))
  const toggle = (key: keyof typeof forms) => setForms(current => ({ ...current, [key]: !current[key] }))

  return (
    <div className="space-y-5">
      {error && <p role="alert" className="text-[10px] text-chip-red">{error}</p>}

      <section className="space-y-2">
        <div className="flex items-center justify-between gap-2">
          <div>
            <span className={label}>Character voices</span>
            <p className="text-[10px] text-text-muted">One to four reference recordings blended into a voice (Seed-VC); a profile is kept after the first cover.</p>
          </div>
          <button className={button} onClick={() => toggle('voice')}><Plus size={10} /> Voice</button>
        </div>
        {forms.voice && <VoiceForm bible={bible} onCreated={voice => { setVoices(current => [...current, voice]); toggle('voice') }} />}
        {voices.map(voice => <VoiceCard key={voice.id} bible={bible} voice={voice} ask={ask}
          onChange={next => setVoices(current => current.map(v => (v.id === next.id ? next : v)))}
          onDeleted={next => setVoices(next.filter(v => !v.bible_id || v.bible_id === bibleId))} />)}
        {voices.length === 0 && !forms.voice && <p className="text-[10px] text-text-muted">No voice yet.</p>}
      </section>

      <section className="space-y-2">
        <div className="flex items-center justify-between gap-2">
          <div>
            <span className={label}>Song covers</span>
            <p className="text-[10px] text-text-muted">A song (uploaded, or written and rendered here) sung by a voice; then a singing video of the character at a microphone.</p>
          </div>
          <button className={button} disabled={voices.length === 0} title={voices.length === 0 ? 'Create a voice first' : ''} onClick={() => toggle('cover')}><Plus size={10} /> Cover</button>
        </div>
        {forms.cover && <CoverForm bible={bible} voices={voices} onCreated={cover => { setCovers(current => [cover, ...current]); toggle('cover') }} />}
        {covers.map(cover => <CoverCard key={cover.id} bible={bible} cover={cover} ask={ask} onChange={replaceCover}
          onDeleted={next => setCovers(next.filter(c => !c.bible_id || c.bible_id === bibleId))} />)}
        {covers.length === 0 && !forms.cover && <p className="text-[10px] text-text-muted">No cover yet.</p>}
      </section>

      <section className="space-y-2">
        <div className="flex items-center justify-between gap-2">
          <div>
            <span className={label}>Persona clips</span>
            <p className="text-[10px] text-text-muted">One character, the same signature gesture, a different place each time; silent, vertical, 6 s.</p>
          </div>
          <button className={button} onClick={() => toggle('clip')}><Plus size={10} /> Clip</button>
        </div>
        {forms.clip && <ClipForm bible={bible} onCreated={clip => { setClips(current => [clip, ...current]); toggle('clip') }} />}
        {clips.map(clip => <ClipCard key={clip.id} bible={bible} clip={clip} ask={ask} onChange={replaceClip} onDeleted={setClips} />)}
        {clips.length === 0 && !forms.clip && <p className="text-[10px] text-text-muted">No clip yet.</p>}
      </section>
    </div>
  )
}
