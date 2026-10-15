import { memo, useCallback, useEffect, useMemo, useState, type CSSProperties, type MouseEvent } from 'react'
import {
  Background, BackgroundVariant, Controls, MiniMap, NodeToolbar, Panel, ReactFlow, ReactFlowProvider, Handle, Position,
  useReactFlow, type Edge, type Node, type NodeProps,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import {
  BookOpen, ChevronDown, ChevronLeft, ChevronRight, ChevronsDownUp, ChevronsUpDown, Clapperboard, DollarSign,
  Download, FileText, Film, Layers, LayoutGrid, MapPin, MessageSquarePlus, MessagesSquare, StickyNote, User, X,
} from 'lucide-react'
import { DIRECTOR_ANNOTATIONS_EXPORT_URL, getFileUrl } from '../../api/client'
import type { DagNode } from '../../lib/directorDag'
import { buildDirectorDag, type CastRef } from '../../lib/directorDag'
import { loadCollapsed, saveCollapsed } from '../../lib/directorCanvasPrefs'
import { formatTimecode, renderUnits, type RunShot, type SceneGroup } from '../../lib/directorRun'
import {
  ANNOTATION_CATEGORIES, CATEGORY_LABEL, anchorIds, anchorNodeId, annotationsByAnchor, badgesByAnchor,
  describeTarget, filterAnnotations, hostNodeId, nodeTakesTimecode, sceneOfShot, sortAnnotations, storyboardCellId,
  targetForNode, type AnnotationCategory, type AnnotationTarget, type NodeBadge, type RunAnnotation,
} from '../../lib/directorAnnotations'
import { AnnotationForm, AnnotationList, NoteBadge } from './Annotations'
import type { AnnotationActions } from './useRunAnnotations'
import { StatusDot } from './runUi'
import { STATUS_BORDER, STATUS_LABEL } from './runStyles'

type FlowData = {
  node: DagNode
  onOpen?: (node: DagNode) => void
  onToggle?: (key: string) => void
  /** CSS aspect ratio of the run's frames. */
  aspect: string
  /** Annotate mode: a click opens a note form for the node (or storyboard cell). */
  onAnnotate?: (node: DagNode, cellShot?: number) => void
  /** Notes anchored on this node. */
  badge?: NodeBadge
  /** Notes anchored on each storyboard cell, by shot index. */
  cellBadges?: ReadonlyMap<number, NodeBadge>
  /** Lists the notes of an anchor id. */
  onBadge?: (anchorId: string) => void
}
type FlowNode = Node<FlowData>

const KIND_ICON: Partial<Record<DagNode['kind'], typeof User>> = {
  brief: FileText, character: User, location: MapPin, scene: BookOpen, planning: BookOpen,
  prompt: FileText, render: Layers, final: Clapperboard, cost: DollarSign, clip: Film, summary: BookOpen,
}

function mediaUrl(media: string): string {
  return media.startsWith('/') || media.startsWith('http') ? media : getFileUrl(media)
}

function Thumb({ media, video }: { media: string; video: boolean }) {
  const url = mediaUrl(media)
  return video
    ? <video src={`${url}#t=0.1`} muted playsInline preload="metadata" className="w-full h-full object-cover" />
    : <img src={url} alt="" loading="lazy" className="w-full h-full object-cover" />
}

const hidden: CSSProperties = { opacity: 0, width: 1, height: 1, minWidth: 0, minHeight: 0, border: 0 }

/** Chevron that toggles a collapsible section without triggering a node click. */
function ToggleButton({ toggleKey, collapsed, label, onToggle, className = '' }: {
  toggleKey: string
  collapsed: boolean
  label: string
  onToggle?: (key: string) => void
  className?: string
}) {
  const Icon = collapsed ? ChevronRight : ChevronDown
  const title = `${collapsed ? 'Expand' : 'Collapse'} ${label}`
  return (
    <button
      type="button"
      className={`nodrag nopan pointer-events-auto inline-flex items-center justify-center rounded text-text-muted hover:text-text-primary hover:bg-bg-hover ${className}`}
      title={title}
      aria-label={title}
      aria-expanded={!collapsed}
      onClick={(event: MouseEvent) => { event.stopPropagation(); onToggle?.(toggleKey) }}
    >
      <Icon size={13} />
    </button>
  )
}

const StageNode = memo(function StageNode({ data }: NodeProps<FlowNode>) {
  const node = data.node
  const Icon = KIND_ICON[node.kind]
  const isVideo = node.kind === 'clip' || node.kind === 'final'
  const clickable = !!data.onAnnotate || (!!data.onOpen && (node.shot != null || node.kind === 'final'))
  const showThumb = !!node.media && node.kind !== 'prompt'
  return (
    <div className="relative h-full w-full">
    <div
      className={`h-full w-full rounded-lg border-2 ${STATUS_BORDER[node.status]} bg-bg-secondary text-left overflow-hidden flex ${clickable ? `${data.onAnnotate ? 'cursor-copy' : 'cursor-pointer'} hover:bg-bg-hover` : ''} ${node.status === 'running' ? 'shadow-[0_0_0_3px_rgba(59,130,246,0.25)]' : ''}`}
      title={node.detail ? `${node.label}\n${node.detail}` : node.label}
    >
      <Handle type="target" position={Position.Left} style={hidden} isConnectable={false} />
      {showThumb && (
        <div className={`${node.kind === 'final' ? 'w-[84px]' : node.kind === 'character' || node.kind === 'location' ? 'w-[48px]' : 'w-[62px]'} shrink-0 bg-media-canvas border-r border-border`}>
          <Thumb media={node.media!} video={isVideo} />
        </div>
      )}
      <div className="min-w-0 flex-1 p-2 flex flex-col gap-1">
        <div className="flex items-center gap-1.5 min-w-0">
          {Icon && <Icon size={11} className="text-text-muted shrink-0" />}
          <span className="text-[11px] font-semibold text-text-primary truncate">{node.label}</span>
          <span className="ml-auto"><StatusDot status={node.status} /></span>
        </div>
        {node.detail && (
          <p className={`text-[10px] text-text-secondary ${node.kind === 'brief' ? 'line-clamp-3' : 'line-clamp-4'} break-words`}>{node.detail}</p>
        )}
        {node.kind !== 'brief' && node.kind !== 'cost' && (
          <span className="mt-auto text-[9px] uppercase tracking-wide text-text-muted">{STATUS_LABEL[node.status]}</span>
        )}
      </div>
      <Handle type="source" position={Position.Right} style={hidden} isConnectable={false} />
    </div>
    <NodeBadgeSlot data={data} />
    </div>
  )
})

/** Note count pinned to a node's top-right corner. */
function NodeBadgeSlot({ data }: { data: FlowData }) {
  if (!data.badge) return null
  return (
    <NoteBadge
      badge={data.badge}
      onClick={() => data.onBadge?.(data.node.id)}
      className="absolute -top-2 -right-2 z-10"
    />
  )
}

/** A collapsed scene: one row with its stats and a strip of start frames. */
const SummaryNode = memo(function SummaryNode({ data }: NodeProps<FlowNode>) {
  const node = data.node
  return (
    <div className="relative h-full w-full">
    <div
      className={`h-full w-full rounded-lg border-2 ${STATUS_BORDER[node.status]} bg-bg-secondary overflow-hidden flex items-stretch ${data.onAnnotate ? 'cursor-copy' : 'cursor-pointer'} hover:bg-bg-hover ${node.status === 'running' ? 'shadow-[0_0_0_3px_rgba(59,130,246,0.25)]' : ''}`}
      title={`${node.label}\n${node.detail ?? ''}\n${data.onAnnotate ? 'Click to annotate this scene' : 'Click to expand'}`}
    >
      <Handle type="target" position={Position.Left} style={hidden} isConnectable={false} />
      <div className="w-[186px] shrink-0 p-2 flex flex-col gap-1 min-w-0">
        <div className="flex items-center gap-1.5 min-w-0">
          <BookOpen size={11} className="text-text-muted shrink-0" />
          <span className="text-[11px] font-semibold text-text-primary truncate">{node.label}</span>
          <span className="ml-auto"><StatusDot status={node.status} /></span>
        </div>
        {node.detail && <p className="text-[10px] text-text-secondary line-clamp-3">{node.detail}</p>}
        <span className="mt-auto text-[9px] uppercase tracking-wide text-text-muted">collapsed scene</span>
      </div>
      <div className="flex-1 min-w-0 flex gap-1 py-1.5 pr-1.5 overflow-hidden">
        {(node.thumbs || []).map((media, i) => (
          <div key={`${media}-${i}`} className="h-full shrink-0 rounded-sm overflow-hidden bg-media-canvas border border-border" style={{ aspectRatio: data.aspect }}>
            <Thumb media={media} video={false} />
          </div>
        ))}
      </div>
      <Handle type="source" position={Position.Right} style={hidden} isConnectable={false} />
    </div>
    <NodeBadgeSlot data={data} />
    </div>
  )
})

/**
 * The whole cut as a dense grid of start frames: no gap, each cell tagged
 * with its shot number and start time. Clicking a cell opens that shot.
 */
const StoryboardNode = memo(function StoryboardNode({ data }: NodeProps<FlowNode>) {
  const node = data.node
  const cells = node.cells || []
  return (
    <div className={`h-full w-full rounded-lg border-2 ${STATUS_BORDER[node.status]} bg-bg-secondary overflow-hidden flex flex-col`}>
      <div className="h-[30px] shrink-0 px-2 flex items-center gap-1.5 min-w-0">
        {node.toggleKey && (
          <ToggleButton
            toggleKey={node.toggleKey}
            collapsed={!!node.collapsed}
            label="storyboard"
            onToggle={data.onToggle}
            className="w-5 h-5"
          />
        )}
        <LayoutGrid size={11} className="text-text-muted shrink-0" />
        <span className="text-[11px] font-semibold text-text-primary">{node.label}</span>
        {node.detail && <span className="text-[10px] text-text-muted truncate">{node.detail}</span>}
        {data.badge && <NoteBadge badge={data.badge} onClick={() => data.onBadge?.(node.id)} className="ml-1" />}
        <span className="ml-auto"><StatusDot status={node.status} /></span>
      </div>
      {!node.collapsed && (
        <div
          className="nodrag grid gap-0 bg-media-canvas"
          style={{ gridTemplateColumns: `repeat(${node.gridColumns || 6}, minmax(0, 1fr))` }}
        >
          {cells.map(cell => {
            const label = `#${cell.shot + 1}`
            const time = formatTimecode(cell.startSec)
            const cellBadge = data.cellBadges?.get(cell.shot)
            return (
              <div key={cell.shot} className="relative w-full" style={{ aspectRatio: data.aspect }}>
              <button
                type="button"
                className={`nodrag absolute inset-0 block w-full h-full overflow-hidden bg-bg-tertiary ${data.onAnnotate ? 'cursor-copy' : 'cursor-pointer'} focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent-blue focus-visible:-outline-offset-2 hover:brightness-110`}
                title={`Shot ${cell.shot + 1} at ${time} (${STATUS_LABEL[cell.status]})`}
                aria-label={data.onAnnotate
                  ? `Annotate shot ${cell.shot + 1}, starts at ${time}`
                  : `Open shot ${cell.shot + 1}, starts at ${time}`}
                onClick={event => {
                  event.stopPropagation()
                  if (data.onAnnotate) data.onAnnotate(node, cell.shot)
                  else data.onOpen?.({ ...node, shot: cell.shot })
                }}
              >
                {cell.media
                  ? <img src={mediaUrl(cell.media)} alt="" loading="lazy" className="absolute inset-0 w-full h-full object-cover" />
                  : cell.status === 'running' && <span className="absolute inset-0 animate-pulse bg-accent-blue/10" />}
                <span className="absolute left-0 right-0 bottom-0 px-1 py-0.5 flex justify-between gap-1 bg-gradient-to-t from-black/70 to-transparent text-[9px] leading-none font-mono text-white [text-shadow:0_1px_1px_rgba(0,0,0,0.8)]">
                  <span>{label}</span>
                  <span>{time}</span>
                </span>
              </button>
              {cellBadge && (
                <NoteBadge
                  small
                  badge={cellBadge}
                  onClick={() => data.onBadge?.(storyboardCellId(cell.shot))}
                  className="absolute top-0.5 right-0.5"
                />
              )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
})

function FrameNode({ data }: NodeProps<FlowNode>) {
  const node = data.node
  return (
    <div className={`h-full w-full rounded-xl border border-dashed bg-bg-tertiary/40 pointer-events-none ${data.onAnnotate ? 'border-accent-blue/50' : 'border-border-light'}`}>
      <div className="px-2 py-1.5 text-[11px] font-semibold text-text-secondary flex items-center gap-1.5">
        {node.toggleKey && (
          <ToggleButton
            toggleKey={node.toggleKey}
            collapsed={!!node.collapsed}
            label={node.label}
            onToggle={data.onToggle}
            className="w-5 h-5"
          />
        )}
        {node.label}
        {node.detail && <span className="font-mono font-normal text-text-muted">{node.detail}</span>}
        {data.badge && <NoteBadge badge={data.badge} onClick={() => data.onBadge?.(node.id)} />}
      </div>
    </div>
  )
}

function ColumnNode({ data }: NodeProps<FlowNode>) {
  const node = data.node
  if (node.toggleKey && node.collapsed) {
    const title = `Show ${node.label}`
    return (
      <div className="h-full w-full flex items-end pb-1 border-b border-border">
        <button
          type="button"
          className="nodrag nopan w-full h-6 inline-flex items-center justify-center rounded text-text-muted hover:text-text-primary hover:bg-bg-hover"
          title={title}
          aria-label={title}
          aria-expanded={false}
          onClick={event => { event.stopPropagation(); data.onToggle?.(node.toggleKey!) }}
        >
          <ChevronRight size={13} />
        </button>
      </div>
    )
  }
  return (
    <div className={`h-full w-full flex items-end gap-1 pb-1 border-b border-border text-[10px] font-semibold uppercase tracking-wider text-text-muted ${data.onAnnotate ? 'cursor-copy hover:text-text-primary' : node.toggleKey ? '' : 'pointer-events-none'}`}>
      <span className="truncate">{node.label}</span>
      {data.badge && <NoteBadge badge={data.badge} onClick={() => data.onBadge?.(node.id)} />}
      {node.toggleKey && (
        <button
          type="button"
          className="nodrag nopan ml-auto w-5 h-5 inline-flex items-center justify-center rounded hover:text-text-primary hover:bg-bg-hover"
          title={`Hide ${node.label}`}
          aria-label={`Hide ${node.label}`}
          aria-expanded
          onClick={event => { event.stopPropagation(); data.onToggle?.(node.toggleKey!) }}
        >
          <ChevronLeft size={13} />
        </button>
      )}
    </div>
  )
}

const nodeTypes = {
  stage: StageNode, frame: FrameNode, column: ColumnNode, summary: SummaryNode, storyboard: StoryboardNode,
}

function flowType(kind: DagNode['kind']): keyof typeof nodeTypes {
  if (kind === 'frame' || kind === 'column' || kind === 'summary' || kind === 'storyboard') return kind
  return 'stage'
}

const flowStyle = {
  '--xy-background-color': 'var(--color-bg-primary)',
  '--xy-background-pattern-color': 'var(--color-border)',
  '--xy-edge-stroke': 'var(--color-border-light)',
  '--xy-edge-stroke-selected': 'var(--color-accent-blue)',
  '--xy-controls-button-background-color': 'var(--color-bg-secondary)',
  '--xy-controls-button-background-color-hover': 'var(--color-bg-hover)',
  '--xy-controls-button-color': 'var(--color-text-primary)',
  '--xy-controls-button-color-hover': 'var(--color-text-primary)',
  '--xy-controls-button-border-color': 'var(--color-border)',
  '--xy-minimap-background-color': 'var(--color-bg-secondary)',
  '--xy-minimap-mask-background-color': 'rgba(0, 0, 0, 0.35)',
  '--xy-attribution-background-color': 'transparent',
} as CSSProperties

const MINIMAP_COLOR: Record<DagNode['status'], string> = {
  pending: '#6b7280', running: '#3b82f6', done: '#22c55e', failed: '#f87171', flagged: '#fbbf24', skipped: '#4b5563',
}

/**
 * Collapsed canvas sections of one run, persisted per viewer. The state is
 * keyed by run id so it survives every live re-layout of the same run.
 */
function useCollapsedSections(runId: string) {
  const [state, setState] = useState(() => ({ runId, keys: loadCollapsed(runId) as ReadonlySet<string> }))
  let current = state
  if (state.runId !== runId) {
    current = { runId, keys: loadCollapsed(runId) }
    setState(current)
  }
  const update = useCallback((next: (keys: ReadonlySet<string>) => ReadonlySet<string>) => {
    setState(prev => {
      const keys = next(prev.keys)
      saveCollapsed(prev.runId, keys)
      return { runId: prev.runId, keys }
    })
  }, [])
  return [current.keys, update] as const
}

export interface CanvasViewProps {
  runId: string
  /** CSS aspect ratio of the run's frames ("9 / 16"). */
  aspect: string
  shots: RunShot[]
  scenes: SceneGroup[]
  characters: CastRef[]
  locations: CastRef[]
  brief: string
  finalFile: string | null
  runStatus: string
  phase: string
  costUsd: number | null
  onOpenShot: (index: number) => void
  /** Post-analysis notes of the run; omit the actions to hide annotation tools. */
  annotations?: RunAnnotation[]
  annotationActions?: AnnotationActions
}

/** Note popover anchored to a canvas node: a new note, or the node's notes. */
type NotePopover =
  | { anchor: string; mode: 'new'; target: AnnotationTarget; allowTimecode: boolean; title: string }
  | { anchor: string; mode: 'list' }

const NO_ANNOTATIONS: RunAnnotation[] = []

function popoverTitle(node: DagNode, target: AnnotationTarget, scenes: SceneGroup[]): string {
  if (target.kind === 'shot' || target.kind === 'scene' || target.kind === 'stage') return describeTarget(target, scenes)
  return node.label
}

/** Left-to-right flow of the run, from inputs to the final edit. */
export default function CanvasView(props: CanvasViewProps) {
  return (
    <ReactFlowProvider>
      <CanvasBody {...props} />
    </ReactFlowProvider>
  )
}

function CanvasBody(props: CanvasViewProps) {
  const { runId, aspect, shots, scenes, characters, locations, brief, finalFile, runStatus, phase, costUsd, onOpenShot } = props
  const annotations = props.annotations ?? NO_ANNOTATIONS
  const annotationActions = props.annotationActions
  const [collapsed, updateCollapsed] = useCollapsedSections(runId)
  const [annotating, setAnnotating] = useState(false)
  const [popover, setPopover] = useState<NotePopover | null>(null)
  const [panelOpen, setPanelOpen] = useState(false)
  const [runNoteOpen, setRunNoteOpen] = useState(false)
  const [categoryFilter, setCategoryFilter] = useState<AnnotationCategory | 'all'>('all')
  const [showResolved, setShowResolved] = useState(false)
  const { fitView } = useReactFlow()

  const graph = useMemo(() => buildDirectorDag({
    shots, scenes, units: renderUnits(shots), characters, locations, brief, finalFile, runStatus, phase, costUsd,
    collapsed, aspect,
  }), [shots, scenes, characters, locations, brief, finalFile, runStatus, phase, costUsd, collapsed, aspect])

  const nodeById = useMemo(() => new Map(graph.nodes.map(node => [node.id, node])), [graph])
  const visibleAnchors = useMemo(() => anchorIds(graph.nodes), [graph])
  const shotScenes = useMemo(() => sceneOfShot(scenes), [scenes])
  const notesByAnchor = useMemo(
    () => annotationsByAnchor(annotations, graph.nodes, scenes), [annotations, graph, scenes],
  )
  const badges = useMemo(() => badgesByAnchor(notesByAnchor), [notesByAnchor])
  const cellBadges = useMemo(() => {
    const map = new Map<number, NodeBadge>()
    for (const [anchor, badge] of badges) {
      if (anchor.startsWith('storyboard:')) map.set(Number(anchor.slice('storyboard:'.length)), badge)
    }
    return map
  }, [badges])
  const openCount = useMemo(() => annotations.filter(a => !a.resolved).length, [annotations])

  const onToggle = useCallback((key: string) => updateCollapsed(keys => {
    const next = new Set(keys)
    if (next.has(key)) next.delete(key)
    else next.add(key)
    return next
  }), [updateCollapsed])

  const sceneKeys = useMemo(() => scenes.map(scene => scene.key), [scenes])
  const allScenesCollapsed = sceneKeys.length > 0 && sceneKeys.every(key => collapsed.has(key))
  const anyCollapsed = collapsed.size > 0
  const collapseAllScenes = () => updateCollapsed(keys => new Set([...keys, ...sceneKeys]))
  const expandAll = () => updateCollapsed(() => new Set())

  const onOpen = useMemo(() => (node: DagNode) => {
    if (node.kind === 'summary' && node.toggleKey) onToggle(node.toggleKey)
    else if (node.kind === 'final' && node.media) window.open(mediaUrl(node.media), '_blank', 'noopener')
    else if (node.shot != null) onOpenShot(node.shot)
  }, [onOpenShot, onToggle])

  const startNote = useCallback((node: DagNode, cellShot?: number) => {
    const target = targetForNode(node, cellShot)
    setPopover({
      anchor: cellShot != null ? storyboardCellId(cellShot) : node.id,
      mode: 'new',
      target,
      allowTimecode: cellShot == null && nodeTakesTimecode(node),
      title: popoverTitle(node, target, scenes),
    })
  }, [scenes])

  const onAnnotate = annotating && annotationActions ? startNote : undefined
  const onBadge = useCallback((anchor: string) => setPopover({ anchor, mode: 'list' }), [])

  const nodes: FlowNode[] = useMemo(() => graph.nodes.map(node => {
    const structural = node.kind === 'frame' || node.kind === 'column'
    return {
      id: node.id,
      type: flowType(node.kind),
      position: { x: node.x, y: node.y },
      data: {
        node, onOpen: structural ? undefined : onOpen, onToggle, aspect,
        onAnnotate, badge: badges.get(node.id), onBadge,
        cellBadges: node.kind === 'storyboard' ? cellBadges : undefined,
      },
      style: { width: node.width, height: node.height },
      width: node.width,
      height: node.height,
      zIndex: node.kind === 'frame' ? -1 : 1,
      draggable: false,
      connectable: false,
      selectable: !structural,
      focusable: !structural,
    }
  }), [graph, onOpen, onToggle, aspect, onAnnotate, badges, onBadge, cellBadges])

  const edges: Edge[] = useMemo(() => graph.edges.map(edge => ({
    id: edge.id,
    source: edge.source,
    target: edge.target,
    animated: edge.animated,
    type: 'smoothstep',
    style: edge.animated ? { stroke: 'var(--color-accent-blue)', strokeWidth: 2 } : undefined,
  })), [graph])

  // Esc closes the innermost layer: the popover, the run-note form, then annotate mode.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return
      if (popover) setPopover(null)
      else if (runNoteOpen) setRunNoteOpen(false)
      else if (annotating) setAnnotating(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [popover, runNoteOpen, annotating])

  const focusAnnotation = (annotation: RunAnnotation) => {
    const anchor = anchorNodeId(annotation.target, visibleAnchors, shotScenes)
    if (!anchor) return
    void fitView({ nodes: [{ id: hostNodeId(anchor) }], duration: 400, maxZoom: 1.2, padding: 0.6 })
    setPopover({ anchor, mode: 'list' })
  }

  const popoverHost = popover ? nodeById.get(hostNodeId(popover.anchor)) : undefined
  const popoverNotes = popover?.mode === 'list' ? sortAnnotations(notesByAnchor.get(popover.anchor) || []) : []
  const addFromList = () => {
    if (!popover || !popoverHost) return
    const cellShot = popover.anchor.startsWith('storyboard:') ? Number(popover.anchor.slice('storyboard:'.length)) : undefined
    startNote(popoverHost, cellShot)
  }
  const panelNotes = filterAnnotations(annotations, { category: categoryFilter, showResolved })

  const buttonClass = 'inline-flex items-center gap-1 px-2 py-1 rounded border border-border bg-bg-secondary text-[11px] text-text-secondary hover:text-text-primary hover:bg-bg-hover disabled:opacity-40 disabled:pointer-events-none'
  const activeButtonClass = 'inline-flex items-center gap-1 px-2 py-1 rounded border border-accent-blue bg-accent-blue/15 text-[11px] text-accent-blue hover:bg-accent-blue/25'

  return (
    <div className="h-full w-full min-h-[420px] rounded-lg border border-border overflow-hidden flex">
      <div className="relative flex-1 min-w-0 h-full">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          onNodeClick={(_, node) => {
            const data = (node as FlowNode).data
            if (data.onAnnotate) data.onAnnotate(data.node)
            else data.onOpen?.(data.node)
          }}
          nodesDraggable={false}
          nodesConnectable={false}
          edgesFocusable={false}
          fitView
          fitViewOptions={{ padding: 0.08, maxZoom: 1 }}
          minZoom={0.1}
          maxZoom={2}
          proOptions={{ hideAttribution: true }}
          style={flowStyle}
        >
          <Background variant={BackgroundVariant.Dots} gap={20} size={1} />
          <Panel position="top-right" className="flex flex-wrap justify-end gap-1.5">
            {annotationActions && (
              <>
                <button
                  type="button"
                  className={annotating ? activeButtonClass : buttonClass}
                  onClick={() => { setAnnotating(value => !value); setPopover(null) }}
                  aria-pressed={annotating}
                  title={annotating ? 'Leave annotate mode (Esc)' : 'Click any node to pin a post-analysis note on it'}
                >
                  <MessageSquarePlus size={12} /> Annotate
                </button>
                <button
                  type="button"
                  className={buttonClass}
                  onClick={() => { setPanelOpen(true); setRunNoteOpen(true) }}
                  title="Add a note about the whole run"
                >
                  <StickyNote size={12} /> Add note
                </button>
                <button
                  type="button"
                  className={panelOpen ? activeButtonClass : buttonClass}
                  onClick={() => setPanelOpen(value => !value)}
                  aria-pressed={panelOpen}
                  title="All notes of this run"
                >
                  <MessagesSquare size={12} /> Notes{annotations.length ? ` (${openCount}/${annotations.length})` : ''}
                </button>
              </>
            )}
            <button type="button" className={buttonClass} onClick={collapseAllScenes} disabled={allScenesCollapsed}>
              <ChevronsDownUp size={12} /> Collapse all
            </button>
            <button type="button" className={buttonClass} onClick={expandAll} disabled={!anyCollapsed}>
              <ChevronsUpDown size={12} /> Expand all
            </button>
          </Panel>
          {annotating && (
            <Panel position="top-left" className="rounded border border-accent-blue/40 bg-bg-secondary/95 px-2 py-1 text-[11px] text-accent-blue">
              Annotate mode: click a node, a scene frame, a column title or a storyboard frame · Esc to leave
            </Panel>
          )}
          {popover && popoverHost && annotationActions && (
            <NodeToolbar nodeId={popoverHost.id} isVisible position={Position.Bottom} align="start" offset={8}>
              <div
                className="nodrag nopan nowheel w-[320px] max-h-[420px] overflow-y-auto rounded-lg border border-border bg-bg-secondary shadow-2xl p-3 space-y-2 text-left"
                role="dialog"
                aria-label={popover.mode === 'new' ? 'New note' : 'Notes'}
                onClick={event => event.stopPropagation()}
                onPointerDown={event => event.stopPropagation()}
              >
                <div className="flex items-center gap-2">
                  <h3 className="text-[11px] font-semibold text-text-primary truncate">
                    {popover.mode === 'new'
                      ? `New note · ${popover.title}`
                      : `Notes · ${popoverNotes[0] ? describeTarget(popoverNotes[0].target, scenes) : popoverHost.label}`}
                  </h3>
                  <button
                    type="button"
                    className="ml-auto p-0.5 rounded text-text-muted hover:text-text-primary hover:bg-bg-hover"
                    aria-label="Close"
                    onClick={() => setPopover(null)}
                  ><X size={13} /></button>
                </div>
                {popover.mode === 'new' ? (
                  <AnnotationForm
                    key={`${popover.anchor}-new`}
                    target={popover.target}
                    allowTimecode={popover.allowTimecode}
                    onCancel={() => setPopover(null)}
                    onSubmit={async draft => {
                      await annotationActions.create(draft)
                      setPopover(null)
                    }}
                  />
                ) : (
                  <>
                    <AnnotationList
                      annotations={popoverNotes}
                      actions={annotationActions}
                      allowTimecode={nodeTakesTimecode(popoverHost)}
                      showTarget
                      scenes={scenes}
                      empty="No notes here any more."
                    />
                    <button type="button" className={buttonClass} onClick={addFromList}>
                      <MessageSquarePlus size={12} /> Add note
                    </button>
                  </>
                )}
              </div>
            </NodeToolbar>
          )}
          <Controls showInteractive={false} />
          <MiniMap
            pannable
            zoomable
            nodeColor={node => {
              const data = (node as FlowNode).data.node
              return data.kind === 'frame' || data.kind === 'column' ? 'transparent' : MINIMAP_COLOR[data.status]
            }}
            nodeStrokeWidth={0}
          />
        </ReactFlow>
      </div>
      {panelOpen && annotationActions && (
        <aside className="w-[300px] max-w-[45%] shrink-0 h-full border-l border-border bg-bg-secondary flex flex-col" aria-label="Run notes">
          <header className="px-3 py-2 border-b border-border flex items-center gap-2 shrink-0">
            <MessagesSquare size={13} className="text-text-muted" />
            <h3 className="text-xs font-semibold text-text-primary">Notes</h3>
            <span className="text-[10px] text-text-muted">{openCount} open · {annotations.length} total</span>
            <button
              type="button"
              className="ml-auto p-0.5 rounded text-text-muted hover:text-text-primary hover:bg-bg-hover"
              aria-label="Close notes"
              onClick={() => { setPanelOpen(false); setRunNoteOpen(false) }}
            ><X size={13} /></button>
          </header>
          <div className="px-3 py-2 border-b border-border flex flex-wrap items-center gap-2 shrink-0">
            <select
              value={categoryFilter}
              onChange={event => setCategoryFilter(event.target.value as AnnotationCategory | 'all')}
              aria-label="Filter by category"
              className="bg-bg-tertiary border border-border rounded px-1.5 py-0.5 text-[11px] text-text-primary"
            >
              <option value="all">All categories</option>
              {ANNOTATION_CATEGORIES.map(value => <option key={value} value={value}>{CATEGORY_LABEL[value]}</option>)}
            </select>
            <label className="flex items-center gap-1 text-[11px] text-text-secondary">
              <input type="checkbox" checked={showResolved} onChange={event => setShowResolved(event.target.checked)} />
              Show resolved
            </label>
          </div>
          <div className="flex-1 min-h-0 overflow-y-auto p-3 space-y-3">
            {runNoteOpen ? (
              <div className="rounded border border-accent-blue/40 p-2 space-y-1.5">
                <div className="text-[10px] uppercase tracking-wide text-text-muted">Note on the whole run</div>
                <AnnotationForm
                  target={{ kind: 'run', ref: '' }}
                  allowTimecode
                  onCancel={() => setRunNoteOpen(false)}
                  onSubmit={async draft => {
                    await annotationActions.create(draft)
                    setRunNoteOpen(false)
                  }}
                />
              </div>
            ) : (
              <button type="button" className={buttonClass} onClick={() => setRunNoteOpen(true)}>
                <StickyNote size={12} /> Add a run note
              </button>
            )}
            <AnnotationList
              annotations={panelNotes}
              actions={annotationActions}
              allowTimecode={false}
              showTarget
              scenes={scenes}
              onFocus={focusAnnotation}
              empty={annotations.length
                ? 'No note matches this filter.'
                : 'No notes yet. Turn on Annotate and click a node, or add a run note.'}
            />
          </div>
          <footer className="px-3 py-2 border-t border-border shrink-0">
            <a
              href={DIRECTOR_ANNOTATIONS_EXPORT_URL}
              download
              className="inline-flex items-center gap-1 text-[11px] text-text-secondary hover:text-accent-blue"
              title="Every run's notes, with prompt excerpts, as one JSON file"
            >
              <Download size={12} /> Export all runs' notes (JSON)
            </a>
          </footer>
        </aside>
      )}
    </div>
  )
}
