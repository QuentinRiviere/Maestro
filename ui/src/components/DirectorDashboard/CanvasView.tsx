import { memo, useCallback, useMemo, useState, type CSSProperties, type MouseEvent } from 'react'
import {
  Background, BackgroundVariant, Controls, MiniMap, Panel, ReactFlow, ReactFlowProvider, Handle, Position,
  type Edge, type Node, type NodeProps,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import {
  BookOpen, ChevronDown, ChevronLeft, ChevronRight, ChevronsDownUp, ChevronsUpDown, Clapperboard, DollarSign,
  FileText, Film, Layers, MapPin, User,
} from 'lucide-react'
import { getFileUrl } from '../../api/client'
import type { DagNode } from '../../lib/directorDag'
import { buildDirectorDag, type CastRef } from '../../lib/directorDag'
import { loadCollapsed, saveCollapsed } from '../../lib/directorCanvasPrefs'
import { renderUnits, type RunShot, type SceneGroup } from '../../lib/directorRun'
import { StatusDot } from './runUi'
import { STATUS_BORDER, STATUS_LABEL } from './runStyles'

type FlowData = {
  node: DagNode
  onOpen?: (node: DagNode) => void
  onToggle?: (key: string) => void
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
  const clickable = !!data.onOpen && (node.shot != null || node.kind === 'final')
  const showThumb = !!node.media && node.kind !== 'prompt'
  return (
    <div
      className={`h-full w-full rounded-lg border-2 ${STATUS_BORDER[node.status]} bg-bg-secondary text-left overflow-hidden flex ${clickable ? 'cursor-pointer hover:bg-bg-hover' : ''} ${node.status === 'running' ? 'shadow-[0_0_0_3px_rgba(59,130,246,0.25)]' : ''}`}
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
  )
})

/** A collapsed scene: one row with its stats and a strip of start frames. */
const SummaryNode = memo(function SummaryNode({ data }: NodeProps<FlowNode>) {
  const node = data.node
  return (
    <div
      className={`h-full w-full rounded-lg border-2 ${STATUS_BORDER[node.status]} bg-bg-secondary overflow-hidden flex items-stretch cursor-pointer hover:bg-bg-hover ${node.status === 'running' ? 'shadow-[0_0_0_3px_rgba(59,130,246,0.25)]' : ''}`}
      title={`${node.label}\n${node.detail ?? ''}\nClick to expand`}
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
          <div key={`${media}-${i}`} className="h-full aspect-[9/16] shrink-0 rounded-sm overflow-hidden bg-media-canvas border border-border">
            <Thumb media={media} video={false} />
          </div>
        ))}
      </div>
      <Handle type="source" position={Position.Right} style={hidden} isConnectable={false} />
    </div>
  )
})

function FrameNode({ data }: NodeProps<FlowNode>) {
  const node = data.node
  return (
    <div className="h-full w-full rounded-xl border border-dashed border-border-light bg-bg-tertiary/40 pointer-events-none">
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
    <div className={`h-full w-full flex items-end gap-1 pb-1 border-b border-border text-[10px] font-semibold uppercase tracking-wider text-text-muted ${node.toggleKey ? '' : 'pointer-events-none'}`}>
      <span className="truncate">{node.label}</span>
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

const nodeTypes = { stage: StageNode, frame: FrameNode, column: ColumnNode, summary: SummaryNode }

function flowType(kind: DagNode['kind']): keyof typeof nodeTypes {
  if (kind === 'frame' || kind === 'column' || kind === 'summary') return kind
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
}

/** Left-to-right flow of the run, from inputs to the final edit. */
export default function CanvasView(props: CanvasViewProps) {
  const { runId, shots, scenes, characters, locations, brief, finalFile, runStatus, phase, costUsd, onOpenShot } = props
  const [collapsed, updateCollapsed] = useCollapsedSections(runId)

  const graph = useMemo(() => buildDirectorDag({
    shots, scenes, units: renderUnits(shots), characters, locations, brief, finalFile, runStatus, phase, costUsd,
    collapsed,
  }), [shots, scenes, characters, locations, brief, finalFile, runStatus, phase, costUsd, collapsed])

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

  const nodes: FlowNode[] = useMemo(() => graph.nodes.map(node => {
    const structural = node.kind === 'frame' || node.kind === 'column'
    return {
      id: node.id,
      type: flowType(node.kind),
      position: { x: node.x, y: node.y },
      data: { node, onOpen: structural ? undefined : onOpen, onToggle },
      style: { width: node.width, height: node.height },
      width: node.width,
      height: node.height,
      zIndex: node.kind === 'frame' ? -1 : 1,
      draggable: false,
      connectable: false,
      selectable: !structural,
      focusable: !structural,
    }
  }), [graph, onOpen, onToggle])

  const edges: Edge[] = useMemo(() => graph.edges.map(edge => ({
    id: edge.id,
    source: edge.source,
    target: edge.target,
    animated: edge.animated,
    type: 'smoothstep',
    style: edge.animated ? { stroke: 'var(--color-accent-blue)', strokeWidth: 2 } : undefined,
  })), [graph])

  const buttonClass = 'inline-flex items-center gap-1 px-2 py-1 rounded border border-border bg-bg-secondary text-[11px] text-text-secondary hover:text-text-primary hover:bg-bg-hover disabled:opacity-40 disabled:pointer-events-none'

  return (
    <div className="h-full w-full min-h-[420px] rounded-lg border border-border overflow-hidden">
      <ReactFlowProvider>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          onNodeClick={(_, node) => (node as FlowNode).data.onOpen?.((node as FlowNode).data.node)}
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
          <Panel position="top-right" className="flex gap-1.5">
            <button type="button" className={buttonClass} onClick={collapseAllScenes} disabled={allScenesCollapsed}>
              <ChevronsDownUp size={12} /> Collapse all
            </button>
            <button type="button" className={buttonClass} onClick={expandAll} disabled={!anyCollapsed}>
              <ChevronsUpDown size={12} /> Expand all
            </button>
          </Panel>
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
      </ReactFlowProvider>
    </div>
  )
}
