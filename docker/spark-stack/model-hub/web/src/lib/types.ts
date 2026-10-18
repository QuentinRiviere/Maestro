// Response shapes of the hub's API, as app.py returns them.

export interface Budget {
  total_gb: number;
  reserved_gb: number;
  resident_gb: number;
  available_gb: number;
  resident_available_gb: number;
  swap_reserve_gb: number;
  resident: string[];
}

export interface Status {
  disk: { free: number; total: number };
  budget: Budget;
  groups: string[];
  default_group: string;
  base_error: string;
  trusted_authors: string[];
  hf_token: boolean;
  llama_swap: boolean;
}

export type Fit = "fits" | "tight" | "too big";

export interface Serve {
  id: string;
  name?: string;
  model: string;
  mmproj?: string;
  ctx_size: number;
  parallel: number;
  cache_type_k?: string;
  cache_type_v?: string;
  ttl: number;
  group?: string;
  aliases: string[];
  extra_args: string[];
  mtp: boolean;
  mtp_draft_n_max?: number;
  preload: boolean;
  enabled: boolean;
}

export interface CatalogItem {
  name: string;
  repo: string;
  revision: string;
  category: string;
  size_gb: number;
  notes: string;
  serve?: Partial<Serve> & { model: string };
  installed: boolean;
  in_models_yml: boolean;
  memory_gb: number;
  kv_source: string;
  resident: boolean;
  fit: Fit;
}

export interface SearchResult {
  repo: string;
  author: string;
  downloads: number | null;
  likes: number | null;
  last_modified: string | null;
  gated: boolean;
  trusted: boolean;
}

export interface GgufMemory {
  weights: number;
  mmproj: number;
  kv: number;
  overhead: number;
  total: number;
  kv_source: string;
}

export interface GgufGroup {
  key: string;
  first: string;
  files: string[];
  parts: number;
  complete: boolean;
  quant: string;
  size: number;
  kind: "model" | "mmproj" | "mtp" | "draft" | string;
  installed: boolean;
  memory?: GgufMemory;
  fit?: Fit;
}

export interface ArchInfo {
  architecture: string;
  block_count: number;
  head_count_kv: number | number[];
  key_length: number;
  value_length: number;
  expert_count?: number;
  sliding_window?: number;
  full_attention_interval?: number;
  context_length?: number;
}

export interface RepoDetail {
  repo: string;
  commit: string;
  revision: string;
  pinned: boolean;
  last_modified: string | null;
  gated: boolean;
  license: string | null;
  tags: string[];
  downloads: number | null;
  likes: number | null;
  trusted: boolean;
  ggufs: GgufGroup[];
  default_mmproj: string | null;
  mtp: { mtp: boolean; draft_n_max?: number | null };
  refused: string[];
  ctx: number;
  budget: Budget;
  in_models_yml: boolean;
  arch: ArchInfo | null;
  kv_source: "metadata" | "heuristic";
  resident: boolean;
}

export interface ContextReport {
  max: { ctx: number; fits: boolean; limited_by: string; per_slot: number; kv_gb: number };
  kv_source: string;
  budget_gb: number;
  parallel: number;
  kv_per_token: number;
  resident: boolean;
  estimate: { ctx: number; weights: number; mmproj: number; kv: number; overhead: number; total: number };
}

export interface PreviewResult {
  error?: string;
  warnings?: string[];
  diff?: string;
  changed?: boolean;
}

export interface Job {
  id: string;
  repo: string;
  state: "queued" | "running" | "done" | "cancelled" | "failed" | string;
  phase?: string;
  files: string[];
  current?: string | null;
  done_bytes: number;
  total_bytes: number;
  rate?: number | null;
  eta_seconds?: number | null;
  message?: string | null;
  error?: string | null;
}

export interface InstalledFile {
  name: string;
  size: number;
  present: boolean;
  commit?: string;
}

export interface VerifyState {
  state: "running" | "ok" | "mismatch" | "error" | string;
  files?: Record<string, boolean>;
  error?: string;
}

export interface InstalledRepo {
  repo: string;
  files: InstalledFile[];
  bytes: number;
  commit?: string | null;
  in_models_yml: boolean;
  serve: Serve | null;
  mapped: boolean;
  verify: VerifyState | null;
  downloading: boolean;
}

export interface ConfigResponse {
  rendered: string;
  pending_diff: string | null;
  warnings: string[] | null;
  error: string;
  models_yml: string;
}

export interface WorkerSummary {
  reachable: boolean;
  error?: string;
  paused?: boolean;
  loaded_models?: string[];
  queued?: number | null;
  running_job?: { id?: string | null; phase?: string | null } | null;
  idle_unload_seconds?: number | null;
  gpu_free_gb?: number;
  gpu_total_gb?: number;
  cuda_free_gb?: number;
  unified_memory?: boolean;
  process_rss_gb?: number;
}

export interface MemoryModel {
  model: string;
  state: string;
  ttl?: number | null;
  resident: boolean;
  estimated_gb?: number;
  worker?: WorkerSummary;
}

export interface MemoryHost {
  total_gb?: number;
  available_gb?: number;
  used_gb?: number;
  cache_gb?: number;
  shmem_gb?: number;
  swap_total_gb?: number;
  swap_used_gb?: number;
}

export interface MemoryView {
  host: MemoryHost;
  llama_swap: { reachable: boolean; error: string; models: MemoryModel[] };
  worker: WorkerSummary;
  accounted_gb: number;
  other_gb: number | null;
}
