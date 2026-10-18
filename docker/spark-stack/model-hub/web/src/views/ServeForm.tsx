// Serve settings editor (llama-swap entry). Controlled: the parent owns the
// state and turns it into the `serve` payload with `serveValue()`. The form
// asks /api/context for the largest --ctx-size that fits whenever a field
// that changes the KV cache or the budget changes.
import { useEffect, useRef, useState, type ReactNode } from "react";
import { Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { api, ApiError } from "@/lib/api";
import { num } from "@/lib/format";
import type { ContextReport, Serve, Status } from "@/lib/types";

export const CACHE_TYPES = ["f16", "q8_0", "q4_0"];
const NONE = "__none__";

export interface FileOption {
  key: string;
  first: string;
  files: string[];
}

export interface ServeFormState {
  id: string;
  name: string;
  model: string;
  mmproj: string;
  ctx_size: string;
  parallel: string;
  cache_type_k: string;
  cache_type_v: string;
  ttl: string;
  group: string;
  aliases: string;
  extra_args: string;
  mtp: boolean;
  mtp_draft_n_max: string;
  preload: boolean;
  enabled: boolean;
}

export function initServeState(
  serve: Partial<Serve> | null | undefined,
  models: FileOption[],
  mmprojs: FileOption[],
  defaultMmproj: string,
  status: Status | null,
): ServeFormState {
  const s = { parallel: 1, ttl: 600, ctx_size: 32768, mtp: false, mtp_draft_n_max: 6, preload: false, enabled: true, aliases: [], extra_args: [], ...(serve || {}) };
  const group = s.group || status?.default_group || "on-demand";
  // Like a native <select>, a file that is not among the options falls back (model: first option, mmproj: none).
  const model = s.model && models.some((g) => g.first === s.model) ? s.model : models[0]?.first || "";
  const wantedMmproj = s.mmproj !== undefined ? s.mmproj || "" : defaultMmproj || "";
  return {
    id: s.id || "",
    name: s.name || "",
    model,
    mmproj: mmprojs.some((g) => g.first === wantedMmproj) ? wantedMmproj : "",
    ctx_size: String(s.ctx_size),
    parallel: String(s.parallel),
    cache_type_k: s.cache_type_k || "f16",
    cache_type_v: s.cache_type_v || "f16",
    ttl: String(s.ttl),
    group,
    aliases: (s.aliases || []).join(", "),
    extra_args: (s.extra_args || []).join(" "),
    mtp: !!s.mtp,
    mtp_draft_n_max: String(s.mtp_draft_n_max || 6),
    preload: !!s.preload,
    enabled: s.enabled !== false,
  };
}

/** The `serve` payload sent to /api/downloads, /api/serve and /api/serve/preview. */
export function serveValue(f: ServeFormState): Serve {
  const split = (v: string, re: RegExp) => v.split(re).map((x) => x.trim()).filter(Boolean);
  const out: Serve = {
    id: f.id.trim(),
    model: f.model,
    ctx_size: Number(f.ctx_size),
    parallel: Number(f.parallel),
    ttl: Number(f.ttl),
    group: f.group.trim(),
    aliases: split(f.aliases, /,/),
    extra_args: split(f.extra_args, /\s+/),
    mtp: f.mtp,
    preload: f.preload,
    enabled: f.enabled,
  };
  if (f.name.trim()) out.name = f.name.trim();
  if (f.mmproj) out.mmproj = f.mmproj;
  if (f.cache_type_k !== "f16") out.cache_type_k = f.cache_type_k;
  if (f.cache_type_v !== "f16") out.cache_type_v = f.cache_type_v;
  if (out.mtp) out.mtp_draft_n_max = Number(f.mtp_draft_n_max);
  return out;
}

/** Selecting a model file also suggests the llama-swap id when it is still empty. */
export function withModel(f: ServeFormState, first: string, keepId = false): ServeFormState {
  const id = !keepId && !f.id ? first.replace(/\.gguf$/i, "").toLowerCase() : f.id;
  return { ...f, model: first, id };
}

interface Props {
  repo: string;
  revision: string;
  models: FileOption[];
  mmprojs: FileOption[];
  state: ServeFormState;
  onChange: (next: ServeFormState) => void;
  status: Status | null;
}

function Field({ label, title, children }: { label: string; title?: string; children: ReactNode }) {
  return (
    <div className="grid gap-1" title={title}>
      <Label className="text-xs text-muted-foreground">{label}</Label>
      {children}
    </div>
  );
}

function Check({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <Label className="gap-2 text-sm font-normal">
      <Checkbox checked={checked} onCheckedChange={(v) => onChange(v === true)} />
      {label}
    </Label>
  );
}

export function ServeForm({ repo, revision, models, mmprojs, state, onChange, status }: Props) {
  const set = <K extends keyof ServeFormState>(k: K, v: ServeFormState[K]) => onChange({ ...state, [k]: v });
  const [hint, setHint] = useState<{ text: string; title?: string; max?: number; busy?: boolean }>({ text: "" });
  const seq = useRef(0);
  const groups = status?.groups || [];
  const listId = `group-list-${repo.replace(/[^a-z0-9]/gi, "-")}`;

  const { model, mmproj, parallel, cache_type_k, cache_type_v, group, preload } = state;
  useEffect(() => {
    if (!repo || !model) return;
    const mine = ++seq.current;
    const timer = window.setTimeout(async () => {
      setHint((h) => ({ ...h, text: "computing max...", busy: true }));
      const params = new URLSearchParams({
        revision: revision || "main",
        file: model,
        mmproj,
        parallel: parallel || "1",
        cache_type_k,
        cache_type_v,
        group: group.trim(),
        preload: String(preload),
        ctx: String(Number(state.ctx_size) || 0),
      });
      try {
        const r = await api<ContextReport>(`/api/context/${repo}?` + params);
        if (mine !== seq.current) return;
        const m = r.max;
        const exact = r.kv_source === "metadata";
        const text =
          `max ${num(m.ctx)} (${m.fits ? (m.limited_by === "trained" ? "trained limit" : "fits " + r.budget_gb.toFixed(1) + " GB") : "does not fit"}` +
          (r.parallel > 1 ? `, ${num(m.per_slot)} per slot` : "") +
          `), KV ${m.kv_gb.toFixed(1)} GB` +
          (exact ? ` · ${(r.kv_per_token / 1024).toFixed(1)} KB/token` : " · estimated, no GGUF metadata") +
          (r.resident ? " · resident budget" : "");
        const e = r.estimate;
        setHint({
          text,
          max: m.ctx,
          title: `at ${num(e.ctx)}: weights ${e.weights} + mmproj ${e.mmproj} + KV ${e.kv} + overhead ${e.overhead} = ${e.total} GB`,
        });
      } catch (err) {
        if (mine !== seq.current) return;
        setHint({ text: err instanceof ApiError ? "max unavailable: " + err.message : "" });
      }
    }, 300);
    return () => window.clearTimeout(timer);
    // ctx_size only feeds the estimate tooltip; the old UI did not refetch on it either.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [repo, revision, model, mmproj, parallel, cache_type_k, cache_type_v, group, preload]);

  return (
    <div className="grid gap-3">
      <datalist id={listId}>
        {groups.map((g) => (
          <option key={g} value={g} />
        ))}
      </datalist>
      <div className="grid gap-x-4 gap-y-3 sm:grid-cols-2 lg:grid-cols-3">
        <Field label="Model id (llama-swap)">
          <Input value={state.id} required onChange={(e) => set("id", e.target.value)} className="font-mono" />
        </Field>
        <Field label="Display name">
          <Input value={state.name} onChange={(e) => set("name", e.target.value)} />
        </Field>
        <Field label="Model file">
          <Select value={state.model || NONE} onValueChange={(v) => onChange(withModel(state, v === NONE ? "" : v))}>
            <SelectTrigger className="w-full font-mono">
              <SelectValue placeholder="no model file" />
            </SelectTrigger>
            <SelectContent>
              {models.length === 0 ? <SelectItem value={NONE}>no model file</SelectItem> : null}
              {models.map((g) => (
                <SelectItem key={g.first} value={g.first} className="font-mono">
                  {g.key}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Field>
        <Field label="Vision projector (mmproj)">
          <Select value={state.mmproj || NONE} onValueChange={(v) => set("mmproj", v === NONE ? "" : v)}>
            <SelectTrigger className="w-full font-mono">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={NONE}>none</SelectItem>
              {mmprojs.map((g) => (
                <SelectItem key={g.first} value={g.first} className="font-mono">
                  {g.key}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Field>
        <Field label="Context size (--ctx-size, 0 = model default)">
          <Input type="number" min={0} step={1024} value={state.ctx_size} onChange={(e) => set("ctx_size", e.target.value)} className="font-mono" />
        </Field>
        <Field label="Parallel slots (--parallel)">
          <Input type="number" min={1} max={64} value={state.parallel} onChange={(e) => set("parallel", e.target.value)} className="font-mono" />
        </Field>
        <Field label="KV cache type K (--cache-type-k)">
          <Select value={state.cache_type_k} onValueChange={(v) => set("cache_type_k", v)}>
            <SelectTrigger className="w-full font-mono">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {CACHE_TYPES.map((t) => (
                <SelectItem key={t} value={t} className="font-mono">
                  {t}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Field>
        <Field label="KV cache type V (--cache-type-v)">
          <Select value={state.cache_type_v} onValueChange={(v) => set("cache_type_v", v)}>
            <SelectTrigger className="w-full font-mono">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {CACHE_TYPES.map((t) => (
                <SelectItem key={t} value={t} className="font-mono">
                  {t}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Field>
        <Field label="TTL seconds (0 = never unload, -1 = global)">
          <Input type="number" min={-1} value={state.ttl} onChange={(e) => set("ttl", e.target.value)} className="font-mono" />
        </Field>
        <Field label="Group (from llama-swap.base.yaml)">
          <Input value={state.group} list={listId} onChange={(e) => set("group", e.target.value)} className="font-mono" />
        </Field>
        <Field label="Aliases">
          <Input value={state.aliases} placeholder="comma separated" onChange={(e) => set("aliases", e.target.value)} className="font-mono" />
        </Field>
        <Field label="Extra llama-server args">
          <Input
            value={state.extra_args}
            placeholder="e.g. --temp 0.6 --top-k 20"
            onChange={(e) => set("extra_args", e.target.value)}
            className="font-mono"
          />
        </Field>
        <Field label="MTP draft tokens (--spec-draft-n-max)">
          <Input
            type="number"
            min={1}
            max={32}
            value={state.mtp_draft_n_max}
            disabled={!state.mtp}
            onChange={(e) => set("mtp_draft_n_max", e.target.value)}
            className="font-mono"
          />
        </Field>
      </div>

      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="text-muted-foreground">Context:</span>
        {hint.busy ? <Loader2 className="size-3 animate-spin text-muted-foreground" /> : null}
        {hint.title ? (
          <Tooltip>
            <TooltipTrigger asChild>
              <span className="cursor-help font-mono text-muted-foreground underline decoration-dotted underline-offset-2">{hint.text}</span>
            </TooltipTrigger>
            <TooltipContent className="font-mono">{hint.title}</TooltipContent>
          </Tooltip>
        ) : (
          <span className="font-mono text-muted-foreground">{hint.text}</span>
        )}
        {hint.max != null ? (
          <Button type="button" size="xs" variant="outline" onClick={() => set("ctx_size", String(hint.max))}>
            Use max
          </Button>
        ) : null}
      </div>

      <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
        <Check label="MTP speculative decoding (--spec-type draft-mtp)" checked={state.mtp} onChange={(v) => set("mtp", v)} />
        <Check label="Preload at llama-swap start" checked={state.preload} onChange={(v) => set("preload", v)} />
        <Check label="Enabled" checked={state.enabled} onChange={(v) => set("enabled", v)} />
      </div>
    </div>
  );
}
