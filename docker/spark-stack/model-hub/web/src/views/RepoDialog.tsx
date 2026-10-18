// Repo view: commit pin, quants grouped (split files merged), mmproj and
// draft files, per-quant memory estimate for a chosen context, and the
// serve settings used by the download.
import { useCallback, useEffect, useState } from "react";
import { Download, ExternalLink, Eye, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { StateBadge } from "@/components/StateBadge";
import { api, errorMessage } from "@/lib/api";
import { date, gb, num, suggestId } from "@/lib/format";
import type { ArchInfo, CatalogItem, GgufGroup, RepoDetail, Status } from "@/lib/types";
import { initServeState, ServeForm, serveValue, withModel, type ServeFormState } from "@/views/ServeForm";
import { PreviewOutput, previewServe, type Output } from "@/views/PreviewOutput";

export interface RepoRequest {
  repo: string;
  revision: string;
  catalogItem?: CatalogItem;
}

// One line about the architecture read from the GGUF header (null when unreadable).
function archSummary(a: ArchInfo | null, source: string) {
  if (!a || source !== "metadata") return "GGUF metadata unavailable: KV cache guessed from the weights size (~6 KB per token per GB).";
  const heads = Array.isArray(a.head_count_kv) ? "per-layer" : a.head_count_kv;
  const parts = [`${a.architecture} · ${a.block_count} layers`, `KV heads ${heads} × ${a.key_length}/${a.value_length}`];
  if (a.expert_count) parts.push(`${a.expert_count} experts`);
  if (a.sliding_window) parts.push(`sliding window ${a.sliding_window}`);
  if (a.full_attention_interval) parts.push(`full attention every ${a.full_attention_interval} layers`);
  if (a.context_length) parts.push(`trained context ${num(a.context_length)}`);
  return parts.join(" · ");
}

interface Props {
  request: RepoRequest | null;
  status: Status | null;
  onClose: () => void;
  onQueued: () => void;
}

export function RepoDialog({ request, status, onClose, onQueued }: Props) {
  return (
    <Dialog open={!!request} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-h-[calc(100vh-2rem)] overflow-y-auto sm:max-w-[1050px]">
        {request ? <RepoBody key={request.repo + "@" + request.revision} request={request} status={status} onQueued={onQueued} /> : null}
      </DialogContent>
    </Dialog>
  );
}

function RepoBody({ request, status, onQueued }: { request: RepoRequest; status: Status | null; onQueued: () => void }) {
  const { repo, revision, catalogItem } = request;
  const preset = catalogItem?.serve;
  const [ctx, setCtx] = useState<number>(preset?.ctx_size || 32768);
  const [ctxInput, setCtxInput] = useState(String(preset?.ctx_size || 32768));
  const [d, setD] = useState<RepoDetail | null>(null);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState<GgufGroup | null>(null);
  const [form, setForm] = useState<ServeFormState | null>(null);
  const [mapAfter, setMapAfter] = useState(true);
  const [out, setOut] = useState<Output | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let alive = true;
    setError("");
    api<RepoDetail>(`/api/repo/${repo}?` + new URLSearchParams({ revision, ctx: String(ctx) }))
      .then((detail) => {
        if (!alive) return;
        setD(detail);
        setCtxInput(String(detail.ctx));
        const models = detail.ggufs.filter((g) => g.kind === "model");
        // Only the first load picks a file and builds the form; a recompute keeps the user's edits.
        setForm((prev) => {
          if (prev) return prev;
          const wanted = preset ? preset.model : null;
          const pick = models.find((g) => g.files.includes(wanted || "")) || models.find((g) => g.fit === "fits") || models[0] || null;
          setSelected(pick);
          const serve = preset || {
            id: suggestId(detail.repo),
            ctx_size: detail.ctx,
            parallel: 1,
            ttl: 600,
            mtp: detail.mtp.mtp,
            mtp_draft_n_max: detail.mtp.draft_n_max || 6,
          };
          const mmprojs = detail.ggufs.filter((g) => g.kind === "mmproj");
          const init = initServeState(serve, models, mmprojs, preset ? preset.mmproj || "" : detail.default_mmproj || "", status);
          return pick ? withModel(init, pick.first, true) : init;
        });
      })
      .catch((e) => alive && setError(errorMessage(e)));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [repo, revision, ctx]);

  const pickModel = useCallback((g: GgufGroup) => {
    setSelected(g);
    setForm((f) => (f ? withModel(f, g.first) : f));
  }, []);

  const download = async () => {
    if (!d || !selected || !form) return;
    const files = [...selected.files];
    const serve = mapAfter ? serveValue(form) : null;
    if (serve && serve.mmproj) files.push(serve.mmproj);
    setBusy(true);
    setOut(null);
    try {
      await api("/api/downloads", { method: "POST", body: { repo: d.repo, revision: d.commit, files, serve } });
      onQueued();
    } catch (e) {
      setOut({ error: errorMessage(e) });
    } finally {
      setBusy(false);
    }
  };

  const preview = async () => {
    if (!d || !form) return;
    setOut(await previewServe(d.repo, serveValue(form)));
  };

  const models = d ? d.ggufs.filter((g) => g.kind === "model") : [];
  const mmprojs = d ? d.ggufs.filter((g) => g.kind === "mmproj") : [];
  const others = d ? d.ggufs.filter((g) => g.kind !== "model" && g.kind !== "mmproj") : [];
  const exact = d?.kv_source === "metadata";

  return (
    <>
      <DialogHeader>
        <DialogTitle className="font-mono text-base">{repo}</DialogTitle>
        <DialogDescription asChild>
          <div className="grid gap-1 text-xs">
            {error ? (
              <span className="text-bad">{error}</span>
            ) : !d ? (
              <span>Loading...</span>
            ) : (
              <>
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="text-muted-foreground">commit</span>
                  <code>{d.commit}</code>
                  {d.pinned ? <StateBadge>pinned</StateBadge> : <StateBadge tone="warn">resolved from {d.revision}</StateBadge>}
                  <StateBadge>{d.trusted ? "trusted" : "untrusted"}</StateBadge>
                  {d.gated ? <StateBadge tone="warn">gated</StateBadge> : null}
                  {d.mtp.mtp ? <StateBadge tone="info">MTP{d.mtp.draft_n_max ? ` n=${d.mtp.draft_n_max}` : ""}</StateBadge> : null}
                  {d.license ? <StateBadge tone="neutral">{d.license}</StateBadge> : null}
                </div>
                <div className="text-muted-foreground">
                  updated {date(d.last_modified)} · {num(d.downloads)} downloads · {num(d.likes)} likes ·{" "}
                  <a
                    href={`https://huggingface.co/${d.repo}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 text-primary hover:underline"
                  >
                    model card <ExternalLink className="size-3" />
                  </a>
                </div>
                {d.refused.length ? (
                  <div className="text-warn">Refused (unsafe formats, never downloaded): {d.refused.join(", ")}</div>
                ) : null}
              </>
            )}
          </div>
        </DialogDescription>
      </DialogHeader>

      {!d && !error ? (
        <div className="grid gap-3">
          <Skeleton className="h-8" />
          <Skeleton className="h-40" />
          <Skeleton className="h-64" />
        </div>
      ) : null}

      {d ? (
        <div className="grid gap-4">
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="text-muted-foreground">Memory estimate for context</span>
            <Input
              type="number"
              min={0}
              step={1024}
              value={ctxInput}
              onChange={(e) => setCtxInput(e.target.value)}
              className="h-7 w-28 font-mono"
            />
            <span className="text-muted-foreground">tokens</span>
            <Button size="xs" variant="outline" onClick={() => setCtx(Number(ctxInput) || 0)}>
              <RefreshCw data-icon="inline-start" />
              Recompute
            </Button>
            <span className="text-muted-foreground">
              · {d.budget.available_gb.toFixed(1)} GB available (resident: {d.budget.resident.join(", ") || "none"})
            </span>
          </div>
          <p className="text-xs text-muted-foreground">{archSummary(d.arch, d.kv_source)}</p>

          {models.length ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-8" />
                  <TableHead>File</TableHead>
                  <TableHead>Quant</TableHead>
                  <TableHead className="text-right">Size</TableHead>
                  <TableHead className="text-right">Memory</TableHead>
                  <TableHead />
                </TableRow>
              </TableHeader>
              <TableBody>
                {models.map((g) => (
                  <TableRow key={g.key} className="cursor-pointer" data-state={g === selected ? "selected" : undefined} onClick={() => pickModel(g)}>
                    <TableCell>
                      <input
                        type="radio"
                        name="gguf"
                        checked={g === selected}
                        onChange={() => pickModel(g)}
                        className="accent-primary"
                        aria-label={g.key}
                      />
                    </TableCell>
                    <TableCell>
                      <code>{g.key}</code>
                      {g.parts > 1 ? (
                        <span className="ml-1 text-xs text-muted-foreground">
                          ({g.parts} parts{g.complete ? "" : ", INCOMPLETE"})
                        </span>
                      ) : null}
                    </TableCell>
                    <TableCell className="font-mono">{g.quant}</TableCell>
                    <TableCell className="text-right font-mono tabular-nums">{gb(g.size)}</TableCell>
                    <TableCell
                      className="text-right font-mono tabular-nums"
                      title={g.memory ? `KV ${g.memory.kv.toFixed(1)} GB (${g.memory.kv_source})` : ""}
                    >
                      {g.memory ? `${exact ? "" : "~"}${g.memory.total.toFixed(1)} GB` : ""}
                    </TableCell>
                    <TableCell>
                      <div className="flex gap-1">
                        {g.fit ? <StateBadge>{g.fit}</StateBadge> : null}
                        {g.installed ? <StateBadge>installed</StateBadge> : null}
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <p className="text-sm text-muted-foreground">No loadable GGUF model in this repository.</p>
          )}

          {mmprojs.length + others.length ? (
            <ul className="grid gap-0.5 text-xs">
              {[...mmprojs, ...others].map((g) => (
                <li key={g.key}>
                  <code>{g.key}</code> <span className="text-muted-foreground">{g.kind} · {gb(g.size)}</span>
                  {g.kind === "mtp" || g.kind === "draft" ? (
                    <span className="text-muted-foreground"> (draft model, not wired automatically)</span>
                  ) : null}
                </li>
              ))}
            </ul>
          ) : null}

          <Separator />
          <h3 className="text-sm font-semibold">Serve with llama-swap</h3>
          <Label className="gap-2 text-sm font-normal">
            <Checkbox checked={mapAfter} onCheckedChange={(v) => setMapAfter(v === true)} />
            Map into llama-swap after the download
          </Label>
          {form ? (
            <ServeForm repo={d.repo} revision={d.commit} models={models} mmprojs={mmprojs} state={form} onChange={setForm} status={status} />
          ) : null}
          <div className="flex flex-wrap gap-2">
            <Button onClick={download} disabled={busy || !selected}>
              <Download data-icon="inline-start" />
              Download (verified)
            </Button>
            <Button variant="outline" onClick={preview} disabled={!form}>
              <Eye data-icon="inline-start" />
              Preview llama-swap.yaml diff
            </Button>
          </div>
          <PreviewOutput out={out} />
        </div>
      ) : null}
    </>
  );
}
