// Memory hero: the host's free memory, a stacked bar (accounted models,
// other processes, page cache, free), the host figures, then one row per
// llama-swap model with the Maestro worker's own models under its row.
// Data: GET /api/memory (model_hub/memory.py), refreshed every 5 s while shown.
import { useEffect, useState } from "react";
import { Cpu, MemoryStick } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { StateBadge } from "@/components/StateBadge";
import { api, errorMessage } from "@/lib/api";
import { gbs } from "@/lib/format";
import type { MemoryHost, MemoryModel, MemoryView, WorkerSummary } from "@/lib/types";
import { cn } from "@/lib/utils";

const POLL_MS = 5000;
const LOW_FREE_GB = 15;
const SWAP_WARN_GB = 1;

function MemoryBar({ host, accounted }: { host: MemoryHost; accounted: number }) {
  const total = Number(host.total_gb) || 0;
  if (!total) return null;
  const used = Number(host.used_gb || 0);
  const models = Math.max(0, Math.min(used, Number(accounted || 0)));
  const other = Math.max(0, used - models);
  const cache = Math.max(0, Math.min(total - used, Number(host.cache_gb || 0)));
  const pct = (v: number) => `${Math.max(0, Math.min(100, (100 * v) / total)).toFixed(2)}%`;
  const segments: { label: string; gb: number; className: string }[] = [
    { label: "accounted models", gb: models, className: "bg-primary" },
    { label: "other processes and kernel", gb: other, className: "bg-muted-foreground/60" },
    { label: "page cache (reclaimable)", gb: cache, className: "hatched bg-muted-foreground/15" },
  ];
  return (
    <div className="grid gap-1.5">
      <div
        role="img"
        aria-label={`${gbs(used)} used of ${gbs(total)}`}
        className="flex h-3 w-full overflow-hidden rounded-full bg-muted"
      >
        {segments.map((s) => (
          <Tooltip key={s.label}>
            <TooltipTrigger asChild>
              <span className={cn("h-full shrink-0 transition-all", s.className)} style={{ width: pct(s.gb) }} />
            </TooltipTrigger>
            <TooltipContent>
              {s.label}: {gbs(s.gb)}
            </TooltipContent>
          </Tooltip>
        ))}
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
        {segments.map((s) => (
          <span key={s.label} className="inline-flex items-center gap-1.5">
            <span className={cn("inline-block size-2.5 rounded-sm", s.className)} />
            {s.label} <span className="font-mono tabular-nums">{gbs(s.gb)}</span>
          </span>
        ))}
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block size-2.5 rounded-sm bg-muted ring-1 ring-border" />
          free <span className="font-mono tabular-nums">{gbs(Math.max(0, total - used - cache))}</span>
        </span>
      </div>
    </div>
  );
}

function WorkerRow({ w }: { w: WorkerSummary }) {
  if (!w.reachable)
    return <div className="text-xs text-muted-foreground">worker: {w.error || "unreachable"}</div>;
  return (
    <div className="flex flex-wrap items-center gap-1.5 text-xs">
      <Cpu className="size-3.5 text-muted-foreground" />
      <span className="text-muted-foreground">holds</span>
      {w.loaded_models && w.loaded_models.length ? (
        w.loaded_models.map((m) => (
          <StateBadge key={m} tone="info" className="font-mono">
            {m}
          </StateBadge>
        ))
      ) : (
        <span className="text-muted-foreground">nothing loaded</span>
      )}
      {w.process_rss_gb != null ? (
        <span className="text-muted-foreground">
          · <span className="font-mono tabular-nums text-foreground">{gbs(w.process_rss_gb)}</span> process
        </span>
      ) : null}
      {w.gpu_free_gb != null ? (
        <span className="text-muted-foreground">
          · <span className="font-mono tabular-nums text-foreground">{gbs(w.gpu_free_gb)}</span> free for it
        </span>
      ) : null}
      {w.running_job ? (
        <StateBadge tone="warn">{`job ${w.running_job.id || ""} ${w.running_job.phase || ""}`.trim()}</StateBadge>
      ) : w.paused ? (
        <StateBadge tone="neutral">paused</StateBadge>
      ) : null}
    </div>
  );
}

function ModelRow({ x }: { x: MemoryModel }) {
  return (
    <li className="grid gap-1.5 py-2 first:pt-0 last:pb-0">
      <div className="flex flex-wrap items-center gap-2">
        <code className="text-[13px]">{x.model}</code>
        <StateBadge>{x.state}</StateBadge>
        {x.resident ? <StateBadge>resident</StateBadge> : null}
        {x.ttl ? <span className="text-xs text-muted-foreground">ttl {x.ttl}s</span> : null}
        {x.estimated_gb != null ? (
          <span className="ml-auto font-mono text-xs tabular-nums" title="hub estimate (worker: process size)">
            ~{gbs(x.estimated_gb)}
          </span>
        ) : null}
      </div>
      {x.worker ? (
        <div className="border-l-2 border-primary/40 pl-3">
          <WorkerRow w={x.worker} />
        </div>
      ) : null}
    </li>
  );
}

export function MemoryCard() {
  const [m, setM] = useState<MemoryView | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let alive = true;
    let timer: number | undefined;
    const tick = async () => {
      try {
        const data = await api<MemoryView>("/api/memory");
        if (!alive) return;
        setM(data);
        setError("");
      } catch (e) {
        if (!alive) return;
        setError(errorMessage(e));
      }
      if (alive) timer = window.setTimeout(tick, POLL_MS);
    };
    tick();
    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
  }, []);

  const host = m?.host || {};
  const free = host.available_gb;
  const lowFree = free != null && free < LOW_FREE_GB;
  const swapWarn = (host.swap_used_gb || 0) > SWAP_WARN_GB;
  const models = m?.llama_swap?.models || [];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <MemoryStick className="size-4 text-primary" />
          Memory
        </CardTitle>
      </CardHeader>
      <CardContent className="grid gap-4">
        {!m && !error ? (
          <>
            <Skeleton className="h-10 w-48" />
            <Skeleton className="h-3" />
            <Skeleton className="h-16" />
          </>
        ) : null}
        {error ? <p className="text-xs text-bad">{error}</p> : null}
        {m ? (
          <>
            {host.total_gb ? (
              <div className="grid gap-3">
                <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
                  <span className={cn("font-mono text-3xl font-semibold tabular-nums", lowFree ? "text-warn" : "text-foreground")}>
                    {gbs(free)} free
                  </span>
                  <span className="text-xs text-muted-foreground">for a new model (kernel MemAvailable)</span>
                </div>
                <MemoryBar host={host} accounted={m.accounted_gb} />
                <dl className="grid grid-cols-2 gap-x-6 gap-y-1 text-xs sm:grid-cols-4">
                  <div>
                    <dt className="text-muted-foreground">used / total</dt>
                    <dd className="font-mono tabular-nums">
                      {gbs(host.used_gb)} / {gbs(host.total_gb)}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-muted-foreground">other processes and kernel</dt>
                    <dd className="font-mono tabular-nums">{m.other_gb != null ? gbs(m.other_gb) : "?"}</dd>
                  </div>
                  <div>
                    <dt className="text-muted-foreground">page cache (reclaimable)</dt>
                    <dd className="font-mono tabular-nums">{host.cache_gb != null ? gbs(host.cache_gb) : "?"}</dd>
                  </div>
                  <div>
                    <dt className="text-muted-foreground">swap</dt>
                    <dd className={cn("font-mono tabular-nums", swapWarn && "text-warn")}>
                      {host.swap_used_gb ? gbs(host.swap_used_gb) : "none"}
                      {host.swap_total_gb ? <span className="text-muted-foreground"> of {gbs(host.swap_total_gb)}</span> : null}
                    </dd>
                  </div>
                </dl>
              </div>
            ) : (
              <p className="text-xs text-muted-foreground">Host memory unknown.</p>
            )}

            <div className="rounded-lg border bg-muted/30 px-3 py-2">
              {!m.llama_swap || !m.llama_swap.reachable ? (
                <p className="text-xs text-muted-foreground">llama-swap not reachable: {m.llama_swap?.error || ""}</p>
              ) : !models.length ? (
                <p className="text-xs text-muted-foreground">No model loaded in llama-swap.</p>
              ) : (
                <ul className="divide-y">
                  {models.map((x) => (
                    <ModelRow key={x.model} x={x} />
                  ))}
                </ul>
              )}
            </div>
          </>
        ) : null}
      </CardContent>
    </Card>
  );
}
