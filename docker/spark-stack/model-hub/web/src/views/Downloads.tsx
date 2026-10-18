import { useCallback, useEffect, useState } from "react";
import { Play, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { StateBadge } from "@/components/StateBadge";
import { api } from "@/lib/api";
import { fmtDuration, gb } from "@/lib/format";
import type { Job } from "@/lib/types";

const POLL_MS = 1500;

export function Downloads({ report }: { report: (e: unknown) => void }) {
  const [jobs, setJobs] = useState<Job[] | null>(null);

  const load = useCallback(async () => {
    try {
      const data = await api<{ jobs: Job[] }>("/api/downloads");
      setJobs(data.jobs);
      return true;
    } catch (e) {
      report(e);
      return false;
    }
  }, [report]);

  // Polls while the tab is open; stops on the first failed call like the old UI.
  useEffect(() => {
    let timer: number | undefined;
    let alive = true;
    const tick = async () => {
      const ok = await load();
      if (alive && ok) timer = window.setTimeout(tick, POLL_MS);
    };
    tick();
    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
  }, [load]);

  const action = async (id: string, what: "cancel" | "resume") => {
    try {
      await api(`/api/downloads/${id}/${what}`, { method: "POST" });
      await load();
    } catch (e) {
      report(e);
    }
  };

  if (jobs === null)
    return (
      <div className="grid gap-3">
        <Skeleton className="h-28 rounded-xl" />
        <Skeleton className="h-28 rounded-xl" />
      </div>
    );
  if (!jobs.length)
    return <p className="text-sm text-muted-foreground">No download yet. Open a model from the Catalog or Explore tab.</p>;

  return (
    <div className="grid gap-3">
      {jobs.map((j) => {
        const pct = j.total_bytes ? (100 * j.done_bytes) / j.total_bytes : 0;
        const active = j.state === "queued" || j.state === "running";
        return (
          <Card key={j.id} size="sm">
            <CardHeader>
              <CardTitle className="flex items-center justify-between gap-2">
                <code className="text-sm">{j.repo}</code>
                <StateBadge>{j.state}</StateBadge>
              </CardTitle>
            </CardHeader>
            <CardContent className="grid gap-2">
              <div className="flex flex-wrap items-center gap-3 text-xs">
                <Progress value={pct} className="h-2 w-56" />
                <span className="font-mono tabular-nums">
                  {gb(j.done_bytes)} / {gb(j.total_bytes)}
                </span>
                {j.rate ? <span className="font-mono tabular-nums text-muted-foreground">{(j.rate / 1e6).toFixed(1)} MB/s</span> : null}
                {j.eta_seconds != null ? <span className="text-muted-foreground">ETA {fmtDuration(j.eta_seconds)}</span> : null}
                {j.phase === "verify" ? <span className="text-warn">verifying hash</span> : null}
              </div>
              <div className="truncate text-xs text-muted-foreground" title={j.current ? j.current : j.files.join(", ")}>
                {j.current ? (
                  <>
                    current: <code>{j.current}</code>
                  </>
                ) : (
                  j.files.join(", ")
                )}
              </div>
              {j.message ? <div className="text-xs">{j.message}</div> : null}
              {j.error ? <div className="text-xs text-bad">{j.error}</div> : null}
              <div className="flex gap-2">
                {active ? (
                  <Button size="sm" variant="destructive" onClick={() => action(j.id, "cancel")}>
                    <X data-icon="inline-start" />
                    Cancel
                  </Button>
                ) : null}
                {j.state === "cancelled" || j.state === "failed" ? (
                  <Button size="sm" variant="outline" onClick={() => action(j.id, "resume")}>
                    <Play data-icon="inline-start" />
                    Resume
                  </Button>
                ) : null}
              </div>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}
