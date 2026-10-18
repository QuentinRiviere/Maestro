import { useCallback, useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { CodeBlock, DiffView } from "@/components/DiffView";
import { useToast } from "@/components/toast";
import { api, errorMessage } from "@/lib/api";
import type { ConfigResponse } from "@/lib/types";

export function ConfigView({ report }: { report: (e: unknown) => void }) {
  const toast = useToast();
  const [c, setC] = useState<ConfigResponse | null>(null);
  const [msg, setMsg] = useState("");
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    try {
      const data = await api<ConfigResponse>("/api/config");
      setC(data);
      setMsg(data.error ? "Render error: " + data.error : (data.warnings || []).join(" · "));
    } catch (e) {
      report(e);
    }
  }, [report]);

  useEffect(() => {
    load();
  }, [load]);

  const rerender = async () => {
    setBusy(true);
    try {
      const r = await api<{ changed: boolean }>("/api/render", { method: "POST" });
      const text = r.changed ? "llama-swap.yaml written." : "Unchanged.";
      setMsg(text);
      toast.success(text);
      await load();
    } catch (e) {
      setMsg(errorMessage(e));
      toast.error(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="grid gap-4">
      <div className="flex flex-wrap items-center gap-3">
        <Button variant="outline" onClick={rerender} disabled={busy}>
          <RefreshCw data-icon="inline-start" className={busy ? "animate-spin" : ""} />
          Re-render llama-swap.yaml
        </Button>
        <span className={"text-xs " + (c?.error ? "text-bad" : "text-muted-foreground")}>{msg}</span>
      </div>
      {!c ? (
        <>
          <Skeleton className="h-24 rounded-xl" />
          <Skeleton className="h-64 rounded-xl" />
        </>
      ) : (
        <>
          <Card size="sm">
            <CardHeader>
              <CardTitle>Pending changes</CardTitle>
            </CardHeader>
            <CardContent>
              {c.pending_diff ? (
                <DiffView text={c.pending_diff} />
              ) : (
                <p className="text-xs text-muted-foreground">None: llama-swap.yaml matches the base file and models.yml.</p>
              )}
            </CardContent>
          </Card>
          <Card size="sm">
            <CardHeader>
              <CardTitle>Current llama-swap.yaml</CardTitle>
            </CardHeader>
            <CardContent>
              <CodeBlock text={c.rendered || "(not rendered yet)"} className="max-h-[70vh]" />
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
