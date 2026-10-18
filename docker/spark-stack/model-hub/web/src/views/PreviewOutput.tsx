import { DiffView } from "@/components/DiffView";
import { api, errorMessage } from "@/lib/api";
import type { PreviewResult, Serve } from "@/lib/types";

/** What a Preview / Save shows under the serve form. */
export interface Output {
  error?: string;
  warnings?: string[];
  diff?: string | null;
  note?: string;
}

/** POST /api/serve/preview and turn the answer into an Output. */
export async function previewServe(repo: string, serve: Serve): Promise<Output> {
  try {
    const r = await api<PreviewResult>("/api/serve/preview", { method: "POST", body: { repo, serve }, allowError: true });
    if (r.error) return { error: r.error };
    return { warnings: r.warnings || [], diff: r.diff || null, note: r.diff ? undefined : "No change to llama-swap.yaml." };
  } catch (e) {
    return { error: errorMessage(e) };
  }
}

export function PreviewOutput({ out }: { out: Output | null }) {
  if (!out) return null;
  return (
    <div className="grid gap-2">
      {out.error ? <p className="text-xs text-bad">{out.error}</p> : null}
      {(out.warnings || []).map((w, i) => (
        <p key={i} className="rounded-md border border-warn/40 bg-warn/10 px-2.5 py-1.5 text-xs text-warn">
          {w}
        </p>
      ))}
      {out.note ? <p className="text-xs text-muted-foreground">{out.note}</p> : null}
      {out.diff ? <DiffView text={out.diff} /> : null}
    </div>
  );
}
