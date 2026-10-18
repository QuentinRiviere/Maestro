import { useCallback, useEffect, useState } from "react";
import { Eye, Pencil, Save, ShieldCheck, Trash2, Unlink } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { StateBadge } from "@/components/StateBadge";
import { useToast } from "@/components/toast";
import { api, errorMessage } from "@/lib/api";
import { gb, suggestId } from "@/lib/format";
import type { InstalledRepo, PreviewResult, Status } from "@/lib/types";
import { MemoryCard } from "@/views/MemoryCard";
import { initServeState, ServeForm, serveValue, type FileOption, type ServeFormState } from "@/views/ServeForm";
import { PreviewOutput, previewServe, type Output } from "@/views/PreviewOutput";

interface Props {
  status: Status | null;
  report: (e: unknown) => void;
}

export function Installed({ status, report }: Props) {
  const toast = useToast();
  const [repos, setRepos] = useState<InstalledRepo[] | null>(null);
  const [disk, setDisk] = useState<{ free: number; total: number } | null>(null);
  const [editing, setEditing] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<InstalledRepo | null>(null);

  const load = useCallback(async () => {
    try {
      const data = await api<{ repos: InstalledRepo[]; disk: { free: number; total: number } }>("/api/installed");
      setRepos(data.repos);
      setDisk(data.disk);
    } catch (e) {
      report(e);
    }
  }, [report]);

  useEffect(() => {
    load();
  }, [load]);

  const verify = async (repo: string) => {
    try {
      await api("/api/installed/verify", { method: "POST", body: { repo } });
      toast.toast("Verification started", { description: repo });
    } catch (e) {
      return report(e);
    }
    window.setTimeout(load, 1500);
  };

  return (
    <div className="grid gap-4">
      <MemoryCard />

      <div className="flex items-center gap-3 text-xs text-muted-foreground">
        <span className="font-medium text-foreground">Installed</span>
        {disk ? (
          <span>
            <span className="font-mono tabular-nums">{gb(disk.free)}</span> free of{" "}
            <span className="font-mono tabular-nums">{gb(disk.total)}</span>
          </span>
        ) : null}
      </div>

      {repos === null ? (
        <>
          <Skeleton className="h-36 rounded-xl" />
          <Skeleton className="h-36 rounded-xl" />
        </>
      ) : repos.length === 0 ? (
        <p className="text-sm text-muted-foreground">Nothing installed yet.</p>
      ) : (
        repos.map((r) => (
          <RepoCard
            key={r.repo}
            r={r}
            status={status}
            editing={editing === r.repo}
            onEdit={() => setEditing(editing === r.repo ? null : r.repo)}
            onVerify={() => verify(r.repo)}
            onDelete={() => setDeleting(r)}
            onSaved={() => window.setTimeout(load, 800)}
            onRemoved={load}
            report={report}
          />
        ))
      )}

      <DeleteDialog
        repo={deleting}
        onClose={() => setDeleting(null)}
        onDeleted={() => {
          setDeleting(null);
          load();
        }}
      />
    </div>
  );
}

interface CardProps {
  r: InstalledRepo;
  status: Status | null;
  editing: boolean;
  onEdit: () => void;
  onVerify: () => void;
  onDelete: () => void;
  onSaved: () => void;
  onRemoved: () => void;
  report: (e: unknown) => void;
}

function RepoCard({ r, status, editing, onEdit, onVerify, onDelete, onSaved, onRemoved, report }: CardProps) {
  const v = r.verify;
  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle className="flex flex-wrap items-center justify-between gap-2">
          <code className="text-sm">{r.repo}</code>
          <span className="flex flex-wrap gap-1.5">
            {r.serve ? (
              r.mapped ? (
                <StateBadge tone="ok">mapped as {r.serve.id}</StateBadge>
              ) : (
                <StateBadge tone="warn">not mapped</StateBadge>
              )
            ) : (
              <StateBadge tone="neutral">no serve settings</StateBadge>
            )}
            {v ? (
              <StateBadge tone={v.state === "ok" ? "ok" : v.state === "running" ? "warn" : "bad"} title={v.error}>
                hash {v.state}
              </StateBadge>
            ) : (
              <StateBadge tone="ok">hash verified at download</StateBadge>
            )}
          </span>
        </CardTitle>
        <div className="text-xs text-muted-foreground">
          <span className="font-mono tabular-nums">{gb(r.bytes)}</span> · commit <code>{(r.commit || "").slice(0, 12)}</code>
          {r.in_models_yml ? "" : " · not in models.yml"}
        </div>
      </CardHeader>
      <CardContent className="grid gap-3">
        <ul className="grid gap-0.5 text-xs">
          {r.files.map((f) => (
            <li key={f.name} className="flex flex-wrap items-center gap-2">
              <code>{f.name}</code>
              <span className="font-mono tabular-nums text-muted-foreground">{gb(f.size)}</span>
              {f.present ? null : <StateBadge tone="bad">MISSING</StateBadge>}
              {v && v.files && v.files[f.name] === false ? <StateBadge tone="bad">hash mismatch</StateBadge> : null}
            </li>
          ))}
        </ul>
        <div className="flex flex-wrap gap-2">
          <Button size="sm" variant="outline" onClick={onVerify}>
            <ShieldCheck data-icon="inline-start" />
            Re-verify hashes
          </Button>
          <Button size="sm" variant="outline" onClick={onEdit} aria-expanded={editing}>
            <Pencil data-icon="inline-start" />
            {r.serve ? "Edit mapping" : "Map into llama-swap"}
          </Button>
          <Button size="sm" variant="destructive" disabled={r.downloading} onClick={onDelete}>
            <Trash2 data-icon="inline-start" />
            Delete
          </Button>
        </div>
        {editing ? <EditServe r={r} status={status} onSaved={onSaved} onRemoved={onRemoved} report={report} /> : null}
      </CardContent>
    </Card>
  );
}

function EditServe({
  r,
  status,
  onSaved,
  onRemoved,
  report,
}: {
  r: InstalledRepo;
  status: Status | null;
  onSaved: () => void;
  onRemoved: () => void;
  report: (e: unknown) => void;
}) {
  // Split files: only the first part is a selectable model, as in the old UI.
  const ggufFiles = r.files.map((f) => f.name).filter((n) => /\.gguf$/i.test(n));
  const firstParts = ggufFiles.filter((n) => !/-\d{5}-of-\d{5}\.gguf$/i.test(n) || /-00001-of-\d{5}\.gguf$/i.test(n));
  const toEntry = (n: string): FileOption => ({ key: n, first: n, files: [n] });
  const models = firstParts.filter((n) => !/mmproj/i.test(n)).map(toEntry);
  const mmprojs = firstParts.filter((n) => /mmproj/i.test(n)).map(toEntry);
  const [form, setForm] = useState<ServeFormState>(() =>
    initServeState(r.serve || { id: suggestId(r.repo) }, models, mmprojs, "", status),
  );
  const [out, setOut] = useState<Output | null>(null);
  const [busy, setBusy] = useState(false);

  const save = async () => {
    setBusy(true);
    try {
      const res = await api<PreviewResult>("/api/serve", { method: "POST", body: { repo: r.repo, serve: serveValue(form) } });
      setOut({ warnings: res.warnings || [], note: res.changed ? "Saved; llama-swap.yaml updated." : "Saved; llama-swap.yaml unchanged." });
      onSaved();
    } catch (e) {
      setOut({ error: errorMessage(e) });
    } finally {
      setBusy(false);
    }
  };

  const remove = async () => {
    try {
      await api("/api/serve", { method: "POST", body: { repo: r.repo, serve: null } });
      onRemoved();
    } catch (e) {
      report(e);
    }
  };

  return (
    <div className="grid gap-3 rounded-lg border bg-muted/20 p-3">
      <ServeForm repo={r.repo} revision={r.commit || "main"} models={models} mmprojs={mmprojs} state={form} onChange={setForm} status={status} />
      <div className="flex flex-wrap gap-2">
        <Button size="sm" variant="outline" onClick={async () => setOut(await previewServe(r.repo, serveValue(form)))}>
          <Eye data-icon="inline-start" />
          Preview diff
        </Button>
        <Button size="sm" onClick={save} disabled={busy}>
          <Save data-icon="inline-start" />
          Save mapping
        </Button>
        {r.serve ? (
          <Button size="sm" variant="ghost" onClick={remove}>
            <Unlink data-icon="inline-start" />
            Remove mapping
          </Button>
        ) : null}
      </div>
      <PreviewOutput out={out} />
    </div>
  );
}

function DeleteDialog({ repo, onClose, onDeleted }: { repo: InstalledRepo | null; onClose: () => void; onDeleted: () => void }) {
  const toast = useToast();
  const [typed, setTyped] = useState("");
  const [busy, setBusy] = useState(false);
  const id = repo?.repo || "";
  const matches = typed.trim() === id;

  useEffect(() => {
    setTyped("");
  }, [id]);

  const confirm = async () => {
    if (!repo || !matches) return;
    setBusy(true);
    try {
      const r = await api<{ deleted: string; freed: number }>(`/api/installed/${repo.repo}?` + new URLSearchParams({ confirm: repo.repo }), {
        method: "DELETE",
      });
      toast.success(`Deleted ${r.deleted}`, `freed ${gb(r.freed)}`);
      onDeleted();
    } catch (e) {
      toast.error(errorMessage(e));
      onClose();
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog open={!!repo} onOpenChange={(open) => !open && onClose()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Delete {id}?</DialogTitle>
          <DialogDescription>
            Every file of <code>{id}</code> and its llama-swap mapping will be removed. Type the repo id to confirm.
          </DialogDescription>
        </DialogHeader>
        <Input
          value={typed}
          onChange={(e) => setTyped(e.target.value)}
          placeholder={id}
          className="font-mono"
          autoFocus
          onKeyDown={(e) => e.key === "Enter" && confirm()}
        />
        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button variant="destructive" disabled={!matches || busy} onClick={confirm}>
            <Trash2 data-icon="inline-start" />
            Delete
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
