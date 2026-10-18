import { useState, type FormEvent } from "react";
import { AlertTriangle, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Skeleton } from "@/components/ui/skeleton";
import { StateBadge } from "@/components/StateBadge";
import { api, errorMessage } from "@/lib/api";
import { date, num } from "@/lib/format";
import type { SearchResult, Status } from "@/lib/types";
import type { RepoRequest } from "@/views/RepoDialog";

const SORTS = [
  { value: "downloads", label: "Most downloads" },
  { value: "trendingScore", label: "Trending" },
  { value: "likes", label: "Most likes" },
  { value: "lastModified", label: "Recently updated" },
];

export function Explore({ status, onOpen }: { status: Status | null; onOpen: (r: RepoRequest) => void }) {
  const [q, setQ] = useState("");
  const [sort, setSort] = useState("downloads");
  const [all, setAll] = useState(false);
  const [results, setResults] = useState<SearchResult[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const search = async (ev: FormEvent) => {
    ev.preventDefault();
    setBusy(true);
    setError("");
    const params = new URLSearchParams({ q, sort, limit: "40", all: String(all) });
    try {
      const data = await api<{ results: SearchResult[] }>("/api/search?" + params);
      setResults(data.results);
    } catch (e) {
      setError(errorMessage(e));
      setResults(null);
    } finally {
      setBusy(false);
    }
  };

  const open = (repo: string) => onOpen({ repo, revision: "main" });

  return (
    <div className="grid gap-3">
      <form onSubmit={search} className="flex flex-wrap items-center gap-2">
        <Input
          type="search"
          placeholder="Search GGUF models (e.g. qwen3, gemma, coder)"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          className="min-w-[260px] flex-1"
        />
        <Select value={sort} onValueChange={setSort}>
          <SelectTrigger className="w-44">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {SORTS.map((s) => (
              <SelectItem key={s.value} value={s.value}>
                {s.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Label className="gap-2 text-xs">
          <Switch checked={all} onCheckedChange={setAll} />
          All publishers
        </Label>
        <Button type="submit" disabled={busy}>
          <Search data-icon="inline-start" />
          Search
        </Button>
      </form>

      {all ? (
        <p className="flex items-start gap-2 rounded-lg border border-warn/40 bg-warn/10 px-3 py-2 text-xs text-warn">
          <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
          <span>
            Showing every publisher. Anyone can upload a GGUF: prefer the trusted ones, check the model card, and remember an
            unknown quant can be broken or mislabeled even though the hub verifies its hash.
          </span>
        </p>
      ) : null}
      {status ? <p className="text-xs text-muted-foreground">Trusted publishers: {status.trusted_authors.join(", ")}</p> : null}

      {error ? <p className="text-sm text-bad">{error}</p> : null}
      {busy ? (
        <div className="grid gap-2">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-9" />
          ))}
        </div>
      ) : results ? (
        results.length === 0 ? (
          <p className="text-sm text-muted-foreground">No GGUF repository found.</p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Repository</TableHead>
                <TableHead className="text-right">Downloads</TableHead>
                <TableHead className="text-right">Likes</TableHead>
                <TableHead>Updated</TableHead>
                <TableHead />
              </TableRow>
            </TableHeader>
            <TableBody>
              {results.map((r) => (
                <TableRow key={r.repo}>
                  <TableCell>
                    <div className="flex flex-wrap items-center gap-1.5">
                      <button type="button" className="font-mono text-primary hover:underline" onClick={() => open(r.repo)}>
                        {r.repo}
                      </button>
                      <StateBadge>{r.trusted ? "trusted" : "untrusted"}</StateBadge>
                      {r.gated ? <StateBadge tone="neutral">gated</StateBadge> : null}
                    </div>
                  </TableCell>
                  <TableCell className="text-right font-mono tabular-nums">{num(r.downloads)}</TableCell>
                  <TableCell className="text-right font-mono tabular-nums">{num(r.likes)}</TableCell>
                  <TableCell className="text-muted-foreground">{date(r.last_modified)}</TableCell>
                  <TableCell className="text-right">
                    <Button size="sm" variant="outline" onClick={() => open(r.repo)}>
                      Open
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )
      ) : null}
    </div>
  );
}
