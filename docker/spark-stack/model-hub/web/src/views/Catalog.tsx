import { useEffect, useState } from "react";
import { ExternalLink } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { StateBadge } from "@/components/StateBadge";
import { api } from "@/lib/api";
import type { Budget, CatalogItem } from "@/lib/types";
import type { RepoRequest } from "@/views/RepoDialog";

const CATEGORIES = ["general", "abliterated", "vision", "code", "small"];
const ALL = "__all__";

export function Catalog({ onOpen, report }: { onOpen: (r: RepoRequest) => void; report: (e: unknown) => void }) {
  const [items, setItems] = useState<CatalogItem[] | null>(null);
  const [budget, setBudget] = useState<Budget | null>(null);
  const [category, setCategory] = useState(ALL);

  useEffect(() => {
    let alive = true;
    api<{ items: CatalogItem[]; budget: Budget }>("/api/catalog")
      .then((d) => {
        if (!alive) return;
        setItems(d.items);
        setBudget(d.budget);
      })
      .catch(report);
    return () => {
      alive = false;
    };
  }, [report]);

  const shown = (items || []).filter((i) => category === ALL || i.category === category);

  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap items-center gap-3">
        <Select value={category} onValueChange={setCategory}>
          <SelectTrigger className="w-44">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={ALL}>All categories</SelectItem>
            {CATEGORIES.map((c) => (
              <SelectItem key={c} value={c}>
                {c}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {budget ? (
          <span className="text-xs text-muted-foreground">Fit badges against {budget.available_gb.toFixed(1)} GB available</span>
        ) : null}
      </div>

      {items === null ? (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-44 rounded-xl" />
          ))}
        </div>
      ) : shown.length === 0 ? (
        <p className="text-sm text-muted-foreground">No catalog entry in this category.</p>
      ) : (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {shown.map((i) => (
            <Card key={i.repo} size="sm" className="justify-between">
              <CardHeader>
                <CardTitle className="flex items-start justify-between gap-2">
                  <span>{i.name}</span>
                  <StateBadge tone="neutral">{i.category}</StateBadge>
                </CardTitle>
                <code className="truncate text-muted-foreground" title={i.repo}>
                  {i.repo}
                </code>
              </CardHeader>
              <CardContent className="grid gap-2">
                <div className="flex flex-wrap items-center gap-1.5 text-xs">
                  <span className="font-mono tabular-nums">{i.size_gb.toFixed(1)} GB</span>
                  <span className="text-muted-foreground">download ·</span>
                  <span className="font-mono tabular-nums">~{i.memory_gb} GB</span>
                  <span className="text-muted-foreground">in memory</span>
                  <StateBadge>{i.fit}</StateBadge>
                  {i.installed ? <StateBadge>installed</StateBadge> : null}
                  {i.serve?.mtp ? <StateBadge>MTP</StateBadge> : null}
                  {i.serve?.mmproj ? <StateBadge>vision</StateBadge> : null}
                </div>
                <p className="text-xs text-muted-foreground">{i.notes}</p>
                <p className="text-xs text-muted-foreground">
                  pinned <code>{i.revision.slice(0, 12)}</code>
                </p>
              </CardContent>
              <CardFooter>
                <Button size="sm" onClick={() => onOpen({ repo: i.repo, revision: i.revision, catalogItem: i })}>
                  <ExternalLink data-icon="inline-start" />
                  Open / download
                </Button>
              </CardFooter>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
