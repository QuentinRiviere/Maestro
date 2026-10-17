"""Command line: verified fetch of models.yml, llama-swap.yaml rendering, context sizing.

  python -m model_hub.cli fetch   [--config models.yml] [--models /models] [--dry-run] [--verify]
  python -m model_hub.cli render  [--base llama-swap.base.yaml] [--config models.yml]
                                  [--out llama-swap.yaml] [--models /models] [--check]
  python -m model_hub.cli context <repo> [--file F] [--revision main] [--mmproj M] [--parallel N]
                                  [--cache q8_0 | --cache-k T --cache-v T] [--group G] [--preload]

`fetch` downloads every entry of models.yml (see fetcher.py for the checks).
`render` writes llama-swap.yaml from the base file and the `serve:` sections;
`--check` only validates and prints the diff.
`context` reads a GGUF's metadata (installed file, else a few MB from the
Hub) and prints the exact KV cache per token and the largest --ctx-size that
fits the memory budget (MODEL_HUB_MEMORY_GB, _RESERVED_GB, _SWAP_RESERVE_GB,
minus the resident models of models.yml).
Env: HF_TOKEN (optional, read-only; only sent to huggingface.co).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from . import fetcher, gguf_meta, ggufs, registry, render
from .hf import HubError
from .paths import check_repo_id


def _progress_printer():
    state = {"last": 0.0}

    def progress(p: dict) -> None:
        now = time.monotonic()
        if now - state["last"] < 5:
            return
        state["last"] = now
        if p["phase"] == "download":
            print(f"    {p['done'] / 1e9:6.2f} / {p['size'] / 1e9:.2f} GB  {p['rate'] / 1e6:6.1f} MB/s", flush=True)
        else:
            print("    verifying hash...", flush=True)
    return progress


def cmd_fetch(args) -> int:
    registry.seed_config(os.path.dirname(os.path.abspath(args.config)))
    config = args.config
    example = os.path.join(os.path.dirname(os.path.abspath(config)), "models.example.yml")
    if not os.path.exists(config) and os.path.exists(example):
        print(f"{config} does not exist yet: using {example}")
        config = example
    entries = registry.load(config)
    store = fetcher.Store(args.models)
    failures = 0
    for entry in entries:
        repo = entry["repo"]
        revision = str(entry.get("revision", "main"))
        try:
            snap = fetcher.resolve(repo, revision)
        except (HubError, fetcher.FetchError) as exc:
            print(f"{repo}: ERROR: {exc}", file=sys.stderr)
            failures += 1
            continue
        pinned = registry.is_pinned(entry)
        print(f"{repo} @ {snap.commit}" + ("" if pinned else f"  (floating '{revision}': pin revision: {snap.commit})"))
        files, refused = fetcher.select(snap, entry.get("include"), entry.get("exclude"))
        for name in refused:
            print(f"  refused (unsafe format): {name}")
        if not files:
            print("  ERROR: no file matches include/exclude", file=sys.stderr)
            failures += 1
            continue
        for f in files:
            print(f"  {f.name}  {f.size / 1e9:.2f} GB")
            if args.dry_run:
                continue
            try:
                result = store.fetch_file(repo, snap.commit, f, force_verify=args.verify,
                                          progress=_progress_printer())
            except fetcher.FetchError as exc:
                print(f"    ERROR: {exc}", file=sys.stderr)
                failures += 1
                continue
            print({"skipped": "    already verified, skipped", "in-place": "    in place, hash OK",
                   "downloaded": f"    OK ({f.algo} verified)"}[result])
    print("done" if not failures else f"{failures} error(s)")
    return 1 if failures else 0


def cmd_render(args) -> int:
    registry.seed_config(os.path.dirname(os.path.abspath(args.config)))
    store = fetcher.Store(args.models) if args.models and os.path.isdir(args.models) else None
    available = None
    if store is not None:
        available = lambda repo, name: os.path.exists(store.target(repo, name))  # noqa: E731
    try:
        result = render.render_files(args.base, args.config, args.out, available, dry_run=args.check)
    except (render.RenderError, registry.ConfigError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    for warning in result["warnings"]:
        print(f"warning: {warning}", file=sys.stderr)
    if result["diff"]:
        print(result["diff"])
    print(("would change " if args.check else "wrote ") + args.out if result["changed"] else "unchanged")
    return 0


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        return default


def cmd_context(args) -> int:
    try:
        repo = check_repo_id(args.repo)
        store = fetcher.Store(args.models) if args.models and os.path.isdir(args.models) else None
        installed = {f["name"]: int(f["size"]) for f in (store.installed().get(repo, []) if store else [])
                     if f["present"]}
        if installed:
            sizes, revision = installed, args.revision
        else:
            snap = fetcher.resolve(repo, args.revision)
            sizes, revision = {f.name: f.size for f in snap.files if f.name.lower().endswith(".gguf")}, snap.commit
        groups = [g for g in ggufs.group_files([{"name": n, "size": s} for n, s in sizes.items()]) if g.kind == "model"]
        if args.file:
            entry = next((g for g in groups if args.file in g.files), None)
            if entry is None:
                raise ValueError(f"{args.file} is not a model file of {repo} (have: {', '.join(g.key for g in groups)})")
        elif len(groups) == 1:
            entry = groups[0]
        else:
            raise ValueError("--file is needed, the repo has several models: " + ", ".join(g.key for g in groups))
        if args.mmproj and args.mmproj not in sizes:
            raise ValueError(f"{args.mmproj} is not a file of {repo}")
        cache_k, cache_v = args.cache_k or args.cache or "f16", args.cache_v or args.cache or "f16"
        ggufs.cache_type_bytes(cache_k)
        ggufs.cache_type_bytes(cache_v)
    except (ValueError, HubError, fetcher.FetchError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    meta = gguf_meta.MetaStore(os.path.dirname(os.path.abspath(args.config)))
    local = store.target(repo, entry.first) if installed else None
    info = meta.info(repo, revision, entry.first, local)
    # Budget: like the hub, minus the resident models of models.yml (heuristic KV when not installed).
    budget = ggufs.Budget(_env_float("MODEL_HUB_MEMORY_GB", 128.0), _env_float("MODEL_HUB_RESERVED_GB", 10.0),
                          swap_reserve_gb=_env_float("MODEL_HUB_SWAP_RESERVE_GB", 0.0))
    base_groups = render.base_groups(render.load_base(args.base)) if os.path.exists(args.base) else {}
    manifest = store.load_manifest() if store else {}
    for other in registry.load(args.config) if os.path.exists(args.config) else []:
        serve = other.get("serve")
        if not serve or other["repo"] == repo or not serve["enabled"] or not ggufs.is_resident(
                serve["group"], base_groups, serve["preload"]):
            continue
        size = lambda n: int((manifest.get(f"{other['repo']}/{n}") or {}).get("size", 0))  # noqa: E731
        est = ggufs.estimate_memory(size(serve["model"]), serve["ctx_size"], size(serve.get("mmproj") or ""),
                                    parallel=serve["parallel"])
        budget.resident_gb += est["total"] / ggufs.GB
        budget.resident.append(serve["id"])
    group = args.group or os.environ.get("MODEL_HUB_DEFAULT_GROUP") or "on-demand"
    resident = ggufs.is_resident(group, base_groups, args.preload)
    report = ggufs.context_report(info, entry.size, sizes.get(args.mmproj, 0) if args.mmproj else 0,
                                  budget.available_for(resident) * ggufs.GB, args.ctx, cache_k, cache_v, args.parallel)
    arch, best, est = report["arch"], report["max"], report["estimate"]
    print(f"{repo} @ {revision}  {entry.key}  {entry.size / 1e9:.2f} GB" + (f" + mmproj {args.mmproj}" if args.mmproj else ""))
    if arch and report["kv_source"] == "metadata":
        kv = report["kv"]
        heads = arch["head_count_kv"]
        print(f"  {arch['architecture']}: {arch['block_count']} layers ({kv['attention_layers']} attention"
              + (f", {kv['swa_layers']} sliding window {kv['window']}" if kv["swa_layers"] else "")
              + (f", {kv['linear_layers']} without KV cache" if kv["linear_layers"] else "")
              + f"), KV heads {heads if isinstance(heads, int) else 'per layer'} x {arch['key_length']}/{arch['value_length']}"
              + (f", {arch['expert_count']} experts" if arch["expert_count"] else "")
              + f", trained context {arch['context_length']}")
        print(f"  KV cache: {report['kv_per_token'] / 1024:.1f} KB/token ({cache_k}/{cache_v})")
    else:
        print(f"  no GGUF metadata ({meta.last_error(repo, revision, entry.first) or 'incomplete'}): "
              f"KV cache guessed at {report['kv_per_token'] / 1024:.1f} KB/token from the weights size")
    print(f"  budget: {report['budget_gb']} GB for a{'n always-loaded (resident)' if resident else 'n on-demand'} "
          f"model in group {group}" + (f" (residents: {', '.join(budget.resident)})" if budget.resident else ""))
    print(f"  max --ctx-size {best['ctx']} ({best['per_slot']} per slot x {args.parallel}, "
          f"{'trained limit' if best['limited_by'] == 'trained' else 'budget limit'}"
          + ("" if best["fits"] else ", DOES NOT FIT even at 2048") + f"): KV {best['kv_gb']} GB, total {best['total_gb']} GB")
    print(f"  at --ctx-size {est['ctx']}: weights {est['weights']} + mmproj {est['mmproj']} + KV {est['kv']} "
          f"+ overhead {est['overhead']} = {est['total']} GB")
    return 0 if best["fits"] else 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-"):
        argv.insert(0, "fetch")  # the legacy fetch_models.py interface
    parser = argparse.ArgumentParser(prog="model_hub.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch", help="download and verify the models of models.yml")
    f.add_argument("--config", default=os.environ.get("MODEL_HUB_MODELS_YML", "/config/models.yml"))
    f.add_argument("--models", default=os.environ.get("MODEL_HUB_MODELS_DIR", "/models"))
    f.add_argument("--dry-run", action="store_true")
    f.add_argument("--verify", action="store_true", help="re-hash files already in place")
    f.set_defaults(func=cmd_fetch)
    r = sub.add_parser("render", help="render llama-swap.yaml from the base file and models.yml")
    r.add_argument("--base", default="/config/llama-swap.base.yaml")
    r.add_argument("--config", default=os.environ.get("MODEL_HUB_MODELS_YML", "/config/models.yml"))
    r.add_argument("--out", default="/config/llama-swap.yaml")
    r.add_argument("--models", default=os.environ.get("MODEL_HUB_MODELS_DIR", "/models"),
                   help="skip models whose files are not here (pass '' to map everything)")
    r.add_argument("--check", action="store_true", help="validate and print the diff, write nothing")
    r.set_defaults(func=cmd_render)
    c = sub.add_parser("context", help="exact KV cache per token and the largest --ctx-size that fits")
    c.add_argument("repo", help="owner/name on the Hub")
    c.add_argument("--file", help="model GGUF (first part of a split); optional when the repo has one model")
    c.add_argument("--revision", default="main")
    c.add_argument("--mmproj", help="vision projector loaded next to the model")
    c.add_argument("--parallel", type=int, default=1)
    c.add_argument("--cache", help="--cache-type-k and -v at once: f16, q8_0, q4_0, ...")
    c.add_argument("--cache-k", help="--cache-type-k")
    c.add_argument("--cache-v", help="--cache-type-v")
    c.add_argument("--ctx", type=int, default=0, help="also print the estimate at this --ctx-size (default: the max)")
    c.add_argument("--group", help="llama-swap group (default MODEL_HUB_DEFAULT_GROUP or on-demand)")
    c.add_argument("--preload", action="store_true", help="budget as an always-loaded model")
    c.add_argument("--config", default=os.environ.get("MODEL_HUB_MODELS_YML", "/config/models.yml"))
    c.add_argument("--base", default="/config/llama-swap.base.yaml")
    c.add_argument("--models", default=os.environ.get("MODEL_HUB_MODELS_DIR", "/models"))
    c.set_defaults(func=cmd_context)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
