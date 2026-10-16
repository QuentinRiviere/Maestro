"""Command line: verified fetch of models.yml, and llama-swap.yaml rendering.

  python -m model_hub.cli fetch  [--config models.yml] [--models /models] [--dry-run] [--verify]
  python -m model_hub.cli render [--base llama-swap.base.yaml] [--config models.yml]
                                 [--out llama-swap.yaml] [--models /models] [--check]

`fetch` downloads every entry of models.yml (see fetcher.py for the checks).
`render` writes llama-swap.yaml from the base file and the `serve:` sections;
`--check` only validates and prints the diff.
Env: HF_TOKEN (optional, read-only; only sent to huggingface.co).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from . import fetcher, registry, render
from .hf import HubError


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
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
