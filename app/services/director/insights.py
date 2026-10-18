"""Insights registry: every judge finding of every Director run in one SQLite file.

The storyboard judge, the draft ranking and the video judge each write their
report beside the run (``_director_storyboard_<pid>.json``,
``_director_review_<pid>.json``); the pipeline state
(``_director_pipeline_<pid>.json``) names the models, the skill and each
shot's scene type.  ``ingest_run`` folds the three into
``<out_dir>/_director_insights.sqlite`` so recurring defects can be read per
scene type and per model, and ``summary`` is the read API behind
``GET /api/v1/director/insights``.

``out_dir`` is the outputs root: a run's files are looked up there or in one
of its workspace subdirectories (as ``director_pipeline`` does), and the
database always lives at the root.  Ingestion is idempotent: a run's rows are
deleted and reinserted inside one transaction.

Findings are stored by ``source``:

- ``storyboard``: the storyboard judge's issues on a shot image (the report's
  ``shots[].issues``).  In ``drafts`` mode those issues are the chosen
  candidate's, so they are recorded under ``draft`` instead;
- ``draft``: every ranked candidate's issues (``shots[].drafts.candidates``);
- ``clip``: the per-shot video review issues, with their category, severity
  and ``fixable_by``;
- ``episode``: the episode review defects, with their criterion, severity
  and ``root_cause``.

CLI: ``python -m services.director.insights backfill <out_dir>`` and
``summary <out_dir>`` (JSON on stdout).
"""
from __future__ import annotations

import glob
import json
import os
import re
import sqlite3
import time
from typing import Any, Iterator, Mapping, Optional

DB_FILE = "_director_insights.sqlite"
PIPELINE_FILE_PREFIX = "_director_pipeline_"
STORYBOARD_FILE_PREFIX = "_director_storyboard_"
REVIEW_FILE_PREFIX = "_director_review_"

SOURCES = ("storyboard", "draft", "clip", "episode")
TOP_DESCRIPTIONS = 20

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    pid TEXT PRIMARY KEY,
    created_at REAL,
    completed_at REAL,
    skill TEXT,
    skill_version INTEGER,
    bible TEXT,
    image_model TEXT,
    video_model TEXT,
    drafts_count INTEGER,
    storyboard_judge_model TEXT,
    storyboard_judge_provider TEXT,
    review_backend TEXT,
    review_model TEXT,
    episode_score REAL,
    episode_verdict TEXT,
    scores_json TEXT,
    ingested_at REAL
);
CREATE TABLE IF NOT EXISTS shots (
    pid TEXT NOT NULL,
    "index" INTEGER NOT NULL,
    scene_type TEXT,
    framing TEXT,
    movement TEXT,
    duration_sec REAL,
    keep_json TEXT,
    has_dialogue INTEGER,
    storyboard_verdict TEXT,
    draft_candidates INTEGER,
    draft_rounds INTEGER,
    draft_flagged INTEGER,
    clip_score REAL,
    PRIMARY KEY (pid, "index")
);
CREATE TABLE IF NOT EXISTS findings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pid TEXT NOT NULL,
    shot_index INTEGER,
    source TEXT NOT NULL,
    stage TEXT NOT NULL,
    category TEXT,
    severity TEXT,
    fixable_by TEXT,
    root_cause TEXT,
    description TEXT,
    scene_type TEXT,
    framing TEXT,
    image_model TEXT,
    video_model TEXT,
    judge_model TEXT,
    created_at REAL
);
CREATE INDEX IF NOT EXISTS findings_category_scene ON findings (category, scene_type);
CREATE INDEX IF NOT EXISTS findings_pid ON findings (pid);
"""


# ── Files ─────────────────────────────────────────────────────────────


def db_path(out_dir: str) -> str:
    return os.path.join(out_dir, DB_FILE)


def connect(out_dir: str) -> sqlite3.Connection:
    """Open (and create on demand) the registry of ``out_dir`` in WAL mode."""
    os.makedirs(out_dir, exist_ok=True)
    connection = sqlite3.connect(db_path(out_dir), timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript(_SCHEMA)
    # Columns added after the first registries were written.
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(runs)")}
    if "skill_version" not in columns:
        connection.execute("ALTER TABLE runs ADD COLUMN skill_version INTEGER")
    return connection


def run_dir(out_dir: str, pid: str) -> Optional[str]:
    """The directory holding the run's files: ``out_dir`` or one workspace below it."""
    name = f"{PIPELINE_FILE_PREFIX}{pid}.json"
    if os.path.isfile(os.path.join(out_dir, name)):
        return out_dir
    try:
        for entry in sorted(os.listdir(out_dir)):
            sub = os.path.join(out_dir, entry)
            if os.path.isdir(sub) and os.path.isfile(os.path.join(sub, name)):
                return sub
    except OSError:
        pass
    return None


def iter_pipeline_ids(out_dir: str) -> Iterator[str]:
    """Every saved run id under ``out_dir`` and its workspace subdirectories."""
    pattern = f"{PIPELINE_FILE_PREFIX}*.json"
    paths = glob.glob(os.path.join(out_dir, pattern)) + glob.glob(os.path.join(out_dir, "*", pattern))
    seen: set[str] = set()
    for path in sorted(paths):
        pid = os.path.basename(path)[len(PIPELINE_FILE_PREFIX):-len(".json")]
        if pid and pid not in seen:
            seen.add(pid)
            yield pid


def _load_json(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


# ── Extraction ────────────────────────────────────────────────────────


def _text(value: Any, limit: int = 400) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text[:limit] if text else None


def _number(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> Optional[int]:
    number = _number(value)
    return int(number) if number is not None else None


def _bible_name(bible: Any) -> Optional[str]:
    if isinstance(bible, Mapping):
        return _text(bible.get("id") or bible.get("name") or bible.get("title"), 200)
    return _text(bible, 200)


def normalise_description(text: str) -> str:
    """The frequency key of a finding: lower-case, single spaces, no trailing punctuation."""
    return re.sub(r"\s+", " ", str(text or "")).strip().lower().rstrip(".;:, ")


def _run_row(pid: str, state: Mapping, report: Optional[Mapping], review: Optional[Mapping]) -> dict:
    params = state.get("_params_snapshot") or {}
    episode = (review or {}).get("episode") or {}
    return {
        "pid": pid,
        "created_at": _number(state.get("created_at")),
        "completed_at": _number(state.get("completed_at")),
        "skill": _text(params.get("director_skill"), 200),
        "skill_version": _int(state.get("skill_version")),
        "bible": _bible_name(params.get("bible") if params.get("bible") is not None else state.get("bible")),
        "image_model": _text(state.get("image_model") or params.get("image_model"), 200),
        "video_model": _text(state.get("video_model") or params.get("video_model"), 200),
        "drafts_count": _int(params.get("director_storyboard_drafts")) or 0,
        "storyboard_judge_model": _text((report or {}).get("model"), 200),
        "storyboard_judge_provider": _text((report or {}).get("provider") or (report or {}).get("backend"), 100),
        "review_backend": _text((review or {}).get("backend"), 100),
        "review_model": _text((review or {}).get("model"), 200),
        "episode_score": _number(episode.get("score")),
        "episode_verdict": _text(episode.get("verdict"), 100),
        "scores_json": json.dumps(episode.get("scores")) if isinstance(episode.get("scores"), Mapping) else None,
        "ingested_at": time.time(),
    }


def _by_index(entries: Any) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for entry in entries or []:
        if isinstance(entry, Mapping) and _int(entry.get("index")) is not None:
            out[int(entry["index"])] = dict(entry)
    return out


def _shot_rows(pid: str, state: Mapping, report: Optional[Mapping], review: Optional[Mapping]) -> list[dict]:
    report_shots = _by_index((report or {}).get("shots"))
    review_shots = _by_index((review or {}).get("shots"))
    rows = []
    for position, clip in enumerate(state.get("clips") or []):
        if not isinstance(clip, Mapping):
            continue
        index = _int(clip.get("index"))
        index = position if index is None else index
        judged = report_shots.get(index) or {}
        drafts = judged.get("drafts") if isinstance(judged.get("drafts"), Mapping) else clip.get("storyboard_drafts")
        drafts = drafts if isinstance(drafts, Mapping) else {}
        candidates = [c for c in drafts.get("candidates") or [] if isinstance(c, Mapping)]
        rows.append({
            "pid": pid,
            "index": index,
            "scene_type": _text(clip.get("_director_scene_type"), 100),
            "framing": _text(clip.get("_director_camera_framing"), 100),
            "movement": _text(clip.get("_director_camera_movement"), 100),
            "duration_sec": _number(clip.get("_director_duration_sec")),
            "keep_json": json.dumps(clip.get("_director_keep_sec")) if clip.get("_director_keep_sec") is not None else None,
            "has_dialogue": 1 if clip.get("_director_dialogue_beats") else 0,
            "storyboard_verdict": _text(judged.get("verdict"), 50),
            "draft_candidates": len(candidates),
            "draft_rounds": len(drafts.get("rounds") or []),
            "draft_flagged": sum(1 for c in candidates if c.get("verdict") == "flagged"),
            "clip_score": _number((review_shots.get(index) or {}).get("score")),
        })
    return rows


def _finding_rows(run: Mapping, shots: list[dict], report: Optional[Mapping], review: Optional[Mapping]) -> list[dict]:
    shot_by_index = {shot["index"]: shot for shot in shots}

    def finding(source: str, stage: str, shot_index: Optional[int], description: Any, *, category: Optional[str],
                severity: Any = None, fixable_by: Any = None, root_cause: Any = None,
                judge_model: Any = None) -> Optional[dict]:
        text = _text(description, 400)
        if not text:
            return None
        shot = shot_by_index.get(shot_index) if shot_index is not None else None
        return {
            "pid": run["pid"], "shot_index": shot_index, "source": source, "stage": stage,
            "category": category, "severity": _text(severity, 50), "fixable_by": _text(fixable_by, 100),
            "root_cause": _text(root_cause, 100), "description": text,
            "scene_type": (shot or {}).get("scene_type"), "framing": (shot or {}).get("framing"),
            "image_model": run["image_model"], "video_model": run["video_model"],
            "judge_model": _text(judge_model, 200), "created_at": run["created_at"],
        }

    rows: list[dict] = []
    if report:
        model = report.get("model")
        drafts_mode = report.get("mode") == "drafts"
        for entry in report.get("shots") or []:
            if not isinstance(entry, Mapping):
                continue
            index = _int(entry.get("index"))
            if not drafts_mode:
                for issue in entry.get("issues") or []:
                    rows.append(finding("storyboard", "image", index, issue, category="storyboard",
                                        judge_model=model))
            drafts = entry.get("drafts") if isinstance(entry.get("drafts"), Mapping) else {}
            for candidate in drafts.get("candidates") or []:
                if not isinstance(candidate, Mapping):
                    continue
                for issue in candidate.get("issues") or []:
                    rows.append(finding("draft", "image", index, issue, category="storyboard", judge_model=model))
    if review:
        model = review.get("model")
        for entry in review.get("shots") or []:
            if not isinstance(entry, Mapping):
                continue
            index = _int(entry.get("index"))
            for issue in entry.get("issues") or []:
                if not isinstance(issue, Mapping):
                    continue
                rows.append(finding("clip", "video", index, issue.get("description"),
                                    category=_text(issue.get("category"), 100) or "plan",
                                    severity=issue.get("severity"), fixable_by=issue.get("fixable_by"),
                                    judge_model=model))
        episode = review.get("episode") if isinstance(review.get("episode"), Mapping) else {}
        for defect in episode.get("defects") or []:
            if not isinstance(defect, Mapping):
                continue
            shot = _int(defect.get("shot"))
            index = shot - 1 if shot is not None and shot >= 1 else None
            rows.append(finding("episode", "video", index, defect.get("description"),
                                category=_text(defect.get("criterion"), 100) or "authenticity",
                                severity=defect.get("severity"), root_cause=defect.get("root_cause"),
                                judge_model=episode.get("model") or model))
    return [row for row in rows if row]


# ── Ingestion ─────────────────────────────────────────────────────────


def _insert(connection: sqlite3.Connection, table: str, rows: list[dict]) -> None:
    if not rows:
        return
    columns = list(rows[0].keys())
    quoted = ", ".join(f'"{c}"' for c in columns)
    marks = ", ".join("?" for _ in columns)
    connection.executemany(f"INSERT INTO {table} ({quoted}) VALUES ({marks})",
                           [tuple(row[c] for c in columns) for row in rows])


def ingest_run(out_dir: str, pid: str) -> dict:
    """Replace the registry rows of one run from its saved files; returns the row counts.

    Raises ``FileNotFoundError`` when the run has no saved pipeline state.  A
    missing storyboard or review report is fine (the run may not be judged
    yet): the run and its shots are still recorded.
    """
    directory = run_dir(out_dir, pid)
    if not directory:
        raise FileNotFoundError(f"No saved Director project {pid} under {out_dir}")
    state = _load_json(os.path.join(directory, f"{PIPELINE_FILE_PREFIX}{pid}.json"))
    if state is None:
        raise FileNotFoundError(f"Unreadable Director project {pid} under {directory}")
    report = _load_json(os.path.join(directory, f"{STORYBOARD_FILE_PREFIX}{pid}.json"))
    review = _load_json(os.path.join(directory, f"{REVIEW_FILE_PREFIX}{pid}.json"))

    run = _run_row(pid, state, report, review)
    shots = _shot_rows(pid, state, report, review)
    findings = _finding_rows(run, shots, report, review)

    connection = connect(out_dir)
    try:
        with connection:
            connection.execute("DELETE FROM findings WHERE pid = ?", (pid,))
            connection.execute("DELETE FROM shots WHERE pid = ?", (pid,))
            connection.execute("DELETE FROM runs WHERE pid = ?", (pid,))
            _insert(connection, "runs", [run])
            _insert(connection, "shots", shots)
            _insert(connection, "findings", findings)
    finally:
        connection.close()
    by_source = {source: sum(1 for f in findings if f["source"] == source) for source in SOURCES}
    return {"pid": pid, "runs": 1, "shots": len(shots), "findings": len(findings), "by_source": by_source,
            "has_storyboard": report is not None, "has_review": review is not None}


def backfill(out_dir: str) -> dict:
    """Ingest every saved run under ``out_dir``; a run that fails is listed, not fatal."""
    counts = {"runs": 0, "shots": 0, "findings": 0, "failed": []}
    for pid in iter_pipeline_ids(out_dir):
        try:
            result = ingest_run(out_dir, pid)
        except Exception as error:  # noqa: BLE001 - one bad file must not stop the backfill
            counts["failed"].append({"pid": pid, "error": str(error)[:300]})
            continue
        counts["runs"] += result["runs"]
        counts["shots"] += result["shots"]
        counts["findings"] += result["findings"]
    return counts


# ── Read API ──────────────────────────────────────────────────────────


def _run_filter(days: Optional[float], image_model: Optional[str], video_model: Optional[str],
                skill: Optional[str]) -> tuple[str, list]:
    clauses, values = [], []
    if days:
        clauses.append("runs.created_at >= ?")
        values.append(time.time() - float(days) * 86400)
    for column, value in (("image_model", image_model), ("video_model", video_model), ("skill", skill)):
        if value:
            clauses.append(f"runs.{column} = ?")
            values.append(str(value))
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", values


def summary(out_dir: str, *, days: Optional[float] = None, image_model: Optional[str] = None,
            video_model: Optional[str] = None, skill: Optional[str] = None) -> dict:
    """What recurs across the runs of ``out_dir`` (optionally only recent ones or one model/skill)."""
    if not os.path.isfile(db_path(out_dir)):
        return {"runs": 0, "average_episode_score": None, "trend": [], "findings_by_category": [],
                "findings_by_scene_type": [], "top_descriptions": [], "draft_stats": [], "models": [],
                "filters": {"days": days, "image_model": image_model, "video_model": video_model, "skill": skill}}
    where, values = _run_filter(days, image_model, video_model, skill)
    join = f"JOIN runs ON runs.pid = {{table}}.pid{where}"
    connection = connect(out_dir)
    try:
        runs = connection.execute(
            f"SELECT pid, created_at, episode_score, episode_verdict, image_model, video_model, skill "
            f"FROM runs{where} ORDER BY created_at, pid", values).fetchall()
        scores = [r["episode_score"] for r in runs if r["episode_score"] is not None]
        trend = [{"pid": r["pid"], "created_at": r["created_at"], "episode_score": r["episode_score"],
                  "verdict": r["episode_verdict"]} for r in runs]

        by_category = connection.execute(
            f"SELECT findings.source AS source, findings.category AS category, COUNT(*) AS count "
            f"FROM findings {join.format(table='findings')} GROUP BY findings.source, findings.category "
            f"ORDER BY count DESC, source, category", values).fetchall()
        by_scene = connection.execute(
            f"SELECT findings.scene_type AS scene_type, findings.source AS source, COUNT(*) AS count "
            f"FROM findings {join.format(table='findings')} GROUP BY findings.scene_type, findings.source "
            f"ORDER BY count DESC, scene_type, source", values).fetchall()

        descriptions: dict[str, dict] = {}
        for row in connection.execute(
                f"SELECT findings.description AS description, findings.category AS category, findings.pid AS pid "
                f"FROM findings {join.format(table='findings')}", values):
            key = normalise_description(row["description"])
            if not key:
                continue
            entry = descriptions.setdefault(key, {"description": key, "count": 0, "categories": {}, "pids": []})
            entry["count"] += 1
            entry["categories"][row["category"] or ""] = entry["categories"].get(row["category"] or "", 0) + 1
            if row["pid"] not in entry["pids"]:
                entry["pids"].append(row["pid"])
        top = sorted(descriptions.values(), key=lambda e: (-e["count"], e["description"]))[:TOP_DESCRIPTIONS]
        for entry in top:
            entry["categories"] = sorted(entry["categories"], key=lambda c: (-entry["categories"][c], c))

        draft_stats = connection.execute(
            f"SELECT shots.scene_type AS scene_type, COUNT(*) AS shots, "
            f"SUM(shots.draft_candidates) AS candidates, SUM(shots.draft_flagged) AS flagged, "
            f"AVG(shots.draft_rounds) AS rounds "
            f"FROM shots {join.format(table='shots')}{' AND' if where else ' WHERE'} shots.draft_candidates > 0 "
            f"GROUP BY shots.scene_type ORDER BY candidates DESC, scene_type", values).fetchall()
    finally:
        connection.close()

    models = sorted({m for r in runs for m in (r["image_model"], r["video_model"]) if m})
    return {
        "runs": len(runs),
        "average_episode_score": round(sum(scores) / len(scores), 2) if scores else None,
        "trend": trend,
        "findings_by_category": [dict(r) for r in by_category],
        "findings_by_scene_type": [dict(r) for r in by_scene],
        "top_descriptions": top,
        "draft_stats": [{"scene_type": r["scene_type"], "shots": r["shots"], "candidates": r["candidates"] or 0,
                         "flagged": r["flagged"] or 0,
                         "flagged_ratio": round((r["flagged"] or 0) / r["candidates"], 3) if r["candidates"] else 0.0,
                         "rounds": round(r["rounds"] or 0, 2)} for r in draft_stats],
        "models": models,
        "filters": {"days": days, "image_model": image_model, "video_model": video_model, "skill": skill},
    }


# ── CLI ───────────────────────────────────────────────────────────────


def skill_versions(out_dir: str, skill: str, *, days: Optional[float] = None) -> list[dict]:
    """Per version of ``skill``: runs, reviewed runs, mean episode score, mean clip score (oldest first)."""
    if not os.path.isfile(db_path(out_dir)):
        return []
    where, values = _run_filter(days, None, None, skill)
    connection = connect(out_dir)
    try:
        rows = connection.execute(
            f"SELECT runs.skill_version AS version, COUNT(DISTINCT runs.pid) AS runs, "
            f"COUNT(DISTINCT CASE WHEN runs.episode_score IS NOT NULL THEN runs.pid END) AS reviewed, "
            f"AVG(runs.episode_score) AS episode_mean, "
            f"(SELECT AVG(shots.clip_score) FROM shots JOIN runs AS r2 ON r2.pid = shots.pid "
            f" WHERE r2.skill = runs.skill AND r2.skill_version IS runs.skill_version) AS clip_mean "
            f"FROM runs{where}{' AND' if where else ' WHERE'} runs.skill_version IS NOT NULL "
            f"GROUP BY runs.skill_version "
            f"ORDER BY runs.skill_version", values).fetchall()
    finally:
        connection.close()
    return [{"version": r["version"], "runs": r["runs"], "reviewed": r["reviewed"],
             "episode_mean": round(r["episode_mean"], 2) if r["episode_mean"] is not None else None,
             "clip_mean": round(r["clip_mean"], 2) if r["clip_mean"] is not None else None} for r in rows]


def rollback_decision(out_dir: str, skill: str, *, current: Optional[int] = None, min_runs: int = 3,
                      margin: float = 0.5) -> dict:
    """Whether the skill's current version should give way to the best earlier one.

    The current version is judged once it has ``min_runs`` reviewed runs:
    when an earlier version with as many reviewed runs scored better by
    more than ``margin`` (the run-to-run noise of the local judge), the
    decision names it (``revert_to``). Clip means break a tie.
    """
    versions = [v for v in skill_versions(out_dir, skill) if v["reviewed"]]
    if current is None and versions:
        current = versions[-1]["version"]
    now = next((v for v in versions if v["version"] == current), None)
    decision = {"skill": skill, "current": current, "versions": versions, "revert_to": None, "reason": ""}
    if now is None or now["reviewed"] < min_runs:
        decision["reason"] = (f"version {current} has {now['reviewed'] if now else 0} reviewed run(s), "
                              f"{min_runs} needed")
        return decision
    earlier = [v for v in versions if v["version"] < current and v["reviewed"] >= min_runs
               and v["episode_mean"] is not None]
    if not earlier:
        decision["reason"] = "no earlier version with enough reviewed runs"
        return decision
    best = max(earlier, key=lambda v: (v["episode_mean"], v["clip_mean"] or 0))
    gap = best["episode_mean"] - (now["episode_mean"] or 0)
    if gap > margin:
        decision["revert_to"] = best["version"]
        decision["reason"] = (f"version {best['version']} scored {best['episode_mean']} over {best['reviewed']} "
                              f"run(s) against {now['episode_mean']} for version {current}")
    else:
        decision["reason"] = (f"version {current} holds: {now['episode_mean']} against {best['episode_mean']} "
                              f"for version {best['version']} (margin {margin})")
    return decision


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m services.director.insights",
                                     description="Director insights registry")
    commands = parser.add_subparsers(dest="command", required=True)
    fill = commands.add_parser("backfill", help="ingest every saved run under the outputs dir")
    fill.add_argument("out_dir")
    show = commands.add_parser("summary", help="print what recurs across the runs")
    show.add_argument("out_dir")
    show.add_argument("--days", type=float, default=None)
    show.add_argument("--image-model", default=None)
    show.add_argument("--video-model", default=None)
    show.add_argument("--skill", default=None)
    versions = commands.add_parser("versions", help="compare a skill's versions and say whether to roll back")
    versions.add_argument("out_dir")
    versions.add_argument("skill")
    versions.add_argument("--min-runs", type=int, default=3)
    versions.add_argument("--margin", type=float, default=0.5)
    args = parser.parse_args(argv)
    if args.command == "backfill":
        result = backfill(args.out_dir)
    elif args.command == "versions":
        result = rollback_decision(args.out_dir, args.skill, min_runs=args.min_runs, margin=args.margin)
    else:
        result = summary(args.out_dir, days=args.days, image_model=args.image_model,
                         video_model=args.video_model, skill=args.skill)
    print(json.dumps(result, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
