"""Draft best-of-N storyboard: cheap candidates per shot, the judge picks one, then a full-resolution final.

Off by default (``director_storyboard_drafts`` 0). With N >= 2 each shot's
start image is first rendered N times at a draft resolution (the run's,
scaled so its long side is 768: 432x768 for 720x1280), same prompt, refs and
image model, one seed per candidate. The N candidates of a round are
rendered in one job when the stage has a batch renderer (``repeat_generation``
N: the engine encodes the prompt and references once and denoises N times,
seeds seed, seed + 1, ...); each candidate's seed is read back from the
settings the engine embeds in its output. A batch that returns fewer files
falls back to one job per missing candidate. The storyboard judge ranks the N
candidates in one vision call (``storyboard_judge.rank_candidates``); when its
best is still flagged, N new candidates are rendered with its fix hint
appended to the prompt, up to ``director_draft_rounds`` rounds, and the best
seen overall is kept. Without a fal key candidate 1 is kept.

The chosen draft is then upscaled to the run resolution and re-rendered
image-to-image at that resolution (the texture refine's full-frame masked
edit, ``image_refine.build_refine_job``, at ``director_draft_final_strength``)
so its composition and poses stay; the texture refine then runs on the final
as on any start image.

Draft images are billed under the ``image_draft`` stage, finals under
``image_final``. Each shot's record (candidates, verdicts, chosen draft,
rounds, final) is kept in the pipeline state and the storyboard report.
"""
from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

DRAFT_STAGE = "image_draft"
FINAL_STAGE = "image_final"
DRAFT_LONG_SIDE = 768
DEFAULT_ROUNDS = 2
DEFAULT_FINAL_STRENGTH = 0.55
MAX_DRAFTS = 8
MAX_ROUNDS = 5
MIN_FINAL_STRENGTH, MAX_FINAL_STRENGTH = 0.05, 0.95

FINAL_CLAUSE = (
    "Re-render reference image 1 at full resolution as the finished frame: keep its composition, framing, "
    "camera angle, the people, their number, places and poses, the set and the lighting exactly; add sharp, "
    "clean detail."
)
IDENTITY_CLAUSE = "The other reference images only show who the people are: keep every face identical."


@dataclass(frozen=True)
class DraftSettings:
    count: int = 0
    rounds: int = DEFAULT_ROUNDS
    resolution: str = ""
    final_strength: float = DEFAULT_FINAL_STRENGTH

    @property
    def enabled(self) -> bool:
        return self.count >= 2


def parse_resolution(value: Any) -> Optional[tuple[int, int]]:
    """(width, height) of a "WxH" text, None when it is not one."""
    match = re.fullmatch(r"\s*(\d+)\s*[xX*]\s*(\d+)\s*", str(value or ""))
    if not match:
        return None
    width, height = int(match.group(1)), int(match.group(2))
    return (width, height) if width > 0 and height > 0 else None


def draft_resolution(run_resolution: Any, long_side: int = DRAFT_LONG_SIDE) -> str:
    """The run's resolution scaled so its long side is ``long_side``, both sides multiples of 16.

    A run already that small keeps its resolution: a draft never upscales.
    """
    size = parse_resolution(run_resolution)
    if size is None:
        return str(run_resolution or "")
    width, height = size
    if max(width, height) <= long_side:
        return f"{width}x{height}"
    scale = long_side / max(width, height)

    def snap(value: float) -> int:
        return max(16, int(round(value * scale / 16.0)) * 16)

    return f"{snap(width)}x{snap(height)}"


def _number(value, default, cast, low, high):
    try:
        number = cast(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    return min(high, max(low, number))


def draft_settings(params: Optional[dict], run_resolution: Any) -> DraftSettings:
    """The draft settings of a Director run, with defaults for missing or invalid values."""
    params = params or {}
    count = _number(params.get("director_storyboard_drafts", 0), 0, lambda v: int(float(v)), 0, MAX_DRAFTS)
    resolution = str(params.get("director_draft_resolution") or "").strip()
    if parse_resolution(resolution) is None:
        resolution = draft_resolution(run_resolution)
    return DraftSettings(
        count=count if count >= 2 else 0,
        rounds=_number(params.get("director_draft_rounds", DEFAULT_ROUNDS), DEFAULT_ROUNDS,
                       lambda v: int(float(v)), 1, MAX_ROUNDS),
        resolution=resolution,
        final_strength=_number(params.get("director_draft_final_strength", DEFAULT_FINAL_STRENGTH),
                               DEFAULT_FINAL_STRENGTH, float, MIN_FINAL_STRENGTH, MAX_FINAL_STRENGTH),
    )


def candidate_seeds(base: int, count: int) -> list[int]:
    """``count`` distinct seeds from ``base``, one per candidate (the engine's own repeat seeds)."""
    return [(int(base) + k) % (2 ** 31) for k in range(count)]

def output_seed(path: str) -> Optional[int]:
    """The seed the engine recorded for an output image, None when it cannot be read.

    Read from the ``.json`` settings sidecar when the engine writes one, else
    from the settings embedded in the image (its ``comment``), else from the
    engine's ``_seed<N>_`` file name.
    """
    sources: list[Any] = []
    try:
        with open(os.path.splitext(path)[0] + ".json", "r", encoding="utf-8") as handle:
            sources.append(json.load(handle))
    except (OSError, ValueError):
        pass
    try:
        from PIL import Image

        with Image.open(path) as image:
            comment = (getattr(image, "text", {}) or {}).get("comment") or image.info.get("comment")
        if isinstance(comment, (bytes, bytearray)):
            comment = comment.decode("utf-8", "ignore")
        if comment:
            sources.append(json.loads(comment))
    except Exception:  # noqa: BLE001 - a missing or foreign comment is no seed
        pass
    for settings in sources:
        if not isinstance(settings, dict):
            continue
        try:
            seed = int(settings.get("seed"))
        except (TypeError, ValueError):
            continue
        if seed >= 0:
            return seed
    match = re.search(r"_seed(\d+)_", os.path.basename(path))
    return int(match.group(1)) if match else None


def _score(verdict: str, issues: Sequence[str]) -> tuple[int, int]:
    return (1 if verdict == "pass" else 0, -len(issues or []))


def mark_duplicates(ranking: dict, paths: Sequence[str], others: Sequence[tuple[int, str]]) -> dict:
    """The ranking with candidates that copy another shot of the scene flagged.

    ``others`` are (shot index, image path) of the scene's earlier shots. A
    candidate nearly identical to one of them is flagged "copies shot N"
    (see ``storyboard_judge.duplicate_of``); a flagged best gives way to the
    first passing candidate.
    """
    from . import storyboard_judge

    if not others:
        return ranking
    other_hashes = []
    for shot_index, path in others:
        try:
            other_hashes.append((shot_index, storyboard_judge.image_hashes(path)))
        except (OSError, ValueError):
            continue
    verdicts = ranking["verdicts"]
    for verdict, path in zip(verdicts, paths):
        try:
            own = storyboard_judge.image_hashes(path)
        except (OSError, ValueError):
            continue
        copied = next((index for index, hashes in other_hashes if storyboard_judge.near_identical(own, hashes)),
                      None)
        if copied is not None:
            verdict["verdict"] = "flagged"
            verdict["issues"] = [*verdict.get("issues", []), f"copies shot {copied + 1}"]
            verdict["duplicate_of"] = copied
    best = verdicts[ranking["best"] - 1] if 0 < ranking["best"] <= len(verdicts) else None
    if best is not None and best["verdict"] != "pass":
        passing = next((v["index"] for v in verdicts if v["verdict"] == "pass"), None)
        if passing is not None:
            ranking["best"] = passing
    return ranking


def select_draft(
    *,
    shot_label: str,
    base_prompt: str,
    settings: DraftSettings,
    render: Callable[[str, int], str],
    out_dir: str,
    rank: Optional[Callable[[list[str]], dict]] = None,
    render_batch: Optional[Callable[[str, int, int], Sequence[str]]] = None,
    others: Sequence[tuple[int, str]] = (),
    seed_base: Optional[Callable[[], int]] = None,
    on_progress: Callable[[str], None] = lambda message: None,
    is_cancelled: Callable[[], bool] = lambda: False,
    fatal: tuple = (),
    log: Callable[[str], None] = print,
) -> dict:
    """Render, rank and pick one shot's draft; return its record.

    ``render(prompt, seed)`` renders one candidate at the draft resolution and
    returns its filename in ``out_dir``. ``render_batch(prompt, seed, count)``
    renders ``count`` candidates in one job (seeds ``seed``, ``seed + 1``, ...)
    and returns their filenames in order; a round uses it when given, and
    renders one job each for the candidates it does not return (fewer files,
    or a failure). A candidate's recorded seed is the one its output carries
    (``output_seed``), else the planned one. ``rank(paths)`` is the judge's
    ranking (``storyboard_judge.rank_candidates``), None without a fal key:
    candidate 1 is then kept. A failed candidate is skipped; an exception in
    ``fatal`` (a generation timeout) propagates. The record's ``chosen`` is
    the best candidate seen over all rounds: a passing one first, then the
    fewest issues, the earliest on a tie.
    """
    seed_base = seed_base or (lambda: random.randrange(2 ** 31))
    record: dict[str, Any] = {
        "count": settings.count, "resolution": settings.resolution, "judge": "vision" if rank else "none",
        "candidates": [], "rounds": [], "chosen": None, "verdict": "unranked", "issues": [], "fix_hint": "",
        "final": "", "final_strength": settings.final_strength,
    }
    hint = ""
    best_score: Optional[tuple[int, int]] = None
    for round_number in range(1, settings.rounds + 1):
        if is_cancelled():
            break
        prompt = f"{base_prompt.rstrip()} {hint}".strip() if hint else base_prompt
        round_entry: dict[str, Any] = {"round": round_number, "hint": hint, "best": None, "fix_hint": "",
                                       "cost_usd": 0.0}
        record["rounds"].append(round_entry)
        rendered: list[dict] = []
        seeds = candidate_seeds(seed_base(), settings.count)
        batch_files: list[str] = []
        if render_batch is not None and settings.count >= 2:
            on_progress(f"Storyboard drafts: {shot_label}, round {round_number}, "
                        f"{settings.count} candidates in one job")
            try:
                batch_files = [os.path.basename(name) for name in (render_batch(prompt, seeds[0], settings.count)
                                                                   or []) if name][:settings.count]
            except fatal:
                raise
            except Exception as error:  # noqa: BLE001 - the candidates render one job each instead
                log(f"Storyboard drafts: {shot_label} round {round_number} batch of {settings.count} failed, "
                    f"rendering one job per candidate: {error}")
            if 0 < len(batch_files) < settings.count:
                log(f"Storyboard drafts: {shot_label} round {round_number}: the batch returned "
                    f"{len(batch_files)}/{settings.count} files, rendering the rest one job each")
        for number, seed in enumerate(seeds, start=1):
            if is_cancelled():
                break
            entry: dict[str, Any] = {"round": round_number, "candidate": number, "seed": seed, "file": "",
                                     "verdict": "unranked", "issues": []}
            record["candidates"].append(entry)
            if number <= len(batch_files):
                entry["file"] = batch_files[number - 1]
                entry["batched"] = True
            else:
                on_progress(f"Storyboard drafts: {shot_label}, round {round_number}, "
                            f"candidate {number}/{settings.count}")
                try:
                    entry["file"] = os.path.basename(render(prompt, seed) or "")
                except fatal:
                    raise
                except Exception as error:  # noqa: BLE001 - other candidates may still render
                    entry["error"] = str(error)[:300]
                    log(f"Storyboard drafts: {shot_label} round {round_number} candidate {number} failed: {error}")
                    continue
            if entry["file"]:
                recorded = output_seed(os.path.join(out_dir, entry["file"]))
                if recorded is not None:
                    entry["seed"] = recorded
                rendered.append(entry)
        if not rendered:
            round_entry["error"] = "no candidate rendered"
            break
        paths = [os.path.join(out_dir, entry["file"]) for entry in rendered]
        if rank is None:
            log(f"Storyboard drafts: {shot_label}: no judge (no fal key or judge off), keeping candidate "
                f"{rendered[0]['candidate']}")
            if record["chosen"] is None:
                record["chosen"] = {"round": round_number, "candidate": rendered[0]["candidate"],
                                    "file": rendered[0]["file"]}
            round_entry["best"] = rendered[0]["candidate"]
            break
        on_progress(f"Storyboard drafts: {shot_label}, round {round_number}, ranking {len(rendered)} candidates")
        try:
            ranking = rank(paths)
        except fatal:
            raise
        except Exception as error:  # noqa: BLE001 - the ranking is advisory
            round_entry["error"] = f"Ranking failed: {str(error)[:260]}"
            round_entry["cost_usd"] = float(getattr(error, "cost_usd", 0) or 0)
            log(f"Storyboard drafts: {shot_label} round {round_number} ranking failed: {error}")
            if record["chosen"] is None:
                record["chosen"] = {"round": round_number, "candidate": rendered[0]["candidate"],
                                    "file": rendered[0]["file"]}
            break
        ranking = mark_duplicates(ranking, paths, others)
        round_entry["cost_usd"] = float(ranking.get("cost_usd") or 0)
        for verdict, entry in zip(ranking["verdicts"], rendered):
            entry["verdict"] = verdict["verdict"]
            entry["issues"] = list(verdict.get("issues") or [])
            if verdict.get("duplicate_of") is not None:
                entry["duplicate_of"] = verdict["duplicate_of"]
        best = rendered[ranking["best"] - 1] if 0 < ranking["best"] <= len(rendered) else rendered[0]
        round_entry.update(best=best["candidate"], fix_hint=ranking.get("fix_hint") or "")
        score = _score(best["verdict"], best["issues"])
        if best_score is None or score > best_score:
            best_score = score
            record["chosen"] = {"round": round_number, "candidate": best["candidate"], "file": best["file"]}
            record.update(verdict=best["verdict"], issues=list(best["issues"]),
                          fix_hint=ranking.get("fix_hint") or "")
        if best["verdict"] == "pass":
            break
        hint = ranking.get("fix_hint") or ""
    return record


def chosen_file(record: Optional[dict]) -> str:
    return str(((record or {}).get("chosen") or {}).get("file") or "")


def final_prompt(image_prompt: str, *, with_identity: bool) -> str:
    """The shot's own image prompt followed by the keep-the-composition instruction."""
    parts = [str(image_prompt or "").strip(), FINAL_CLAUSE]
    if with_identity:
        parts.append(IDENTITY_CLAUSE)
    return " ".join(part.rstrip() for part in parts if part)


def upscale(source_path: str, size: tuple[int, int], dest_path: str) -> str:
    """``source_path`` resized to ``size`` (LANCZOS) as ``dest_path``, the final pass's input."""
    from PIL import Image

    with Image.open(source_path) as image:
        image.convert("RGB").resize(tuple(size), Image.Resampling.LANCZOS).save(dest_path)
    return dest_path


def build_final_job(*, source_path: str, out_dir: str, image_prompt: str, identity_refs: Sequence[str],
                    settings: DraftSettings, refine_settings: Any, size: tuple[int, int], pid: str) -> dict:
    """The final pass job of one shot: the texture refine's masked edit at the final strength.

    ``source_path`` is the chosen draft, already upscaled to ``size``; the
    model and steps are the texture refine's (``image_refine.refine_settings``).
    """
    from dataclasses import replace

    from . import image_refine

    refs = [ref for ref in identity_refs if ref and os.path.isfile(ref)]
    job = image_refine.build_refine_job(
        source_path=source_path,
        mask_path=image_refine.full_frame_mask(out_dir, *size, pid=pid),
        prompt=final_prompt(image_prompt, with_identity=bool(refs)),
        identity_refs=refs,
        settings=replace(refine_settings, strength=settings.final_strength),
        width=size[0],
        height=size[1],
        pid=pid,
    )
    job["_director_cost_stage"] = FINAL_STAGE
    return job


def run_final(*, pid: str, index: int, record: dict, out_dir: str, image_prompt: str,
              identity_refs: Sequence[str], settings: DraftSettings, refine_settings: Any,
              size: tuple[int, int], submit: Callable[[dict], list], crop_faces: bool = False) -> str:
    """Upscale shot ``index``'s chosen draft, re-render it at ``size``; return the final's filename.

    Records the final (and the draft it came from) in ``record``. Raises when
    the shot has no chosen draft or the job records no output. With
    ``crop_faces`` (an insert) a draft that still shows faces is cut back to
    the band below them first (``insert_crop``), and the record says so.
    """
    from . import image_refine, insert_crop

    draft = chosen_file(record)
    draft_path = os.path.join(out_dir, draft) if draft else ""
    if not draft_path or not os.path.isfile(draft_path):
        raise RuntimeError(f"shot {index + 1} has no chosen draft image")
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", os.path.splitext(draft)[0])
    source_path = draft_path
    if crop_faces:
        try:
            flagged = insert_crop.judge_saw_faces(record.get("issues") or [])
            cropped = insert_crop.crop_below_faces(
                draft_path, os.path.join(out_dir, f"{stem}_below_faces.png"),
                assume_faces_above=insert_crop.ASSUMED_FACES_ABOVE if flagged else None)
        except Exception as error:  # noqa: BLE001 - the crop is a repair, never the reason a final fails
            cropped = None
            record["crop_error"] = str(error)[:200]
        if cropped:
            source_path = cropped
            record["cropped_below_faces"] = os.path.basename(cropped)
    upscaled = upscale(source_path, size, os.path.join(out_dir, f"{stem}_upscaled_{size[0]}x{size[1]}.png"))
    job = build_final_job(source_path=upscaled, out_dir=out_dir, image_prompt=image_prompt,
                          identity_refs=identity_refs, settings=settings, refine_settings=refine_settings,
                          size=size, pid=pid)
    outputs = submit(job)
    final = outputs[0] if outputs else ""
    final_path = os.path.join(out_dir, final) if final else ""
    if not final or not os.path.isfile(final_path):
        raise RuntimeError("the final pass recorded no output image")
    image_refine._match_size(final_path, size)
    record.update(final=final, final_from=draft, final_strength=settings.final_strength,
                  upscaled=os.path.basename(upscaled))
    return final


def report_entry(index: int, record: Optional[dict]) -> dict:
    """A shot of the storyboard report from its draft record (the judge report's shot shape)."""
    if not record:
        return {"index": index, "verdict": "skipped", "attempts": 0, "issues": [], "fix_hint": "",
                "duplicate_of": None, "image": "", "cost_usd": 0.0}
    verdict = record.get("verdict") or "unranked"
    return {
        "index": index,
        "verdict": {"pass": "pass", "flagged": "flagged"}.get(verdict, "unranked"),
        "attempts": max(0, len(record.get("rounds") or []) - 1),
        "issues": list(record.get("issues") or []),
        "fix_hint": record.get("fix_hint") or "",
        "duplicate_of": None,
        "image": record.get("final") or chosen_file(record),
        "cost_usd": round(sum(float(r.get("cost_usd") or 0) for r in record.get("rounds") or []), 5),
        "drafts": record,
    }


def build_report(records: Sequence[Optional[dict]], *, model: Optional[str], status: str = "running",
                 started_at: Optional[float] = None) -> dict:
    """The storyboard report of a draft run, readable where the judge's report is."""
    import time

    shots = [report_entry(index, record) for index, record in enumerate(records)]
    vision = any((record or {}).get("judge") == "vision" for record in records)
    return {
        "status": status, "mode": "drafts", "model": model if vision else None,
        "started_at": started_at or time.time(),
        "cost_usd": round(sum(shot["cost_usd"] for shot in shots), 5),
        "shots": shots,
        "flagged": [shot["index"] for shot in shots if shot["verdict"] == "flagged"],
        "regenerated": [],
    }
