#!/usr/bin/env python3
"""Aggregate rhythm statistics from the DramaBoard storyboard annotations.

DramaBoard (iLearn-Lab/DramaDirector, arXiv 2606.24107) annotates every shot
of ~2.8K vertical micro-drama episodes with shot scale, camera angle, camera
motion, dialogue, speaker, emotion and duration. The code is MIT, but the
annotations describe copyrighted series: this script reads them from a local
copy outside the repository and writes ONLY aggregate numbers. No raw shot,
line of dialogue, character or series name ever leaves the input directory.

Usage:
    python3 scripts/dramaboard_rhythm_stats.py <DramaDirector>/dramaboard/storyboard \
        app/services/director/cinema_grammar/data/dramaboard_rhythm.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from collections import Counter, defaultdict

SCHEMA_VERSION = 1

# Raw labels are Chinese (with a few English strays). Map them to a small
# canonical vocabulary; anything else becomes "other".
SCALE = {
    "大特写": "extreme_close_up", "Extreme Close-up": "extreme_close_up", "局部特写": "extreme_close_up",
    "特写": "close_up", "Close-up": "close_up", "Close Up": "close_up",
    "近景": "medium_close_up", "Medium Close-up": "medium_close_up", "Medium Close Up": "medium_close_up",
    "高角度近景": "medium_close_up", "过肩近景": "medium_close_up",
    "中近景": "medium", "Medium Shot": "medium", "中景": "medium_wide", "过肩中景": "medium_wide",
    "低角度中景": "medium_wide", "高角度中景": "medium_wide", "过肩": "medium_close_up", "过肩镜头": "medium_close_up",
    "全景": "wide", "Full Shot": "wide", "中全景": "wide", "Wide Shot": "wide",
    "远景": "extreme_wide", "Long Shot": "extreme_wide",
}
# In Chinese grammar 近景 is chest-up (≈ medium close-up), 中近景 waist-up
# (≈ medium), 中景 knee-up (≈ medium wide / cowboy), 全景 full body (wide).
SCALE_ORDER = ["extreme_close_up", "close_up", "medium_close_up", "medium", "medium_wide", "wide", "extreme_wide"]

ANGLE = {
    "平拍": "eye_level", "Eye Level": "eye_level",
    "侧拍": "profile", "侧角": "profile",
    "俯拍": "high", "高角度": "high", "High Angle": "high", "高角度俯拍": "high",
    "仰拍": "low", "低角度": "low", "Low Angle": "low",
    "过肩": "over_the_shoulder", "Over the Shoulder": "over_the_shoulder",
    "斜角": "dutch", "Dutch Angle": "dutch",
    "背拍": "from_behind", "后侧拍": "from_behind", "侧后方": "from_behind", "后拍": "from_behind",
    "背面": "from_behind", "侧后拍": "from_behind", "背面拍": "from_behind", "背面拍摄": "from_behind",
}

MOTION = {
    "固定": "static", "Static": "static",
    "跟拍": "follow", "跟随": "follow", "Tracking": "follow", "Follow": "follow", "背面跟拍": "follow",
    "摇镜": "pan", "快速摇镜": "pan", "微摇": "pan",
    "推进": "push_in", "快速推进": "push_in", "Push In": "push_in",
    "拉远": "pull_out",
    "晃动": "handheld", "Handheld": "handheld", "轻微晃动": "handheld", "微晃": "handheld",
    "Shake": "handheld", "剧烈晃动": "handheld", "快速晃动": "handheld",
    "升降": "crane",
    "淡出": "transition", "淡入": "transition", "叠化": "transition", "闪白": "transition", "Fade Out": "transition",
}

# Coarse emotion buckets by keyword (labels are free text, ~5.8K distinct).
# Glossary: raw label term -> English translation, grouped by bucket.
EMOTION_KEYWORDS: list[tuple[str, dict[str, str]]] = [
    ("anger", {
        "愤怒": "anger",
        "怒": "rage",
        "暴躁": "irritable",
        "气愤": "indignant",
        "恼": "annoyed",
        "凶狠": "fierce",
        "不满": "discontent",
        "不耐烦": "impatient",
        "指责": "accusing",
        "疯狂": "frenzied",
        "狠厉": "ruthless",
        "不悦": "displeased",
        "激动": "agitated",
        "崩溃": "breaking down",
    }),
    ("contempt", {
        "轻蔑": "contempt",
        "不屑": "disdain",
        "嘲讽": "mocking",
        "讥讽": "sneering",
        "鄙夷": "scorn",
        "挑衅": "provoking",
        "冷笑": "cold smile",
        "得意": "smug",
        "傲慢": "arrogant",
        "讽刺": "sarcastic",
        "玩味": "amused scrutiny",
        "戏谑": "teasing",
        "轻浮": "flippant",
        "嚣张": "brazen",
        "嫌弃": "repelled",
        "厌恶": "disgust",
        "狡黠": "sly",
        "阴险": "sinister",
        "算计": "scheming",
        "满意": "satisfied",
    }),
    ("fear_tension", {
        "紧张": "nervous",
        "惊恐": "terrified",
        "恐惧": "fear",
        "害怕": "afraid",
        "慌": "flustered",
        "焦急": "anxious",
        "急切": "urgent",
        "担忧": "worried",
        "不安": "uneasy",
        "警惕": "wary",
        "焦虑": "anxiety",
        "犹豫": "hesitant",
        "尴尬": "embarrassed",
        "试探": "probing",
        "恳求": "pleading",
        "讨好": "ingratiating",
        "顺从": "submissive",
        "恭敬": "deferential",
    }),
    ("sadness", {
        "悲伤": "sad",
        "痛苦": "pained",
        "绝望": "despair",
        "委屈": "wronged",
        "难过": "upset",
        "哭": "crying",
        "伤心": "heartbroken",
        "心痛": "aching",
        "失落": "dejected",
        "哀": "sorrow",
        "愧疚": "guilty",
        "虚弱": "weak",
        "沉重": "heavy",
        "失望": "disappointed",
        "落寞": "forlorn",
        "隐忍": "holding back",
        "疲惫": "exhausted",
        "复杂": "conflicted",
    }),
    ("surprise", {
        "惊讶": "surprised",
        "震惊": "shocked",
        "吃惊": "startled",
        "疑惑": "puzzled",
        "困惑": "confused",
        "怀疑": "suspicious",
        "诧异": "astonished",
        "质疑": "questioning",
        "好奇": "curious",
        "询问": "inquiring",
        "反问": "retorting",
    }),
    ("threat_command", {
        "威胁": "threatening",
        "命令": "commanding",
        "警告": "warning",
        "质问": "interrogating",
        "逼问": "pressing",
        "决绝": "resolute",
        "坚定": "firm",
        "严厉": "stern",
        "冷峻": "grim",
        "冷酷": "cold-blooded",
        "威严": "imposing",
        "强硬": "hard-line",
        "强势": "domineering",
        "笃定": "certain",
        "坚决": "determined",
        "果断": "decisive",
        "拒绝": "refusing",
        "抗拒": "resisting",
    }),
    ("tenderness_joy", {
        "温柔": "gentle",
        "开心": "happy",
        "喜悦": "joyful",
        "高兴": "glad",
        "微笑": "smiling",
        "甜": "sweet",
        "期待": "hopeful",
        "感动": "moved",
        "宠溺": "doting",
        "害羞": "shy",
        "关切": "caring",
        "轻松": "relaxed",
        "调侃": "joking",
        "深情": "loving",
        "温和": "mild",
        "感激": "grateful",
        "安抚": "soothing",
        "安慰": "comforting",
        "兴奋": "excited",
        "热情": "warm",
        "愉悦": "pleased",
        "暧昧": "flirtatious",
        "撒娇": "coy",
        "诱惑": "seductive",
        "赞赏": "approving",
        "诚恳": "sincere",
    }),
    ("calm_neutral", {
        "平静": "calm",
        "冷静": "composed",
        "严肃": "serious",
        "认真": "earnest",
        "专注": "focused",
        "冷漠": "indifferent",
        "冷淡": "cool",
        "无奈": "resigned",
        "自信": "confident",
        "淡然": "detached",
        "解释": "explaining",
        "确认": "confirming",
        "凝重": "solemn",
        "思考": "thinking",
        "沉思": "pondering",
        "礼貌": "polite",
        "专业": "professional",
        "肯定": "affirming",
        "强调": "emphatic",
        "神秘": "mysterious",
        "辩解": "justifying",
        "回忆": "remembering",
        "请求": "requesting",
        "承诺": "promising",
        "呼唤": "calling out",
    }),
]


def emotion_bucket(label) -> str:
    text = str(label or "").strip()
    if not text or text in ("null", "无", "None"):  # 无 = none
        return "none"
    for bucket, words in EMOTION_KEYWORDS:
        if any(word in text for word in words):
            return bucket
    return "other"


def summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def pct(p: float) -> float:
        k = (len(ordered) - 1) * p
        lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)

    return {
        "n": len(ordered),
        "mean": round(statistics.fmean(ordered), 2),
        "median": round(pct(0.5), 2),
        "p10": round(pct(0.10), 2), "p25": round(pct(0.25), 2),
        "p75": round(pct(0.75), 2), "p90": round(pct(0.90), 2),
    }


def share(counter: Counter) -> dict:
    total = sum(counter.values()) or 1
    return {key: round(count / total, 4) for key, count in counter.most_common()}


def has_dialogue(shot: dict) -> bool:
    text = str(shot.get("dialogue") or "").strip()
    return bool(text) and text not in ("无", "null", "None")  # 无 = none


def speaker_on_screen(shot: dict) -> bool | None:
    speaker = str(shot.get("speaker") or "").strip()
    # 画外 = off-screen voice, 旁白 = narrator
    if not speaker or "画外" in speaker or "旁白" in speaker:
        return None
    names = {str(s.get("name") or "").strip() for s in shot.get("subjects") or [] if isinstance(s, dict)}
    return speaker in names


def compute(root: str) -> dict:
    durations_all: list[float] = []
    by_scale: dict[str, list[float]] = defaultdict(list)
    by_motion: dict[str, list[float]] = defaultdict(list)
    by_dialogue: dict[str, list[float]] = defaultdict(list)
    by_emotion: dict[str, list[float]] = defaultdict(list)
    scale_counts, angle_counts, motion_counts, emotion_counts = Counter(), Counter(), Counter(), Counter()
    transitions: dict[str, Counter] = defaultdict(Counter)
    episode_lengths: list[float] = []
    shots_per_episode: list[int] = []
    episode_asl: list[float] = []
    # Coverage of spoken lines: whose face is on screen while a line plays.
    line_coverage = Counter()
    line_cut_into_shots: list[int] = []
    after_line = Counter()
    episodes = 0

    for path in sorted(glob.glob(os.path.join(root, "shot", "*", "*.json"))):
        try:
            with open(path, encoding="utf-8") as handle:
                shots = json.load(handle)
        except (OSError, ValueError):
            continue
        shots = [s for s in shots if isinstance(s, dict)]
        valid = []
        for shot in shots:
            try:
                duration = float(shot.get("duration"))
            except (TypeError, ValueError):
                continue
            if not 0.05 <= duration <= 60:
                continue
            valid.append((shot, duration))
        if not valid:
            continue
        episodes += 1
        total = sum(d for _, d in valid)
        episode_lengths.append(total)
        shots_per_episode.append(len(valid))
        episode_asl.append(total / len(valid))

        previous_scale = None
        previous_line = None
        run = 0
        for index, (shot, duration) in enumerate(valid):
            scale = SCALE.get(str(shot.get("shot_scale")), "other")
            angle = ANGLE.get(str(shot.get("camera_angle")), "other")
            motion = MOTION.get(str(shot.get("camera_motion")), "other")
            emotion = emotion_bucket(shot.get("emotion"))
            spoken = has_dialogue(shot)
            durations_all.append(duration)
            by_scale[scale].append(duration)
            by_motion[motion].append(duration)
            by_dialogue["with_dialogue" if spoken else "silent"].append(duration)
            by_emotion[emotion].append(duration)
            scale_counts[scale] += 1
            angle_counts[angle] += 1
            motion_counts[motion] += 1
            emotion_counts[emotion] += 1
            if previous_scale is not None:
                transitions[previous_scale][scale] += 1
            previous_scale = scale

            line = str(shot.get("dialogue") or "").strip() if spoken else None
            if spoken:
                on_screen = speaker_on_screen(shot)
                line_coverage["speaker_on_screen" if on_screen else
                              "listener_or_insert" if on_screen is False else "voice_over"] += 1
            # A line repeated across consecutive shots is one line cut over several shots.
            if line and line == previous_line:
                run += 1
            else:
                if previous_line:
                    line_cut_into_shots.append(run)
                run = 1 if line else 0
                # The shot right after a line ends: silent reaction or the answer.
                if previous_line and index < len(valid):
                    if not spoken:
                        after_line[f"silent_{scale}"] += 1
                    else:
                        after_line["next_line"] += 1
            previous_line = line
        if previous_line:
            line_cut_into_shots.append(run)

    transition_share = {
        scale: {"n": sum(nexts.values()), **{k: v for k, v in share(nexts).items()}}
        for scale, nexts in sorted(transitions.items(), key=lambda item: -sum(item[1].values()))
    }
    after_total = sum(after_line.values()) or 1
    silent_after = sum(v for k, v in after_line.items() if k.startswith("silent_"))
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "dataset": "DramaBoard storyboard annotations (iLearn-Lab/DramaDirector, arXiv 2606.24107)",
            "code_license": "MIT",
            "note": ("Aggregate statistics only. Annotations describe copyrighted vertical micro-dramas "
                     "(mostly Chinese); no raw annotation is redistributed."),
            "episodes": episodes,
            "shots": len(durations_all),
        },
        "scale_vocabulary": SCALE_ORDER,
        "episode": {
            "duration_sec": summary(episode_lengths),
            "shots": summary([float(n) for n in shots_per_episode]),
            "asl_sec": summary(episode_asl),
        },
        "shot_duration_sec": {
            "overall": summary(durations_all),
            "by_scale": {k: summary(v) for k, v in sorted(by_scale.items(), key=lambda i: -len(i[1]))},
            "by_motion": {k: summary(v) for k, v in sorted(by_motion.items(), key=lambda i: -len(i[1]))},
            "by_dialogue": {k: summary(v) for k, v in by_dialogue.items()},
            "by_emotion": {k: summary(v) for k, v in sorted(by_emotion.items(), key=lambda i: -len(i[1]))},
        },
        "distribution": {
            "shot_scale": share(scale_counts),
            "camera_angle": share(angle_counts),
            "camera_motion": share(motion_counts),
            "emotion": share(emotion_counts),
        },
        "transitions_scale_to_next": transition_share,
        "dialogue_coverage": {
            "shots_with_line": share(line_coverage),
            "shots_per_line": summary([float(n) for n in line_cut_into_shots if n]),
            "line_cut_over_2plus_shots_share": round(
                sum(1 for n in line_cut_into_shots if n >= 2) / max(1, len(line_cut_into_shots)), 4),
            "after_line_ends": {
                "silent_shot_share": round(silent_after / after_total, 4),
                "next_line_share": round(after_line["next_line"] / after_total, 4),
                "silent_shot_scale": share(Counter({k[len("silent_"):]: v for k, v in after_line.items()
                                                    if k.startswith("silent_")})),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("storyboard_dir", help="DramaDirector/dramaboard/storyboard (local copy, outside the repo)")
    parser.add_argument("output", help="Where to write the aggregate JSON")
    args = parser.parse_args()
    stats = compute(args.storyboard_dir)
    if not stats["source"]["shots"]:
        raise SystemExit("No shot annotations found under " + args.storyboard_dir)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=1, ensure_ascii=False)
        handle.write("\n")
    print(f"{stats['source']['episodes']} episodes, {stats['source']['shots']} shots -> {args.output}")


if __name__ == "__main__":
    main()
