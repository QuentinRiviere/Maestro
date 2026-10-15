# Cinema grammar

Film-grammar knowledge the Director storyboard planner reads, written for a
planner whose shots are rendered one by one by image-to-video models. The
CINEMA GRAMMAR block (about 2,500 characters, hard cap 2,800) stacks three
layers and lets the run's style references calibrate the numbers.

## Layers

| Layer | File | Chosen by | Carries |
|---|---|---|---|
| Universal | `data/universal.json` | always | action -> reaction unit, one camera move per shot, continuity (180-degree rule, eyeline and action matches), scale steps, AI-video robustness, acting basics, angles. No genre numbers. |
| Format | `data/formats.json` | `director_format_profile` (`vertical_short`, `horizontal`), else `director_aspect_ratio` (portrait or square -> `vertical_short`, landscape -> `horizontal`), else `horizontal` | framing habits and the base shot-size mix (DramaBoard mix for 9:16, a guideline for 16:9) |
| Genre | `data/genres/<id>.json` | `director_genre_profile` (an id), else `auto`: the custom skill id (`skill_pattern`, e.g. `custom:k_drama_mini` -> `kdrama`), else `neutral` | rhythm targets, shot-mix bias, camera habits, look, acting style, motifs, do/don't, whether shot/reverse coverage applies |

Genres:

| Genre | Median shot | Average shot band | Silent holds | Under a line | Static | Notes |
|---|---|---|---|---|---|---|
| `kdrama` | ~4 s | 3.5-5.5 s | 3-6 s | 3-5 s | 80% | held looks and silences, slow push-ins on realisations, restrained acting, frame-within-frame, dividing lines, rain and reflections, hand inserts |
| `micro_drama` | 1.7 s | 1.8-2.9 s | 1.3-2.0 s | 1.9-3.0 s | 94% | DramaBoard aggregates; hooks in the first 3 s, cliffhanger endings |
| `anime_mini_series` | ~3 s | 2.5-5 s | 2.5-5 s | 2.5-4.5 s | 85% | limited animation, held poses, stronger angles, stylised looks; no smear or impact frames (AI video morphs them) |
| `vlog` | ~2.5 s | 2-4.5 s | 1.5-3 s | 2.5-5 s | 30% | handheld direct address, jump cuts, POV inserts, natural light; no shot/reverse coverage |
| `neutral` | ~3 s | 2.5-5 s | 2-4 s | 2.5-5 s | 70% | default when no genre applies |

Only `micro_drama` is measured on a corpus; the others are Maestro guidelines
from the conventions of each genre. Adding a genre is adding a JSON file with
the same keys (`id`, `label`, `tag`, `summary`, `skill_pattern`, `source`,
`rhythm` or `rhythm_source`, `shot_mix_bias`, `coverage`, `camera`, `look`,
`acting`, `motifs`, `do`, `dont`).

## Calibration by style references

When the run uses analysed style references of its bible
(`director_style_reference`, `app/services/director/style_reference.py`),
their measured numbers replace the genre and format defaults:
average and median shot length, cuts per minute, shot-size mix, static share
of camera movement and the shot length under a line vs silent. Several
references are averaged, weighted by their shot count; a number only averages
the references that measured it. Without a measured silent/line split, the
genre's holds are scaled by the references' pace. Every length is clamped into
the video model's shot range, and the block names the source of each number
("from your references", "K-drama default", "K-drama default scaled to your
references", "DramaBoard"). The REFERENCE STAGING GRAMMAR block then leaves its
rhythm and shot-mix numbers out and keeps the references' rules and patterns.

## Files

- `profiles.py`: loads the layers, selects the format and genre, the biased
  shot mix, and `calibrate()` (weighted average of reference measures).
- `render.py`: `rhythm_targets()` and `planner_block()`; lines are kept by
  priority and the hard cap drops the lowest-priority lines whole.
- `data/grammar.json`: the distilled knowledge base the layers summarise:
  shot sizes, angles, camera moves, camera by emotion, lenses, light
  vocabularies and setups, the action-reaction unit and its variants, cut
  logic, dialogue coverage, editing rules, K-drama motifs, acting rules and
  bad-acting fixes, director style notes, duration references, the storyboard
  columns.
- `data/dramaboard_rhythm.json`: aggregate rhythm statistics of vertical
  micro-drama episodes, computed by `scripts/dramaboard_rhythm_stats.py`. It
  feeds the `micro_drama` rhythm, the `vertical_short` shot mix and the
  dialogue-coverage numbers (listener share, silent reaction after a line).
- `vocabulary.py`: the closed camera-move vocabulary used by the H3 prompt
  compiler, and the switch for MiniMax bracket commands (off).

## Rhythm data

DramaBoard (iLearn-Lab/DramaDirector) ships shot-level annotations of 2,825
vertical micro-drama episodes (101,759 shots, mostly Chinese productions).
Those annotations describe copyrighted series, so the repository keeps only
aggregate numbers: duration percentiles by shot scale, camera motion, dialogue
and emotion bucket, the scale, angle and motion distributions, scale-to-scale
transitions and dialogue coverage. To regenerate, download the DramaDirector
repository outside this one and run:

```
python3 scripts/dramaboard_rhythm_stats.py <DramaDirector>/dramaboard/storyboard \
    app/services/director/cinema_grammar/data/dramaboard_rhythm.json
```

The script maps the raw Chinese labels to English (its mapping tables are the
only place original label names appear).

Key figures: median episode 77 s with 34 shots; median shot 1.72 s
(p25 1.10, p75 2.72); per-episode ASL median 2.24 s; shots under a line 1.87 s
median against 1.32 s for silent shots; 94% static camera, 82% eye level;
scale mix medium 26%, close-up 24%, medium close-up 23%, medium wide 19%,
wide 7%; 26% of shots under a line show the listener instead of the speaker;
16% of line endings cut to a silent shot, most often a close-up.

## Sources and licenses

All text is a summary in our own words; nothing is copied verbatim beyond
short technical terms.

| Source | License | What was distilled |
|---|---|---|
| [Shanyin-ai/shanyin-director-master](https://github.com/Shanyin-ai/shanyin-director-master) (Chinese, translated) | MIT | Action-reaction unit and its variants with the 70/30 routine/memorable split, cut logic by shot scale, dialogue staging repertoire and choice logic, storyboard columns, duration references, Lee Chang-dong, Bong Joon-ho and Park Chan-wook style notes |
| [OSideMedia/higgsfield-ai-prompt-skill](https://github.com/OSideMedia/higgsfield-ai-prompt-skill) (`higgsfield-camera`, `higgsfield-acting`, `higgsfield-shotlist-director`) | MIT | One move per shot, camera matched to the focal emotion, lens and aperture by purpose, durations by shot type, move must reach its end framing; eye life, listening and reaction, body and distance, bad-acting atlas; monotony audit |
| [fal-ai-community/skills](https://github.com/fal-ai-community/skills) (`cinematography`, `storytelling`) | MIT | Shot size, angle, movement, light, lens and composition mapped to intent; continuity anchors; pacing bands |
| [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) `wan/utils/system_prompt.py` | Apache-2.0 | Closed vocabularies for light source, light quality, light direction, time, color tone, shot size and angle |
| [inference-sh/skills](https://github.com/inference-sh/skills) (`storyboard-creation`, `video-prompting-guide`) | MIT | 180-degree rule, match on action, eyeline match, screen direction |
| [iLearn-Lab/DramaDirector](https://github.com/iLearn-Lab/DramaDirector), arXiv 2606.24107 (DramaBoard) | Code MIT; annotations from copyrighted series | Aggregate statistics only (see above) |
| MyDramaList, 3GGG, "Popular Visual Cues in K-Dramas" parts 1-4 (2019) | Article, cited | Dividing lines, grouping and boxing (frame-within-frame), Dutch angle and its combination with boxing |

Mirrors and reflections, rain, feet and shoes, the push-in on a tightening hand,
and medium/bust framing as the default with the full-face close-up as a
deliberate emphasis are common K-drama conventions written for Maestro (the
MyDramaList motif series on them was announced but not published).
