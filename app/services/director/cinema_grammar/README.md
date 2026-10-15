# Cinema grammar

Film-grammar knowledge the Director storyboard planner reads for K-drama short
episodes (45-90 s), written for a planner whose shots are rendered one by one by
image-to-video models.

- `data/grammar.json`: shot sizes, angles, camera moves, camera by emotion,
  lenses, light vocabularies and setups, the action-reaction unit and its
  variants, cut logic, dialogue coverage, editing rules, K-drama motifs, acting
  rules and bad-acting fixes, director style notes, duration references, the
  storyboard columns, and `planner_rules` (the lines the planner block renders).
- `data/dramaboard_rhythm.json`: aggregate rhythm statistics of vertical
  micro-drama episodes, computed by `scripts/dramaboard_rhythm_stats.py`.
- `render.py`: `planner_block()` renders the CINEMA GRAMMAR block (about 2,500
  characters, hard cap 2,800), with rhythm targets clamped to the model's shot
  length range.
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
