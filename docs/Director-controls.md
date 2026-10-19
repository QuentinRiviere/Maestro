# Director controls

Director keeps its project setup and planning history in one vertical message
area, with the story composer and Start / Queue actions below it. Character
and speaker fields fit the available panel width. Long descriptions, filenames,
errors and planning logs wrap inside their cards. Progress updates scroll only
the Director message area, leaving the surrounding page in place.

Director already shares Studio's themes, mobile drawer viewport tracking,
keyboard focus handling and gallery scroll lock. H3 references use the same
reference component and saved-character support.

LoRAs and Advanced appear directly below the model selectors, before the music
and image inputs, so generation settings can be chosen before audio analysis.

## Target duration

Target duration uses Studio's compact duration control. Auto and the current
duration stay above the Time / Window tabs. Auto dims the Time slider and
presets, but grabbing the slider, using its arrow keys, selecting a preset or
editing the timecode immediately switches to manual timing. Auto is available
from either tab.

Time follows model-specific duration steps through five minutes, with 10m,
15m, 30m, 60m and Custom choices for longer projects. Window keeps its exact
window-count controls. Director retains its ten-second minimum and chooses
shot capacity from its selected video model, resolution and GPU recommendation.

For H3 First / Last models that support **Seamless**, Director renders the
full timeline through native continuation windows. The saved maximum shot
length limits each window, while the full movie keeps its planned duration,
including the final trimmed window. The total movie length does not have to
match the frame increments of a single H3 shot. Independent shots still need
to fit the model's frame increments and the saved maximum.

## Music-video clip length

**Advanced → GPU clip limit**, beside inference steps, sets the maximum
generated shot length for supported bounded models. Auto uses Maestro's GPU
recommendation. A manual choice is remembered per model across refreshes and
restarts. For example, H3 fused Ref2VA can be set to **14.4s** (345 frames at
24fps) on a setup where the automatic recommendation is 10.1s. Choosing a
longer limit uses more GPU memory; it does not force every shot to that length.

Music-video setup includes **Clip length**, with Auto and a maximum-length slider.
Auto follows the chosen video model, resolution and memory recommendation. Moving
the dimmed slider selects a custom maximum immediately. The displayed time follows
the model's frame increments, and the minimum clip count updates with it.

Cuts follow section changes, lyric phrases, performer changes, and strong beats.
**Cut Speed** controls how tightly these cues divide the song: **0** keeps the
normal musical layout; **+1 / +2** favor shorter clips, including within sections
that already fit the maximum. **−1** favors longer clips. **−2** uses the fewest
clips that fit the selected maximum, balancing their lengths and placing cuts
near musical cues. For a two-minute song with H3's 14.4s limit, −2 produces nine
clips averaging about 13.3s. Section and percussion changes inside a longer clip
still guide its performance and camera plan without requiring another clip.

New music plans keep cut positions on the output frame
grid; Maestro generates a valid model-native duration and trims the extra tail.
That preserves the soundtrack timing through generation, reruns, and rejoining.
Shorter clips can lower peak memory, though
they add preparation and transitions. After planning, choose **Update clip layout**
to apply a changed maximum. The setting is saved with the project. Separate-clip
length is disabled for Seamless, which uses its native continuation windows.

New audio analyses also estimate sustained percussion passages and entrances
using transient energy on the CPU. Director receives the entrance's song time
and time within the shot, favoring the referenced drummer at those moments
instead of inventing a new drum entrance on an arbitrary beat. These are
percussion estimates, not separated drum stems or exact instrument recognition;
other transient sounds can resemble drums. Unknown timing stays unknown.
Re-analyze the soundtrack and create a new plan to add these cues to an older project.

## Shared prompt writing

Director uses the adaptive writing rules from Studio inside its existing planning
stages. Short concepts receive development; detailed scripts retain their events,
exact lines and constraints. Shot planning includes character/prop ownership,
physical transitions, first-frame authority and the final state needed for continuity.
Music videos retain the supplied song's lyrics, voices and vocal timing.

Music planning assigns visible people their vocal or instrumental roles. During
guitar and drum cutaways, prompts keep the lead singer's voice off screen and
the instrumentalist's lips closed while their playing stays active. Wide band
shots reserve lip-sync for the assigned vocalist. Explicit backing singers,
singing guitarists, wind instruments and crowd cheers retain their intended
performance. These directions apply to newly planned shots; create a new plan
to update older prompts.

H3 compilation keeps complete actions, camera destinations, dialogue delivery and
ending states. It may remove repeated boilerplate, but a cosmetic prompt-length
target cannot cut off an action or discard the end of a shot. Planning caches include
the writing contract and optional-content setting. Reviewed per-shot edits remain
authoritative. Prompt review is still useful: successful preparation does not
guarantee perfect movement, lip synchronization or visual continuity.

## H3 Director parameters

These Director request parameters are saved with the project and apply to
every H3 video job of the run, including H3 sequence units, Dashboard reruns
and repairs.

- `director_h3_sequences`: `auto` (default; on for custom story skills),
  `on` or `off`. When on, two or three consecutive shots of one scene render
  as one H3 generation joined by timed cuts
  (`[Shot 2] At 00:05.167, the camera cuts to a medium close-up of Ross.`) and
  are split back per shot. When off, every shot renders as its own generation
  from its start image (and, with a bounded strategy, the next shot of the same
  scene as end image). A shot shorter than the H3 minimum renders at the
  minimum and is trimmed back to its planned length.
- `director_h3_turbo_preset`: empty or `off` (default) keeps the model's own
  settings. A preset id from `app/models/minimax_h3/turbo_presets.json` (or the
  shorthand `pdd` for `alibaba-pai-fl2va-pdd-8step`) turns on H3 Turbo mode
  with that preset and its step count (8 for the Alibaba PAI FL2VA Acc 8-step
  PDD preset). Use it with base H3 First / Last (`minimax_h3`) for live-action
  projects: the fused Turbo checkpoint (`minimax_h3_fused_turbo`) bakes in the
  "Mystic" animation style LoRA, and ignores this parameter because its
  accelerator is already fused. An unknown preset id is rejected when the run
  is submitted. Hand-detail shots never drop below the preset's steps.

## Cinema grammar

`director_cinema_grammar` (`on` or `off`) adds a **CINEMA GRAMMAR** block of
about 2,500 characters (hard cap 2,800) to the storyboard planner prompt,
next to the bible's reference staging grammar. Without the parameter it is on
when a genre profile applies: `director_genre_profile` names one, or a custom
skill id names one (an id containing `k_drama` or `kdrama`, such as
`custom:k_drama_mini`; `micro_drama`, `anime` or `vlog` likewise). It is off
for every other skill.

The block stacks three layers:

- **Universal** (always): build beats as action -> reaction with a few
  deliberate variants; one camera move per shot, named with its speed and end
  framing, static by default; continuity (180-degree rule, screen direction,
  eyeline and action matches, constant light direction and props); scale
  changes one step at a time; AI-video robustness (one simple action per
  shot, no fast choreography, crowds in contact or readable text; body-part
  inserts for entrances and exits); acting with living eyes.
- **Format**, `director_format_profile`: `auto` (default) picks
  `vertical_short` for a portrait or square `director_aspect_ratio` (9:16,
  3:4, 1:1) and `horizontal` otherwise. `vertical_short` frames for a phone
  (one or two large faces, wides only to establish) with the DramaBoard
  shot-size mix (medium 26%, close-up 24%, medium close-up 23%, medium wide
  19%, wide 7%); `horizontal` uses the width (two-shots, over-the-shoulder
  pairs, wides that re-establish geography) with a wider guideline mix.
  Dialogue coverage (about a quarter of the shots under a line on the
  listener, 16% of line ends cut to a silent reaction) applies to both.
- **Genre**, `director_genre_profile`: `auto` (default) takes the skill's
  genre, else `neutral`; or one of `kdrama`, `micro_drama`,
  `anime_mini_series`, `vlog`, `neutral`. Each brings rhythm targets, a
  shot-mix bias, camera habits, a look, an acting style, motifs and do/don't
  rules. `kdrama` holds shots ~3-6 s (median ~4 s, silent reactions 3-6 s),
  80% static with slow push-ins on realisations, restrained acting and its
  framing motifs; `micro_drama` keeps the fast DramaBoard pacing (median
  1.7 s, 94% static) with hooks and cliffhangers; `anime_mini_series` holds
  poses, uses stronger angles and avoids smear and impact frames; `vlog` is
  handheld direct address with jump cuts and POV inserts.

When the run uses analysed style references (`director_style_reference`),
their measured numbers replace the genre and format defaults: average and
median shot length, cuts per minute, shot-size mix, static share and the
shot length under a line vs silent, averaged across references weighted by
their shot count. Every length is clamped to the video model's shot range,
and the block names the source of each number ("from your references", the
genre default, DramaBoard). The REFERENCE STAGING GRAMMAR block then leaves
its numbers to the cinema block and keeps the references' rules and patterns.

The params snapshot records the resolved values (`director_cinema_grammar`,
`director_format_profile`, `director_genre_profile`) and
`_director_cinema_grammar`: block size, target duration, shot-length range,
`format` and `genre` with their source (`param`, `aspect_ratio`, `skill`,
`default`), `calibration` (`references` or `genre_default`), the calibrating
reference ids and the resulting rhythm targets with the source of each number.

The knowledge, its sources and licenses live in
`app/services/director/cinema_grammar/` (see its README).

The compiled H3 prompt writes camera moves in a closed vocabulary: synonyms
such as "dolly in" or "zooms out" become "push-in" or "pulls back", and vague
motion ("the camera drifts") becomes "static camera". Framing and the move's
destination stay as planned. MiniMax bracket commands (`[Push in]`) are
mapped in the same table but stay off (`MINIMAX_BRACKET_COMMANDS`).

## LoRA strengths

Activated Video and Image LoRAs each show a strength slider and numeric input.
H3 has one editable strength even though it does not use a guidance-phase
schedule. Models with multiple phases retain independent controls for each
phase. Values, including zero, are saved and carried into Director projects.
Older empty weight arrays recover serialized strengths where available, or
use 1.0 when no strength was saved.

## Default image model

Shot images, cast sheets and bible renders default to **Qwen Image 2.1 7B**
(`qwen_image_21_7B`) with the **Viggle Turbo v0.2.1** profile: sampler
`viggle_v021`, 6 steps, CFG 1, the KV cache on (`qwen21_kv_cache: Enabled`).
The Qwen 2.1 handler adds the Viggle LoRA itself; it is downloaded from
Hugging Face on first use, and a generation's provenance lists it under
LoRAs. A chosen image model or sampler is always kept: the Turbo profile is
only applied when Qwen 2.1 is used without a sampler choice. A project saves
its sampler when it starts, so reruns of older projects keep theirs; a saved
project without an image model still reruns on Flux 2 Klein 9B.

The remote cache warmer does not hold LoRAs: a fresh worker downloads the
Viggle LoRA (about 7 s) on its first Qwen 2.1 job.

## Vision judges

Director has two advisory vision judges: the **storyboard judge**
(`services/director/storyboard_judge.py`) checks every shot image against its
plan before any video is rendered and ranks best-of-N drafts, and the **video
judge** (`services/director/video_judge.py`) scores each finished clip, picks
between takes and reviews the stitched episode. Both ask one backend
(`services/director/judge_backend.py`), chosen in Settings → Services under
"Director vision judges":

| Setting | Env default | Meaning |
|---|---|---|
| `director_judge_provider` | `MAESTRO_JUDGE_PROVIDER` | `fal` (default: fal's `openrouter/router/video`, billed per call) or `openai` |
| `director_judge_url` | `MAESTRO_JUDGE_URL` | base URL of an OpenAI-compatible vision server, e.g. `https://spark.tailnet:8080/v1` |
| `director_judge_api_key` | `MAESTRO_JUDGE_API_KEY` | its bearer token (llama-swap's API key); stored like the other API keys, returned masked, never logged |
| `director_judge_model` | `MAESTRO_JUDGE_MODEL` | the model name sent in every call, default `judge` |

As with the prompt LLM, an environment variable only fills a setting that
is unset: a value saved in Settings wins.

With the `openai` provider nothing goes to fal.ai and nothing is billed: the
`cost_usd` of every verdict is 0 and the review report records
`"backend": "openai"` with the model alias. The per-shot, episode and
storyboard model fields are ignored in that mode (the alias replaces them),
and a fal key is not needed. Style reference videos are analysed there too,
as silent stills (one per shot, two for a shot of 3 s or more, at most 24 per
call) sent to the alias of `director_style_reference_model` (default
`analyst`: on the Spark a Qwen3.6 vision model, the family Qwen-Image encodes
its prompts with); the measured cuts and rhythm stay exact, only the sound
column of the shot breakdown becomes a guess from the picture. The local
pass has no 15-minute bound: a whole episode goes through, and shots the
model flags as credits (opening titles, end credits, logos, title cards,
recaps, previews) are described but left out of the stats and the grammar
(`credits_shots` / `credits_sec` in the reference's stats).

An upload (up to 4 GB) is reduced to a 480p H.264 proxy (mono sound kept)
before anything reads it, and the proxy replaces the upload in the bible; the
upload's dimensions and size stay on the reference as `source`.

A reference needs no label or tags from the user: the synthesis names it
(`label`, 3-6 generic words; a link keeps its video title) and picks 2-5
`scene_tags` of the moods it guides best. Both stay automatic (`label_auto`,
`tags_auto`) until the user edits them, and a re-analysis refreshes only the
automatic ones.

After the analysis a second pass describes the first and last frame of up
to `director_style_frames` shots (default 150, evenly sampled, credits left
out; 0 = off) as the image prompt that would regenerate the frame (`prompt`,
`shot_size`, `lighting`, `palette`, `mood`; sidecar `<file>.frames.json`,
`GET .../{ref}/frames`, `POST .../{ref}/frames` to run it again, resumable).
The Shots dialog shows them under each shot. Runs that use the references
get a "REFERENCE IMAGE PROMPT STYLE" block after the staging grammar: up to
6 of those prompts spread over shot sizes, so the image prompts come out in
the same register (the image model's own family wrote them).

Each reference has a `graphics` mode (`ignore`, the default, or `include`;
a toggle in the bible panel, `graphics` on the upload form and in the PATCH
body): with `ignore` the analyst leaves burned-in subtitles, captions, logos,
watermarks and lower thirds out of the staging and never counts them as
credits; with `include` every shot gets a `graphics` description and the
grammar a `graphics` section with rules about on-screen text. Changing the
mode applies to the next analysis.

A bible keeps up to 10 references. With two or more enabled, analysed
references the controller consolidates them into one grammar on the bible
(`style_grammar`, `GET .../style-references/merged`, `POST .../merge` to
rebuild): one text call that merges rules, weighted numbers and per-tag
`scene_notes`. It is rebuilt after every analysis, edit or deletion and the
planner block uses it whenever it covers exactly the references of the run;
otherwise the references are stacked within the block's 4000 characters. Media are sent as `image_url` data URIs on
`/v1/chat/completions`:

- a still (shot image, draft candidate, character sheet) is one JPEG,
  downscaled to a 768 px long side; the prompts then speak of "image 1",
  "image 2 = Mina" instead of fal's "video 1";
- a clip is 8 evenly spaced frames (from t=0, capped at 12), each 768 px,
  preceded in the prompt by a line such as `Video 1 is shown as 8 frames at
  t=0.0s, 0.8s, ...` so the model can quote timestamps. Expect 30-120 s per
  call on a DGX Spark; a 5xx or a timeout is retried once.

The intended server is llama-server behind llama-swap on the Spark (see
`Remote-worker-static-and-llama-swap.md`), with a `judge` alias pointing at
Qwen3.6-35B-A3B and its mmproj:

```yaml
models:
  judge:
    cmd: >
      llama-server --port ${PORT}
      -m /models/Qwen3.6-35B-A3B-Q4_K_M.gguf
      --mmproj /models/Qwen3.6-35B-A3B-mmproj-F16.gguf
      -c 32768 -ngl 999 --jinja
    ttl: 600
```

Then set the provider to the local server, the URL to
`https://<spark>/v1`, the key to llama-swap's API key and the model to
`judge`. Any other OpenAI-compatible vision endpoint works the same way.

## Insights registry

Every judge finding of every run lands in one SQLite file,
`<outputs>/_director_insights.sqlite` (`services/director/insights.py`), so
recurring defects can be read per scene type and per model. Three tables:
`runs` (models, skill, bible, draft count, judge models, episode score and
verdict), `shots` (scene type, framing, movement, duration, storyboard
verdict, draft candidates/rounds/flagged, clip score) and `findings` (one row
per storyboard issue, draft candidate issue, clip issue or episode defect,
with its category, severity, `fixable_by` / `root_cause`, scene type, framing
and models).

A run is ingested at the end of its storyboard judge stage and again when its
video review finishes; both are best effort and never fail the run. A run
without a review is recorded with its storyboard findings only. Ingestion
replaces the run's rows, so re-running it is safe.

- `GET /api/v1/director/insights?days=&image_model=&video_model=&skill=`:
  run count, average episode score and trend, findings by (source, category)
  and by scene type, the 20 most frequent descriptions with their categories
  and runs, draft stats per scene type (candidates, flagged ratio, rounds) and
  the models present.
- `POST /api/v1/director/insights/backfill`: ingest every saved run of the
  outputs folder (workspaces included); returns the counts.
- CLI, from `app/`: `python -m services.director.insights backfill <outputs>`
  and `python -m services.director.insights summary <outputs> [--days N]
  [--image-model M] [--video-model M] [--skill S]` (JSON on stdout).

### Skill versions and automatic rollback

Every save of a custom skill keeps the replaced version under
`settings/director_skills/history/<id>/v<N>.json`, and each run records the
skill version it planned with (`skill_version` in the pipeline state and in
the registry's `runs` table). The versions can then be compared on their
reviewed runs and a worse one rolled back:

- `GET /api/v1/director/skills/{id}/revisions`: every kept version with its
  date, origin (`restored_from`) and, from the registry, its reviewed runs,
  mean episode score and mean clip score.
- `POST /api/v1/director/skills/{id}/rollback` with
  `{"apply": false|true, "min_runs": 3, "margin": 0.5}`: the decision
  (`revert_to`, `reason`); the current version is judged once it has
  `min_runs` reviewed runs, and loses to an earlier version with as many
  runs when that one's mean episode score is higher by more than `margin`
  (the run-to-run noise of the judge). With `apply`, the earlier version is
  restored as a new version that names its origin and the reason.
- Batch runner: `scripts/director_batch.py ... --guard-skill custom:k_drama_mini
  [--guard-min-runs 3] [--guard-margin 0.5]` calls that route after every
  reviewed run and logs the decision, so a nightly batch never keeps a skill
  revision that renders worse than the one before.
- CLI: `python -m services.director.insights versions <outputs> custom:k_drama_mini`.

## Recommended next UI pass

These are proposed changes, not yet implemented:

- Replace the permanently expanded resolution, aspect and target-duration
  controls with compact indicators that show their current values and open
  menus sized to the Director panel. Preserve Director's own model capabilities
  and settings rather than changing the active Studio generation settings.
- Group H3 optimizations, LoRAs and finishing under one Advanced area with
  collapsible sections. Show only the sections applicable to the chosen models.
- Use a compact Characters entry and consistent reference tiles across H3 and
  other Director models, retaining character, location, voice and soundtrack
  roles. Keep both video and optional image model choices available.

Retain Director's skill selection, staged planning and prompt review, plus its
existing Start / Queue actions. Studio's explicit Enhance action should not
replace the Director planning workflow.

## Local validation

After building `ui`, set `MAESTRO_UI_DIRECTOR_ONLY=1` and run
`node tests/ui/sidebar_redesign.cjs http://127.0.0.1:<Maestro port>`.
The existing isolated browser harness uses test projects and mocked writes;
it does not submit a real plan or generation. Checks cover horizontal bounds,
vertical scrolling, long descriptions and logs, character and speaker inputs,
and simulated mobile keyboard geometry at desktop, 390px and 320px widths.
The same command checks duration takeover from Auto, native steps, long
presets, H3 and multi-phase LoRA weights, saved strength recovery, and the
duration/weights in an intercepted queue request.
Real iOS keyboard animation still needs device validation.
