# Director blockout guide (phase 1)

The storyboard locks one floor plan per scene: people stay where they are,
and the camera stays on one side of the action line
(`services/director/blocking_map.py`). Image models still get the room wrong
between shots. In a reverse or over-the-shoulder shot, the café window can
appear on the wrong side of the frame. A depth map keeps the composition but
cannot separate a wall from a window. The blockout renders each shot as a
rough 3D set, including a **semantic colour map**, so the window has its own
colour on its own side of the frame.

Code: `app/services/director/blockout/` (`layout.py`, `scene.py`,
`render.py`, `guide.py`). Tests: `tests/test_director_blockout.py`.

## Enabling it

`director_blockout_guide` in the Director request: `off` (default), `depth`
or `semantic`. When it is off, nothing is rendered and no job or prompt
changes. There is no UI control yet; pass the setting in the request
parameters.

When it is on, each scene shot (a shot with a continuity group and a
blocking map; inserts and shots outside a scene are skipped) gets these
files next to its images:

- `<pid>_shotNN_blockout_depth.png`
- `<pid>_shotNN_blockout_semantic.png`
- `<pid>_shotNN_blockout_normal.png`

The shot record (`clips[i].blockout_guide` in the saved state) stores the
chosen image, every file name, the camera (position, target, field of view,
35 mm-equivalent focal length), the layout source, each person's slot and
posture, the over-the-shoulder foreground person, the render time and
`set_sides` (for example `{"window": "frame left"}`).

- **Qwen Image 2.1** (`qwen_image_21*`): the chosen guide is sent as the raw
  control image (`image_guide`, `video_prompt_type` `V` + `I`, with `K`
  dropped). The control image takes the first conditioning slot, so the
  prompt describes it as reference image 1 and renumbers the other
  references.
- **Other models**: no control image. The prompt only gets the set's screen
  sides ("Set layout: the window on frame left.").

## Location layout schema

A bible location can store `layout`. A save without the key keeps the stored
layout, and `null` clears it. The layout is copied into the project cast.
Look-up order for a shot's place (`_director_location`):

1. The `director_location_layouts` request parameter (`{name: layout}`).
2. The bible layout.
3. A keyword layout from the description and notes.
4. The default empty 6 × 5 × 2.8 m room.

```json
{
  "version": 1,
  "room": {"width": 6.0, "depth": 5.0, "height": 2.8},
  "openings": [
    {"wall": "left", "kind": "window", "center": 0.5, "width": 2.5, "bottom": 0.3, "height": 2.2, "name": "street window"},
    {"wall": "front", "kind": "door", "center": 0.15, "width": 0.9, "bottom": 0.0, "height": 2.1, "name": "door"}
  ],
  "furniture": [
    {"name": "espresso bar", "class": "counter", "shape": "box", "x": 0.9, "y": 0.45,
     "width": 0.6, "depth": 2.4, "height": 1.05, "rotation_deg": 0},
    {"name": "table", "class": "table", "shape": "cylinder", "x": 0.5, "y": 0.5,
     "width": 0.8, "depth": 0.8, "height": 0.75, "rotation_deg": 0}
  ]
}
```

- `wall`: `back | front | left | right`. `kind`: `window | door`.
  `center` is a position from 0 to 1 along the wall: left to right on the
  back and front walls, back to front on the left and right walls. `width`,
  `bottom` (sill height) and `height` are in meters.
- Furniture `class`: `table, counter, shelf, chair, sofa, bed, cabinet,
  plant, lamp, prop`. An unknown class is inferred from the name.
  `x` and `y` use the floor plan's normalized coordinates, so they share
  space with the blocking props. Sizes are in meters.
- `validate_layout` clamps room sizes and openings, drops unknown walls or
  kinds and furniture outside the floor, and reports overlapping openings.
- Blocking props of a furniture class (for example "table") move the
  nearest layout piece of that class to the prop's locked place, or add one
  there. The people are staged around the prop.

**First layout.** `POST /api/v1/director/bibles/{id}/elements/{element_id}/layout/suggest`
returns `{layout, source: "llm" | "keywords", problems}`, but does not save
it. The LLM, using the existing `llm_service.generate`, writes JSON that is
then validated. Without the LLM (`"use_llm": false`), or if it fails, the
keyword rules apply: a window the text does not place goes on the back wall,
the wall the place's main image looks at. Save the bible with the returned
layout to keep it.

## Coordinate conventions

- The floor plan (from the storyboard) is a top view seen from the master
  camera. `x` runs from 0 (left wall) to 1 (right wall), and `y` from 0
  (back wall) to 1 (front wall). A facing of 0° looks at the back wall and
  90° at the right wall.
- The 3D world is in meters with Y up. X = `x · width`, Z = `y · depth`, so
  the back wall is at Z = 0. Facings stretch with the room, so a person
  facing another on the plan also faces them in 3D.
- **Camera.** Its position and facing come from the scene-locked blocking,
  which already respects the 180° rule. Its horizontal field of view uses
  the blocking diagram's value for each framing: 70° wide, 42° medium, 22°
  close-up. Anyone the diagram fits in its wedge therefore stays in frame;
  in a portrait frame this makes the vertical field taller. Height and tilt
  follow the framing (face, chest or body). Angle words adjust it: "high"
  and "low" raise or lower the camera, "overhead" puts it near the ceiling,
  and "dutch" adds a 12° roll. An over-the-shoulder camera stands at the
  foreground person's head height. A camera inside a person steps back out.
- **Rooms.** Walls are visible from inside only. A camera placed outside the
  room sees through the near wall, like a set with that wall removed.
- **Mannequins.** Box legs, torso and shoulders with a sphere head; seated
  figures also get a chair. A person is seated when the shot's words seat
  them, judged by the sentence that names them, otherwise by the whole text.
  Slots are stable per scene, with the most present person in slot 0 (see
  `scene_character_slots`).

## Palette

Set classes use the ADE20K colours that segmentation control models were
trained on. People get one colour per slot.

| class | RGB | class | RGB |
|---|---|---|---|
| wall | 120,120,120 | counter | 235,12,255 |
| floor | 80,50,50 | shelf | 255,7,71 |
| ceiling | 120,120,80 | chair | 204,70,3 |
| window | 230,230,230 | sofa | 11,102,255 |
| door | 8,255,51 | bed | 204,5,255 |
| table | 255,6,82 | cabinet | 224,5,255 |
| plant | 204,255,4 | lamp | 224,255,8 |
| prop | 0,255,20 | void | 0,0,0 |
| character 1 (ADE person) | 150,5,61 | character 2 | 0,190,255 |
| character 3 | 255,140,0 | character 4 | 120,255,60 |
| character 5 | 170,80,255 | character 6 | 255,90,190 |

**Depth** is z-depth. Lighter is closer, the frame's own range is used (a
sequence shares one range), and empty space is black. **Normals** are in
camera space: R is frame right, G is up and B points toward the camera.
**Edges** mark changes in class, depth or surface direction.

## Performance

Rendering uses numpy ray casting on the CPU with one ray per pixel and no
antialiasing. A furnished room with two seated people renders in about
170 ms at 512 × 288 and about 380 ms at 432 × 768 (the guide size for a
864 × 1536 shot). The guide's long side is 768 px.

## Not wired yet / next steps

- **Driving video.** `scene.camera_path(camera, movement, intensity, frames)`
  turns the planner's move words (push-in, pull-out, pan, tilt, truck,
  orbit, crane, zoom, handheld) into eased camera poses. `render_sequence`
  and `sequence_depth_range` render them without flicker. The pipeline does
  not write these sequences yet. The next step is to encode them as a
  control video for video-to-video or control models (Wan control, H3 if it
  gains a control input) and compare them with H3 image-to-video drift.
- **Measuring raw control.** Raw `V` control with a semantic map has not
  been checked against generated images. The prototype only checked depth,
  and depth held the composition but not the window side. Before making
  `semantic` a default, run an A/B test on a scene with a reverse pair.
- **Other entry points.** The Dashboard start-image rerun, keyframes and
  `image_source: previous` shots do not send the guide yet. Only the main
  storyboard pass does.
- **UI.** There is no toggle for `director_blockout_guide`, no layout
  editor and no guide preview on the shot card. The record already has the
  file names.
- **Automatic layout from bible images.** Estimate the room box, the window
  and door positions and the large furniture from the location's main and
  reverse images (monocular depth plus segmentation). Then fit the schema
  above so the user only checks it.
- **Placement.** Postures are a text heuristic and mannequins are generic.
  Per-person height, props held in hand and standing-to-seated moves within a
  shot remain to be done.
