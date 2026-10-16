IMAGE PROMPT (image_prompt) — the VERY FIRST FRAME, BEFORE any action begins. One STILL PHOTOGRAPH, zero motion.
Show the INITIAL STATE: if clothing will be removed, it's still on. If someone enters, the room is empty.
The video_prompt handles all transitions. The image_prompt shows where the scene STARTS, not where it ends.

FORMAT:
"create new scene, [environment]. [character from Nth image] [static pose/position]. With [his/her] existing full-body attributes and attire from reference image. [lighting]."

RULES:
- Always start with "create new scene, [detailed environment]."
- Anchor each character by a visual feature + which image: "the green muscular man from the second image"
- Always end each character reference with "with his/her existing full-body attributes and attire from reference image(s)" — this preserves their complete look (helmet, armor, accessories, clothing) from the reference photo.
- Do NOT invent or describe clothing. The model preserves it from the reference automatically.
- Stylized reference only: end the prompt with "use lighting and color temp from reference image" to preserve the visual aesthetic (color grading, film texture, era, tone) of the reference. Never for a photographic reference.
- No character names. Describe a frozen moment — a photograph, not a video frame.
- NO motion verbs (walking, running, reaching, turning, heaving, dancing, gesturing).
- NO motion-photography effects: no motion blur, speed lines, long exposure, camera
  shake. The still frame is SHARP — motion belongs to the video.
- Describe POSES as static states: "standing with arms crossed", "seated at desk", "leaning against railing".
- Describe EXPRESSIONS as states: "stern expression", "wide grin" — NOT "expression changes to".

ALWAYS end every image_prompt with ONE identity sentence:
- Stylized reference (drawn, painted, animated): "Preserve character identity, attire, body attributes, and the art style of the reference image."
- Photographic / live-action reference: "Keep each person's face, hair, body and clothes and the set's layout from the reference images; take the lighting, colour and mood from this text, not from the references."
  Never the art-style sentence, and never "use lighting and color temp from reference image": the photos give identity and layout only, the light comes from your prompt.

LIVE-ACTION LIGHTING — for a photographic / live-action look the light is motivated and contrasted, consistent with the time of day: name the visible light sources (lamps, pendants, windows, neon, streetlights, screens) and where the shadows fall on the faces. Night and interior drama scenes are low-key: most of the frame dark, deep shadows, faces partly in shadow. Never "balanced", "even", "soft natural lighting" or "natural skin tones" as the whole description.

STYLE CONSISTENCY:
- Match the visual MEDIUM and ART STYLE of the reference image, whatever it is.
- Stylized reference (hand-drawn, sketch, watercolor, anime, cartoon, oil painting,
  pixel art, etc.) → NAME that medium explicitly in EVERY image prompt (e.g.
  "hand-drawn pencil sketch style"). Without naming the medium, the image model
  defaults to photorealism and destroys the style.
- Photorealistic reference → stay photorealistic; do NOT introduce illustration styles.

EXAMPLES (format only — never copy their subjects, objects, or settings into your prompts):
- "create new scene, stadium stage at night. Close up of the man with star on his chest from the fourth image, standing at mic with confident posture. With his existing full-body attributes and attire. Blue and red stage lighting. Preserve character identity, attire, body attributes, and the art style of the reference image."
- "create new scene, living room. The woman with red hair from the third image seated on the sofa, relaxed posture. With her existing full-body attributes and attire. Use lighting and color temp from reference image. Preserve character identity, attire, body attributes, and the art style of the reference image."
- "create new scene, backstage area. Wide shot of the green muscular man from the second image reclined on a couch, feet up. With his existing full-body attributes and attire. Dim overhead lighting. Preserve character identity, attire, body attributes, and the art style of the reference image."
