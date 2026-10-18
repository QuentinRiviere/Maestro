"""Crop an insert's draft below the faces it should not show.

Image editors keep drawing an insert ("hands and the mug on the table") as
a two-shot of the people the text names, whatever the prompt's lead says
(every candidate of run c88357e3's insert was flagged "faces visible"). When
the chosen draft still shows faces, the frame is cut back to the band below
them, at the shot's aspect ratio and centred under the faces, where the
hands and the objects are; the final pass then renders that crop at full
resolution. Faces come from OpenCV's Haar cascades (frontal and profile),
which the controller image ships; a detector can be injected for tests.
"""
from __future__ import annotations

import os
from typing import Callable, Optional, Sequence

Box = tuple[int, int, int, int]  # x, y, w, h

MIN_BAND = 0.35      # the band below the faces must keep this share of the height
FACE_MARGIN = 0.04   # of the height, kept clear under the lowest face (chins, collars)


def detect_faces(path: str) -> list[Box]:
    """Face boxes in the image (frontal and profile Haar cascades); [] without OpenCV."""
    try:
        import cv2
    except ImportError:
        return []
    image = cv2.imread(path)
    if image is None:
        return []
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_size = max(24, int(min(image.shape[:2]) * 0.06))
    boxes: list[Box] = []
    for name in ("haarcascade_frontalface_default.xml", "haarcascade_profileface.xml"):
        cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, name))
        if cascade.empty():
            continue
        found = cascade.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size))
        boxes.extend((int(x), int(y), int(w), int(h)) for x, y, w, h in (found if len(found) else []))
    return boxes


def band_below_faces(size: tuple[int, int], faces: Sequence[Box]) -> Optional[Box]:
    """The largest window at the image's aspect ratio below ``faces`` (None when too small).

    The window spans the height left under the lowest face (plus a margin),
    is as wide as that height allows at the image's ratio, and is centred on
    the faces' mean x, clamped inside the image.
    """
    width, height = size
    if not faces or width <= 0 or height <= 0:
        return None
    bottom = max(y + h for _, y, _, h in faces) + int(height * FACE_MARGIN)
    band_height = height - bottom
    if band_height < height * MIN_BAND:
        return None
    band_width = min(width, int(round(band_height * width / height)))
    band_height = int(round(band_width * height / width))
    centre = sum(x + w / 2 for x, _, w, _ in faces) / len(faces)
    left = int(round(centre - band_width / 2))
    left = max(0, min(width - band_width, left))
    top = max(0, height - band_height)
    return left, top, band_width, band_height


def crop_below_faces(source_path: str, dest_path: str,
                     detect: Callable[[str], Sequence[Box]] = detect_faces) -> Optional[str]:
    """``dest_path`` holding ``source_path`` cut below its faces, or None when nothing to cut.

    None when no face is found or the band left would be too small to
    carry the shot; the caller then keeps the draft as it is.
    """
    from PIL import Image

    faces = list(detect(source_path))
    if not faces:
        return None
    with Image.open(source_path) as image:
        band = band_below_faces(image.size, faces)
        if band is None:
            return None
        left, top, width, height = band
        image.convert("RGB").crop((left, top, left + width, top + height)).save(dest_path)
    return dest_path
