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


YUNET_MODEL_ENV = "MAESTRO_YUNET_MODEL"   # path of face_detection_yunet_*.onnx for OpenCV 5 (no Haar cascades)
ASSUMED_FACES_ABOVE = 0.45               # the judge saw faces: they sit in the upper part of a two-shot


def detect_faces(path: str) -> list[Box]:
    """Face boxes in the image: Haar cascades (OpenCV 4), else YuNet when its model is given; [] otherwise.

    OpenCV 5 wheels ship neither the cascades nor ``CascadeClassifier``
    (run b6b1d888 lost its final pass on that AttributeError), so every
    failure here is a plain "no face found" and the caller falls back on
    the judge's verdict.
    """
    try:
        import cv2
    except ImportError:
        return []
    try:
        image = cv2.imread(path)
        if image is None:
            return []
        boxes: list[Box] = []
        cascades = getattr(cv2, "CascadeClassifier", None)
        data = getattr(getattr(cv2, "data", None), "haarcascades", "")
        if cascades and data:
            grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            min_size = max(24, int(min(image.shape[:2]) * 0.06))
            for name in ("haarcascade_frontalface_default.xml", "haarcascade_profileface.xml"):
                cascade = cascades(os.path.join(data, name))
                if cascade.empty():
                    continue
                found = cascade.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size))
                boxes.extend((int(x), int(y), int(w), int(h)) for x, y, w, h in (found if len(found) else []))
            if boxes:
                return boxes
        model = os.environ.get(YUNET_MODEL_ENV, "")
        creator = getattr(cv2, "FaceDetectorYN_create", None) or getattr(cv2, "FaceDetectorYN", None)
        if model and os.path.isfile(model) and creator:
            height, width = image.shape[:2]
            detector = creator(model, "", (width, height), 0.7, 0.3, 500)
            _, faces = detector.detect(image)
            if faces is not None:
                boxes.extend((int(f[0]), int(f[1]), int(f[2]), int(f[3])) for f in faces)
        return boxes
    except Exception:  # noqa: BLE001 - a detector failure is "no face found"
        return []


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


def judge_saw_faces(issues: Sequence[str]) -> bool:
    """Whether the storyboard judge's issues say the insert shows people instead of hands."""
    text = " ".join(str(issue) for issue in issues).lower()
    return any(word in text for word in ("face", "head", "torso", "upper bod", "medium shot", "two-shot",
                                         "medium close-up", "characters"))


def crop_below_faces(source_path: str, dest_path: str,
                     detect: Callable[[str], Sequence[Box]] = detect_faces,
                     assume_faces_above: Optional[float] = None) -> Optional[str]:
    """``dest_path`` holding ``source_path`` cut below its faces, or None when nothing to cut.

    None when no face is found or the band left would be too small to
    carry the shot; the caller then keeps the draft as it is. Without a
    detected face, ``assume_faces_above`` (a share of the height, from the
    judge's word that the draft shows people) stands in for one face
    spanning the width down to that line.
    """
    from PIL import Image

    faces = list(detect(source_path))
    with Image.open(source_path) as image:
        if not faces and assume_faces_above:
            faces = [(0, 0, image.width, int(image.height * assume_faces_above))]
        if not faces:
            return None
        band = band_below_faces(image.size, faces)
        if band is None:
            return None
        left, top, width, height = band
        image.convert("RGB").crop((left, top, left + width, top + height)).save(dest_path)
    return dest_path
