"""Render-size rules for Qwen Image 2.1 checkpoints that need a minimum canvas.

The official Turbo checkpoint is distilled at its native ~2K presets. Small
requests (Director asks for 480x848) are therefore rendered at a larger canvas
with the same aspect ratio and the decoded image is downscaled afterwards.
These helpers stay free of model imports so they can be unit tested on CPU.
"""

import math

import torch
import torch.nn.functional as F
from PIL import Image

# Every Qwen Image 2.1 canvas must be a multiple of the VAE stride (16) times
# the 2x2 latent packing.
CANVAS_MULTIPLE = 32


def render_dimensions(width, height, min_pixels, multiple=CANVAS_MULTIPLE):
    """Return the canvas to denoise for a requested ``width`` x ``height``.

    A request at or above ``min_pixels`` keeps the historic rule (floor to the
    canvas multiple). A smaller request is scaled up to about ``min_pixels``
    with the same aspect ratio, each side rounded to the canvas multiple.
    """
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("Qwen Image 2.1 needs a positive output size.")
    if not min_pixels or width * height >= int(min_pixels):
        return max(multiple, width // multiple * multiple), max(multiple, height // multiple * multiple)
    scale = math.sqrt(int(min_pixels) / (width * height))
    return (max(multiple, round(width * scale / multiple) * multiple),
            max(multiple, round(height * scale / multiple) * multiple))


def downscale_reference(image, max_side):
    """Shrink a PIL reference so its long side is at most ``max_side`` pixels.

    Smaller images, and a falsy ``max_side``, are returned unchanged.
    """
    if not max_side:
        return image
    max_side = int(max_side)
    long_side = max(image.size)
    if long_side <= max_side:
        return image
    scale = max_side / long_side
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def resize_uint8_images(images, width, height):
    """Resize a ``(batch, channels, height, width)`` uint8 tensor on CPU.

    Antialiased bicubic filtering keeps the downscale free of aliasing. Alpha
    is resized like any other channel, which keeps transparent outputs intact.
    """
    width, height = int(width), int(height)
    if tuple(images.shape[-2:]) == (height, width):
        return images
    resized = F.interpolate(images.float(), size=(height, width), mode="bicubic",
                            align_corners=False, antialias=True)
    return resized.round_().clamp_(0, 255).to(torch.uint8)
