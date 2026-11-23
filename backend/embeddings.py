"""Feature extraction for the face attendance classifier.

The saved classifier expects a 1280-dimensional vector. Two input modes are
supported so the API can be matched to the way the model was trained:

  * ``features`` (default): MobileNetV2 ImageNet penultimate features.
  * ``pixels``: a 40x32 grayscale face crop, flattened and scaled to 0..1.

Pick the mode with the ``FACE_EMBEDDING_MODE`` environment variable. If you
still have the original training code, replace ``extract_embedding`` with the
exact same preprocessing used there -- the classifier itself stays unchanged.
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageOps

EMBEDDING_DIM = 1280
PIXEL_SIZE = (40, 32)  # 40 * 32 = 1280

_MODEL = None
_PREPROCESS = None


def _get_feature_extractor():
    """Lazily load MobileNetV2 once (the weights are ~14 MB)."""
    global _MODEL, _PREPROCESS
    if _MODEL is None:
        import torch
        from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

        weights = MobileNet_V2_Weights.DEFAULT
        model = mobilenet_v2(weights=weights)
        # Drop the ImageNet classification head -> 1280-d feature vector.
        model.classifier = torch.nn.Identity()
        model.eval()
        _MODEL = (model, torch)
        _PREPROCESS = weights.transforms()
    return _MODEL[0], _MODEL[1], _PREPROCESS


def crop_face(image: Image.Image) -> Image.Image:
    """Best-effort face crop, with a safe centre-crop fallback.

    A small YCbCr skin-tone heuristic is enough for the close-up portraits a
    kiosk-style attendance app receives. It is deliberately conservative: if
    nothing convincing is found, the centre square of the image is used.
    """
    img = ImageOps.exif_transpose(image).convert("RGB")
    width, height = img.size
    if not width or not height:
        raise ValueError("Image has no pixels.")

    try:
        ycbcr = np.asarray(img.convert("YCbCr"))
        cb, cr = ycbcr[:, :, 1], ycbcr[:, :, 2]
        skin = (cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173)
        coverage = float(skin.mean())

        if 0.02 < coverage < 0.90:
            rows = np.where(skin.sum(axis=1) > width * 0.05)[0]
            cols = np.where(skin.sum(axis=0) > height * 0.05)[0]
            if rows.size > 5 and cols.size > 5:
                left, right = int(cols[0]), int(cols[-1])
                top, bottom = int(rows[0]), int(rows[-1])
                box_w, box_h = right - left, bottom - top
                if box_w >= width * 0.08 and box_h >= height * 0.08:
                    pad_x, pad_y = int(box_w * 0.15), int(box_h * 0.15)
                    left, right = max(0, left - pad_x), min(width, right + pad_x)
                    top, bottom = max(0, top - pad_y), min(height, bottom + pad_y)
                    if right > left and bottom > top:
                        return img.crop((left, top, right, bottom))
    except Exception:
        # A failed heuristic must never break a prediction.
        pass

    side = min(width, height)
    return img.crop(
        ((width - side) // 2, (height - side) // 2,
         (width - side) // 2 + side, (height - side) // 2 + side)
    )


def extract_embedding(image: Image.Image) -> np.ndarray:
    """Return a flat float32 vector of length ``EMBEDDING_DIM``."""
    face = crop_face(image)
    mode = os.environ.get("FACE_EMBEDDING_MODE", "features").strip().lower()

    if mode == "pixels":
        gray = face.convert("L").resize(PIXEL_SIZE, Image.Resampling.LANCZOS)
        vector = np.asarray(gray, dtype=np.float32).reshape(-1) / 255.0
    elif mode in ("features", "mobilenet", "mobilenetv2"):
        model, torch, preprocess = _get_feature_extractor()
        tensor = preprocess(face).unsqueeze(0)
        with torch.no_grad():
            vector = model(tensor).detach().cpu().numpy().reshape(-1)
        vector = vector.astype(np.float32)
    else:
        raise ValueError(
            f"Unknown FACE_EMBEDDING_MODE={mode!r}. Use 'features' or 'pixels'."
        )

    if vector.shape[0] != EMBEDDING_DIM:
        raise ValueError(
            f"Extractor produced {vector.shape[0]} features; the model needs {EMBEDDING_DIM}."
        )
    return vector
