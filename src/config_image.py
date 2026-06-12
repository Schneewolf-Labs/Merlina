"""
Encode / decode a Merlina training config as a single shareable PNG.

Instead of pasting a giant JSON blob into a model README, Merlina can publish
one ``merlina_config.png`` that carries the (secret-stripped) training config
two complementary ways:

  1. **A scannable QR code** of the gzip+base64-compressed config envelope —
     phone-friendly, capacity-bounded, so we compress before encoding.
  2. **The full, uncompressed envelope JSON** embedded in a PNG ``tEXt`` chunk
     under :data:`PNG_TEXT_KEY`. This is the lossless channel Merlina
     re-imports from — reading it needs only Pillow, no QR-decode system deps.

Generation is intentionally **best-effort**: if the optional ``qrcode``
dependency (or Pillow) is missing, :func:`build_config_image_bytes` returns
``None`` rather than raising, so a model upload is never aborted by a missing
optional dep. Decoding the metadata channel needs only Pillow; the QR fallback
additionally needs OpenCV (``cv2``) and is skipped silently when unavailable.

The envelope shape is the same one produced by ``/configs/save`` and the
README embedder (see ``src/config_manager.build_config_envelope``), so anything
extracted here drops straight into the *Load Configuration* flow.
"""

import base64
import gzip
import io
import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Key under which the full, uncompressed config envelope JSON lives in the
# PNG's tEXt metadata. This is the lossless channel Merlina re-imports from.
PNG_TEXT_KEY = "merlina-config"

# Prefix on the QR payload so a decoder can recognise a Merlina config blob
# and version the compressed wire format independently of the JSON schema.
_QR_PREFIX = "merlina-config-v1:"


def _compress_envelope(envelope: Dict[str, Any]) -> str:
    """gzip + urlsafe-base64 the envelope into a compact, prefixed QR payload."""
    raw = json.dumps(envelope, separators=(",", ":")).encode("utf-8")
    packed = gzip.compress(raw, compresslevel=9)
    return _QR_PREFIX + base64.urlsafe_b64encode(packed).decode("ascii")


def _decompress_payload(payload: str) -> Optional[Dict[str, Any]]:
    """Inverse of :func:`_compress_envelope`. Returns None on any mismatch."""
    if not payload.startswith(_QR_PREFIX):
        return None
    b64 = payload[len(_QR_PREFIX):]
    try:
        packed = base64.urlsafe_b64decode(b64.encode("ascii"))
        raw = gzip.decompress(packed)
        data = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def build_config_image_bytes(envelope: Dict[str, Any]) -> Optional[bytes]:
    """Render a shareable PNG carrying ``envelope`` as a QR code + tEXt metadata.

    Args:
        envelope: A config envelope dict (``_metadata`` + TrainingConfig
            fields) — already secret-stripped by the caller.

    Returns:
        PNG bytes, or ``None`` if Pillow/``qrcode`` are unavailable or
        rendering fails. Callers treat ``None`` as "skip the artifact".
    """
    try:
        import qrcode
        from qrcode.constants import ERROR_CORRECT_L
        from PIL import Image, ImageDraw, ImageFont
        from PIL.PngImagePlugin import PngInfo
    except Exception as exc:  # pragma: no cover - optional dep
        logger.warning(
            "config image: qrcode/Pillow unavailable (%s); skipping artifact", exc
        )
        return None

    try:
        # 1. QR code of the compressed envelope (the human/phone channel).
        payload = _compress_envelope(envelope)
        qr = qrcode.QRCode(error_correction=ERROR_CORRECT_L, border=4, box_size=8)
        qr.add_data(payload)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
        qr_w, qr_h = qr_img.size

        # 2. A caption band so the model page shows what the image is for.
        name = ""
        meta = envelope.get("_metadata") if isinstance(envelope, dict) else None
        if isinstance(meta, dict):
            name = str(meta.get("name") or "")
        caption = f"Merlina config — {name}" if name else "Merlina training config"

        band_h = 36
        canvas = Image.new("RGB", (qr_w, qr_h + band_h), "white")
        canvas.paste(qr_img, (0, band_h))
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        try:
            text_w = draw.textlength(caption, font=font)
        except AttributeError:  # pragma: no cover - very old Pillow
            text_w = len(caption) * 6
        draw.text(
            (max(0, (qr_w - text_w) / 2), max(0, (band_h - 11) / 2)),
            caption,
            fill="black",
            font=font,
        )

        # 3. Lossless full envelope in PNG metadata (the re-import channel).
        png_meta = PngInfo()
        png_meta.add_text(PNG_TEXT_KEY, json.dumps(envelope, indent=2))

        buf = io.BytesIO()
        canvas.save(buf, format="PNG", pnginfo=png_meta)
        return buf.getvalue()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("config image: render failed (%s); skipping artifact", exc)
        return None


def extract_config_envelope_from_png(data: bytes) -> Optional[Dict[str, Any]]:
    """Recover a config envelope from a Merlina config PNG.

    Tries the lossless tEXt metadata channel first (Pillow only). Falls back
    to decoding the embedded QR code when OpenCV (``cv2``) is available.

    Args:
        data: Raw PNG bytes.

    Returns:
        The config envelope dict, or ``None`` if no Merlina config could be
        recovered (no matching metadata and no decodable QR).
    """
    # 1. PNG tEXt metadata — preferred, lossless, Pillow-only.
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(data))
        texts: Dict[str, Any] = {}
        texts.update(getattr(img, "text", None) or {})
        for k, v in (getattr(img, "info", None) or {}).items():
            if isinstance(v, str):
                texts.setdefault(k, v)
        blob = texts.get(PNG_TEXT_KEY)
        if blob:
            obj = json.loads(blob)
            if isinstance(obj, dict):
                return obj
    except Exception as exc:
        logger.debug("config image: metadata read failed (%s)", exc)

    # 2. QR fallback — optional OpenCV dependency.
    try:
        import numpy as np
        import cv2
        from PIL import Image

        img = Image.open(io.BytesIO(data)).convert("RGB")
        arr = np.array(img)[:, :, ::-1]  # RGB -> BGR for OpenCV
        detector = cv2.QRCodeDetector()
        payload, _, _ = detector.detectAndDecode(arr)
        if payload:
            obj = _decompress_payload(payload)
            if obj is not None:
                return obj
    except Exception as exc:
        logger.debug("config image: QR decode unavailable/failed (%s)", exc)

    return None
