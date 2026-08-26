"""Foreground isolation for the DimensionX 3D spatial asset pipeline."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

ImageInput = Union[str, Path, Image.Image, np.ndarray]

__all__ = [
    "BackgroundRemovalError",
    "BackgroundRemover",
    "remove_background",
]


class BackgroundRemovalError(Exception):
    """Raised when background removal cannot be performed."""


class BackgroundRemover:
    """Remove image backgrounds with rembg and return RGBA cutouts."""

    def __init__(self, model_name: str = "u2net") -> None:
        self.model_name = model_name
        self._session = None

    def remove_background(self, image_input: ImageInput) -> Image.Image:
        """Strip the background from a PIL Image, NumPy array, or file path."""
        image = self._to_pil(image_input)
        _, rembg_remove = self._import_rembg()
        try:
            result = rembg_remove(image, session=self._get_session())
        except BackgroundRemovalError:
            raise
        except Exception as exc:
            raise BackgroundRemovalError(f"Background removal failed: {exc}") from exc
        return self._ensure_rgba(result)

    def _get_session(self):
        if self._session is None:
            new_session, _ = self._import_rembg()
            try:
                self._session = new_session(self.model_name)
            except BackgroundRemovalError:
                raise
            except Exception as exc:
                raise BackgroundRemovalError(
                    f"Failed to initialize rembg session '{self.model_name}': {exc}"
                ) from exc
        return self._session

    @staticmethod
    def _import_rembg():
        try:
            from rembg import new_session, remove
        except SystemExit as exc:
            raise BackgroundRemovalError(
                'No onnxruntime backend found. Install CPU support with: pip install "rembg[cpu]"'
            ) from exc
        except ImportError as exc:
            raise BackgroundRemovalError(
                'rembg is not installed. Install it with: pip install "rembg[cpu]"'
            ) from exc
        return new_session, remove

    def _to_pil(self, image_input: ImageInput) -> Image.Image:
        if image_input is None:
            raise BackgroundRemovalError("Image input is empty.")

        if isinstance(image_input, Image.Image):
            return self._validate(image_input)

        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.is_file():
                raise BackgroundRemovalError(f"Image file not found: {path}")
            try:
                with Image.open(path) as image:
                    image.load()
                    return self._validate(ImageOps.exif_transpose(image.copy()))
            except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
                raise BackgroundRemovalError(
                    f"Invalid or corrupted image file: {path}"
                ) from exc

        if isinstance(image_input, np.ndarray):
            return self._validate(self._numpy_to_pil(image_input))

        raise BackgroundRemovalError(
            f"Unsupported image input type: {type(image_input).__name__}. "
            "Expected a PIL Image, NumPy array, or file path."
        )

    @staticmethod
    def _numpy_to_pil(array: np.ndarray) -> Image.Image:
        if array.size == 0:
            raise BackgroundRemovalError("Image array is empty.")

        arr = np.asarray(array)
        if arr.ndim == 3 and arr.shape[0] in {1, 3, 4} and arr.shape[0] < min(arr.shape[1:]):
            arr = np.transpose(arr, (1, 2, 0))

        if np.issubdtype(arr.dtype, np.floating):
            scale = 255.0 if np.nanmax(arr) <= 1.5 else 1.0
            arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L")
        if arr.ndim != 3:
            raise BackgroundRemovalError(
                f"Expected a 2D or 3D image array, got shape {arr.shape}."
            )

        channels = arr.shape[2]
        if channels == 1:
            return Image.fromarray(arr.squeeze(-1), mode="L")
        if channels == 3:
            return Image.fromarray(arr, mode="RGB")
        if channels == 4:
            return Image.fromarray(arr, mode="RGBA")
        raise BackgroundRemovalError(f"Unsupported channel count: {channels}.")

    def _ensure_rgba(self, result: Image.Image | np.ndarray | bytes) -> Image.Image:
        if isinstance(result, Image.Image):
            image = result
        elif isinstance(result, np.ndarray):
            image = self._numpy_to_pil(result)
        elif isinstance(result, (bytes, bytearray)):
            try:
                with Image.open(io.BytesIO(bytes(result))) as decoded:
                    decoded.load()
                    image = decoded.copy()
            except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
                raise BackgroundRemovalError(
                    "rembg returned an image that could not be decoded."
                ) from exc
        else:
            raise BackgroundRemovalError(
                f"Unexpected rembg output type: {type(result).__name__}."
            )

        rgba = image.convert("RGBA")
        return self._validate(rgba)

    @staticmethod
    def _validate(image: Image.Image) -> Image.Image:
        width, height = image.size
        if width <= 0 or height <= 0:
            raise BackgroundRemovalError(
                f"Image has invalid dimensions: {width}x{height}."
            )
        return image


_default_remover: BackgroundRemover | None = None


def remove_background(image_input: ImageInput) -> Image.Image:
    """Remove the background and return a 4-channel RGBA PIL Image."""
    global _default_remover
    if _default_remover is None:
        _default_remover = BackgroundRemover()
    return _default_remover.remove_background(image_input)
