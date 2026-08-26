"""2D image preprocessing for the DimensionX 3D spatial asset pipeline."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Union

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

ImageSource = Union[str, Path, bytes, bytearray, memoryview, BinaryIO]

__all__ = [
    "ImageLoadError",
    "ImagePreprocessor",
    "PreprocessedImage",
    "preprocess_image",
]


class ImageLoadError(Exception):
    """Raised when an image cannot be loaded or decoded."""


@dataclass(frozen=True)
class PreprocessedImage:
    """Normalized image in both NumPy (H, W, C) and PyTorch (C, H, W) formats."""

    array: np.ndarray
    tensor: torch.Tensor

    @property
    def height(self) -> int:
        return int(self.array.shape[0])

    @property
    def width(self) -> int:
        return int(self.array.shape[1])


class ImagePreprocessor:
    """Load, convert, resize, and normalize 2D images for downstream 3D inference."""

    DEFAULT_SIZE = 224

    def __init__(self, target_size: int | tuple[int, int] = DEFAULT_SIZE) -> None:
        self.target_size = self._parse_size(target_size)

    def process(self, source: ImageSource) -> PreprocessedImage:
        """Run the full preprocessing pipeline on a file path, bytes, or stream."""
        image = self.load(source)
        image = self.to_rgb(image)
        image = self.resize(image)
        array = self.normalize(image)
        tensor = self.to_tensor(array)
        return PreprocessedImage(array=array, tensor=tensor)

    def load(self, source: ImageSource) -> Image.Image:
        """Load an image from a filesystem path, raw bytes, or a binary stream."""
        if source is None:
            raise ImageLoadError("Image source is empty.")

        try:
            if isinstance(source, (bytes, bytearray, memoryview)):
                payload = bytes(source)
                if not payload:
                    raise ImageLoadError("Image byte payload is empty.")
                return self._validate(self._from_bytes(payload))

            if hasattr(source, "read"):
                payload = source.read()
                if isinstance(payload, str):
                    payload = payload.encode("utf-8")
                if not payload:
                    raise ImageLoadError("Image stream is empty.")
                return self._validate(self._from_bytes(bytes(payload)))

            path = Path(source)
            if not path.is_file():
                raise ImageLoadError(f"Image file not found: {path}")
            return self._validate(self._from_path(path))
        except ImageLoadError:
            raise
        except Exception as exc:
            raise ImageLoadError(f"Failed to load image: {exc}") from exc

    def to_rgb(self, image: Image.Image) -> Image.Image:
        """Convert RGBA, grayscale, and palette images to 3-channel RGB."""
        if image.mode == "RGB":
            return image

        has_alpha = image.mode in {"RGBA", "LA"} or (
            image.mode == "P" and "transparency" in image.info
        )
        if has_alpha:
            rgba = image.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.split()[-1])
            return background

        return image.convert("RGB")

    def resize(self, image: Image.Image) -> Image.Image:
        """Resize to the configured target size using high-quality resampling."""
        return image.resize(self.target_size, Image.Resampling.LANCZOS)

    def normalize(self, image: Image.Image) -> np.ndarray:
        """Return an HWC float32 array with pixel values in [0, 1]."""
        array = np.asarray(image, dtype=np.float32)
        if array.size == 0:
            raise ImageLoadError("Image contains no pixel data.")
        return np.clip(array / 255.0, 0.0, 1.0)

    def to_tensor(self, array: np.ndarray) -> torch.Tensor:
        """Convert an HWC NumPy array to a CHW float32 PyTorch tensor."""
        if array.ndim != 3 or array.shape[2] != 3:
            raise ImageLoadError(
                f"Expected an RGB array with shape (H, W, 3), got {array.shape}."
            )
        return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).contiguous()

    def _from_path(self, path: Path) -> Image.Image:
        try:
            with Image.open(path) as image:
                image.load()
                return ImageOps.exif_transpose(image.copy())
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
            bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if bgr is None:
                raise ImageLoadError(f"Invalid or corrupted image file: {path}") from None
            return self._opencv_to_pil(bgr)

    def _from_bytes(self, payload: bytes) -> Image.Image:
        try:
            with Image.open(io.BytesIO(payload)) as image:
                image.load()
                return ImageOps.exif_transpose(image.copy())
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
            buffer = np.frombuffer(payload, dtype=np.uint8)
            bgr = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
            if bgr is None:
                raise ImageLoadError("Invalid or corrupted image bytes.") from None
            return self._opencv_to_pil(bgr)

    @staticmethod
    def _opencv_to_pil(array: np.ndarray) -> Image.Image:
        if array.ndim == 2:
            return Image.fromarray(array, mode="L")
        channels = array.shape[2]
        if channels == 4:
            return Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGRA2RGBA), mode="RGBA")
        if channels == 3:
            return Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB), mode="RGB")
        raise ImageLoadError(f"Unsupported OpenCV channel count: {channels}.")

    @staticmethod
    def _validate(image: Image.Image) -> Image.Image:
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ImageLoadError(f"Image has invalid dimensions: {width}x{height}.")
        return image

    @staticmethod
    def _parse_size(target_size: int | tuple[int, int]) -> tuple[int, int]:
        if isinstance(target_size, int):
            if target_size <= 0:
                raise ValueError("target_size must be a positive integer.")
            return (target_size, target_size)
        if (
            isinstance(target_size, tuple)
            and len(target_size) == 2
            and all(isinstance(dim, int) and dim > 0 for dim in target_size)
        ):
            width, height = target_size
            return (width, height)
        raise ValueError("target_size must be a positive int or a (width, height) tuple.")


def preprocess_image(
    source: ImageSource,
    target_size: int | tuple[int, int] = ImagePreprocessor.DEFAULT_SIZE,
) -> PreprocessedImage:
    """Convenience wrapper around :class:`ImagePreprocessor`."""
    return ImagePreprocessor(target_size=target_size).process(source)
