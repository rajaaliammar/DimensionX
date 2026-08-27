"""Lightweight depth and surface-normal estimation for DimensionX Phase 1."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

__all__ = [
    "DepthEstimationError",
    "DepthEstimator",
    "estimate_depth_and_normals",
]


class DepthEstimationError(Exception):
    """Raised when depth or normal maps cannot be estimated."""


@dataclass(frozen=True)
class DepthNormalResult:
    depth: Image.Image
    normals: Image.Image
    method: str


class DepthEstimator:
    """Estimate a relative depth map and RGB surface normals from a 2D cutout."""

    def estimate_depth_and_normals(self, pil_image: Image.Image) -> tuple[Image.Image, Image.Image]:
        """Return a grayscale depth map and an RGB normal map for an RGB/RGBA image."""
        result = self.estimate(pil_image)
        return result.depth, result.normals

    def estimate(self, pil_image: Image.Image) -> DepthNormalResult:
        image = self._validate(pil_image)
        rgb, mask = self._split_rgb_mask(image)

        try:
            depth = self._estimate_depth_torch(rgb, mask)
            method = "torchvision"
        except Exception:
            depth = self._estimate_depth_cv2(rgb, mask)
            method = "cv2"

        try:
            normals = self._normals_from_depth(depth, mask)
        except Exception as exc:
            raise DepthEstimationError(f"Failed to compute surface normals: {exc}") from exc

        depth_image = Image.fromarray(self._to_uint8(depth), mode="L")
        normal_image = Image.fromarray(normals, mode="RGB")
        return DepthNormalResult(depth=depth_image, normals=normal_image, method=method)

    def _estimate_depth_torch(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Relative depth from silhouette thickness and shading via torchvision/torch."""
        tensor = TF.to_tensor(Image.fromarray(rgb, mode="RGB"))
        gray = TF.rgb_to_grayscale(tensor).squeeze(0)

        shading = gray.clamp(0.0, 1.0)
        if int(mask.max()) > 0:
            fg = torch.from_numpy(mask.astype(np.float32))
            values = shading[fg > 0.5]
            if values.numel() > 0:
                vmin = values.min()
                vmax = values.max()
                shading = (shading - vmin) / (vmax - vmin + 1e-6)
            shading = shading * fg

        dist = self._silhouette_distance(mask)
        depth = 0.65 * dist + 0.35 * shading.cpu().numpy().astype(np.float32)
        depth = cv2.GaussianBlur(depth, (0, 0), sigmaX=1.2)
        depth[mask == 0] = 0.0
        return self._normalize_foreground(depth, mask)

    def _estimate_depth_cv2(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """CPU fallback using OpenCV distance transform and luminance shading."""
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        dist = self._silhouette_distance(mask)
        shading = gray.copy()
        if int(mask.max()) > 0:
            fg = shading[mask > 0]
            shading = (shading - fg.min()) / (fg.max() - fg.min() + 1e-6)
        shading[mask == 0] = 0.0
        depth = 0.65 * dist + 0.35 * shading
        depth = cv2.GaussianBlur(depth, (0, 0), sigmaX=1.2)
        depth[mask == 0] = 0.0
        return self._normalize_foreground(depth, mask)

    @staticmethod
    def _silhouette_distance(mask: np.ndarray) -> np.ndarray:
        binary = (mask > 0).astype(np.uint8) * 255
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        peak = float(dist.max())
        if peak <= 0.0:
            return np.zeros_like(dist, dtype=np.float32)
        return (dist / peak).astype(np.float32)

    @staticmethod
    def _normals_from_depth(depth: np.ndarray, mask: np.ndarray, strength: float = 2.5) -> np.ndarray:
        dx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
        normals = np.dstack((-dx * strength, -dy * strength, np.ones_like(depth, dtype=np.float32)))
        length = np.linalg.norm(normals, axis=2, keepdims=True)
        normals = normals / (length + 1e-8)
        rgb = ((normals + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
        rgb[mask == 0] = (128, 128, 255)
        return rgb

    @staticmethod
    def _split_rgb_mask(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        if image.mode == "RGBA":
            rgba = np.asarray(image)
            alpha = rgba[..., 3].astype(np.float32) / 255.0
            rgb = rgba[..., :3].astype(np.float32)
            rgb = rgb * alpha[..., None] + 128.0 * (1.0 - alpha[..., None])
            mask = (rgba[..., 3] > 8).astype(np.uint8)
            return rgb.clip(0, 255).astype(np.uint8), mask

        if image.mode == "LA":
            la = np.asarray(image)
            rgb = np.repeat(la[..., :1], 3, axis=2)
            mask = (la[..., 1] > 8).astype(np.uint8)
            return rgb, mask

        rgb = np.asarray(image.convert("RGB"))
        mask = np.ones(rgb.shape[:2], dtype=np.uint8)
        return rgb, mask

    @staticmethod
    def _normalize_foreground(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = np.zeros_like(depth, dtype=np.float32)
        if int(mask.max()) == 0:
            return out
        fg = depth[mask > 0]
        out[mask > 0] = (fg - fg.min()) / (fg.max() - fg.min() + 1e-6)
        return out

    @staticmethod
    def _to_uint8(depth: np.ndarray) -> np.ndarray:
        return np.clip(depth * 255.0, 0, 255).astype(np.uint8)

    @staticmethod
    def _validate(pil_image: Image.Image) -> Image.Image:
        if pil_image is None:
            raise DepthEstimationError("Image input is empty.")
        if not isinstance(pil_image, Image.Image):
            raise DepthEstimationError(
                f"Expected a PIL Image, got {type(pil_image).__name__}."
            )
        width, height = pil_image.size
        if width <= 0 or height <= 0:
            raise DepthEstimationError(f"Image has invalid dimensions: {width}x{height}.")
        return pil_image


_default_estimator: DepthEstimator | None = None


def estimate_depth_and_normals(pil_image: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Estimate grayscale depth and RGB surface normals from an RGB/RGBA PIL image."""
    global _default_estimator
    if _default_estimator is None:
        _default_estimator = DepthEstimator()
    try:
        return _default_estimator.estimate_depth_and_normals(pil_image)
    except DepthEstimationError:
        raise
    except Exception as exc:
        raise DepthEstimationError(f"Depth estimation failed: {exc}") from exc
