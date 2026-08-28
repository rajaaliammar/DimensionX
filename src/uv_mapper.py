"""Dynamic UV mapping and edge-sampled shirt texture generation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from src.mesh_generator import MeshGenerationError
from src.mesh_io import MeshData, compute_vertex_normals, face_normals, load_obj_mesh, write_obj

__all__ = [
    "TEXTURE_PAD",
    "UVMappingError",
    "apply_dynamic_uv_mapping",
    "generate_shirt_texture",
    "map_mesh_from_files",
]

TEXTURE_PAD = 16
CAP_UV_OFFSET_PX = 10.0


class UVMappingError(MeshGenerationError):
    """Raised when UV coordinates or the shirt texture cannot be built."""


def map_mesh_from_files(
    mesh_path: str | Path,
    image_path: str | Path,
    pad: int = TEXTURE_PAD,
) -> tuple[MeshData, Image.Image, dict]:
    """Load a mesh + T-shirt cutout, assign UVs, and build an edge-sampled texture."""
    mesh = load_obj_mesh(mesh_path)
    try:
        with Image.open(image_path) as image:
            image.load()
            cutout = image.convert("RGBA").copy()
    except (OSError, ValueError) as exc:
        raise UVMappingError(f"Failed to load T-shirt image '{image_path}': {exc}") from exc

    mesh, stats = apply_dynamic_uv_mapping(mesh, cutout.size, pad=pad)
    texture = generate_shirt_texture(cutout, pad=pad)
    stats["texture_size"] = texture.size
    return mesh, texture, stats


def apply_dynamic_uv_mapping(
    mesh: MeshData,
    image_size: tuple[int, int],
    pad: int = TEXTURE_PAD,
) -> tuple[MeshData, dict]:
    """Project the front plane onto the T-shirt image; offset cap/side UVs into the edge band."""
    width, height = int(image_size[0]), int(image_size[1])
    if width <= 1 or height <= 1:
        raise UVMappingError("T-shirt image is too small to build UV coordinates.")

    vertices = mesh.vertices
    faces = mesh.faces
    is_cap = _cap_vertex_mask(vertices)
    planar = _planar_uvs(vertices, width, height)

    atlas_w = width + 2 * pad
    atlas_h = height + 2 * pad
    u_px = planar[:, 0] * (width - 1) + pad
    v_px = planar[:, 1] * (height - 1) + pad

    if np.any(is_cap):
        outward = _outward_xy(vertices, is_cap)
        u_px[is_cap] = u_px[is_cap] + outward[:, 0] * CAP_UV_OFFSET_PX
        v_px[is_cap] = v_px[is_cap] + outward[:, 1] * CAP_UV_OFFSET_PX

    uvs = np.stack(
        [
            np.clip(u_px / max(atlas_w - 1, 1), 0.0, 1.0),
            np.clip(v_px / max(atlas_h - 1, 1), 0.0, 1.0),
        ],
        axis=1,
    ).astype(np.float32)

    normals = mesh.normals
    if normals.size == 0 or len(normals) != len(vertices):
        normals = compute_vertex_normals(vertices, faces)

    mapped = MeshData(
        vertices=vertices,
        faces=faces,
        uvs=uvs,
        normals=normals,
        colors=mesh.colors,
    )
    fn = face_normals(vertices, faces)
    cap_faces = np.all(is_cap[faces], axis=1)
    front_faces = (~cap_faces) & (fn[:, 2] >= 0.35)
    side_faces = ~(front_faces | cap_faces)
    stats = {
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "front_vertices": int(np.count_nonzero(~is_cap)),
        "cap_vertices": int(np.count_nonzero(is_cap)),
        "front_faces": int(np.count_nonzero(front_faces)),
        "side_faces": int(np.count_nonzero(side_faces)),
        "back_faces": int(np.count_nonzero(cap_faces)),
        "uv_min": (float(uvs[:, 0].min()), float(uvs[:, 1].min())),
        "uv_max": (float(uvs[:, 0].max()), float(uvs[:, 1].max())),
        "pad": int(pad),
    }
    return mapped, stats


def generate_shirt_texture(cutout: Image.Image, pad: int = TEXTURE_PAD) -> Image.Image:
    """Fill transparent pixels with nearest garment-edge colors, then pad for side-wall UVs."""
    rgba = np.asarray(cutout.convert("RGBA"))
    filled = _fill_with_edge_colors(rgba)
    padded = cv2.copyMakeBorder(
        filled,
        pad,
        pad,
        pad,
        pad,
        borderType=cv2.BORDER_REPLICATE,
    )
    return Image.fromarray(padded, mode="RGB")


def _planar_uvs(vertices: np.ndarray, width: int, height: int) -> np.ndarray:
    """Map mesh XY back onto the source image, matching Phase 1 unprojection."""
    span = float(max(height, width) - 1)
    xs = vertices[:, 0] * span + (width - 1) * 0.5
    ys = (height - 1) * 0.5 - vertices[:, 1] * span
    u = xs / max(width - 1, 1)
    v = 1.0 - ys / max(height - 1, 1)
    return np.stack([u, v], axis=1).astype(np.float32)


def _cap_vertex_mask(vertices: np.ndarray) -> np.ndarray:
    z = vertices[:, 2]
    z_min = float(z.min())
    z_max = float(z.max())
    thickness = max(z_max - z_min, 1e-6)
    return z <= (z_min + 0.08 * thickness)


def _outward_xy(vertices: np.ndarray, is_cap: np.ndarray) -> np.ndarray:
    cap_xy = vertices[is_cap, :2]
    center = vertices[~is_cap, :2].mean(axis=0) if np.any(~is_cap) else cap_xy.mean(axis=0)
    delta = cap_xy - center
    length = np.linalg.norm(delta, axis=1, keepdims=True)
    return (delta / np.clip(length, 1e-8, None)).astype(np.float32)


def _fill_with_edge_colors(rgba: np.ndarray) -> np.ndarray:
    rgb = np.array(rgba[..., :3], copy=True)
    alpha = rgba[..., 3] if rgba.shape[2] == 4 else np.full(rgb.shape[:2], 255, dtype=np.uint8)
    mask = alpha >= 128
    if not np.any(mask):
        return rgb
    core = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
    if not np.any(core):
        core = mask
    missing = ~core
    if not np.any(missing):
        return rgb
    _dist, indices = distance_transform_edt(missing, return_indices=True)
    iy, ix = indices
    filled = rgb.copy()
    filled[missing] = rgb[iy[missing], ix[missing]]
    return filled


def save_uv_mesh(mesh: MeshData, path: str | Path) -> Path:
    return write_obj(path, mesh, header="DimensionX UV-mapped mesh")
