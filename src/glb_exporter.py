"""Convert a textured OBJ into a standalone binary GLB with embedded PBR."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image

from src.mesh_io import MeshData, compute_vertex_normals, load_obj_mesh
from src.pbr_material import FABRIC_METALLIC, FABRIC_ROUGHNESS, MATERIAL_NAME

__all__ = [
    "GlbExportError",
    "export_textured_glb",
    "inspect_glb",
]


class GlbExportError(Exception):
    """Raised when a GLB cannot be built, packed, or validated."""


def export_textured_glb(
    obj_path: str | Path,
    glb_path: str | Path,
    texture_path: str | Path | None = None,
    roughness: float = FABRIC_ROUGHNESS,
    metallic: float = FABRIC_METALLIC,
) -> dict:
    """Load textured OBJ + PNG, pack a binary GLB, and return a validation report."""
    obj_path = Path(obj_path)
    glb_path = Path(glb_path)
    if not obj_path.is_file():
        raise GlbExportError(f"Textured mesh not found: {obj_path}")

    texture_path = Path(texture_path) if texture_path else _resolve_texture(obj_path)
    if texture_path is None or not texture_path.is_file():
        raise GlbExportError(
            "Shirt texture not found. Expected shirt_texture.png next to the OBJ "
            "or a map_Kd entry in the MTL file."
        )

    mesh = load_obj_mesh(obj_path)
    mesh = _optimize_mesh(mesh)
    try:
        texture = Image.open(texture_path)
        texture.load()
        texture = texture.convert("RGB")
    except (OSError, ValueError) as exc:
        raise GlbExportError(f"Failed to load texture '{texture_path}': {exc}") from exc

    _export_with_trimesh(mesh, texture, glb_path, roughness=roughness, metallic=metallic)
    report = inspect_glb(glb_path)
    report.update(
        {
            "source_obj": str(obj_path),
            "source_texture": str(texture_path),
            "vertex_count": int(len(mesh.vertices)),
            "face_count": int(len(mesh.faces)),
            "has_normals": bool(len(mesh.normals) == len(mesh.vertices)),
            "has_uvs": bool(len(mesh.uvs) == len(mesh.vertices)),
            "roughness": float(roughness),
            "metallic": float(metallic),
            "material": MATERIAL_NAME,
            "texture_size": texture.size,
        }
    )
    if not report.get("embedded_texture"):
        raise GlbExportError(
            "GLB was written but no embedded texture image was found in the buffer."
        )
    return report


def inspect_glb(path: str | Path) -> dict:
    """Read GLB chunks and confirm packed normals, UVs, and embedded images."""
    path = Path(path)
    if not path.is_file():
        raise GlbExportError(f"GLB file not found: {path}")
    data = path.read_bytes()
    if len(data) < 12 or data[:4] != b"glTF":
        raise GlbExportError(f"File is not a GLB container: {path}")
    version, length = struct.unpack_from("<II", data, 4)
    if version != 2:
        raise GlbExportError(f"Unsupported GLB version {version}.")
    if length != len(data):
        raise GlbExportError("GLB header length does not match the file size.")

    json_doc, bin_len = _parse_glb_chunks(data)
    meshes = json_doc.get("meshes", [])
    attrs = {}
    if meshes:
        attrs = meshes[0].get("primitives", [{}])[0].get("attributes", {})
    materials = json_doc.get("materials", [])
    pbr = materials[0].get("pbrMetallicRoughness", {}) if materials else {}
    images = json_doc.get("images", [])
    buffers = json_doc.get("buffers", [])
    embedded_image = False
    for image in images:
        if "bufferView" in image:
            embedded_image = True
        elif not image.get("uri"):
            embedded_image = True
    buffer_embedded = bool(buffers) and all(not buf.get("uri") for buf in buffers)
    size_bytes = path.stat().st_size
    return {
        "glb_path": str(path.resolve()),
        "file_size_bytes": int(size_bytes),
        "file_size_mb": float(size_bytes) / (1024.0 * 1024.0),
        "gltf_version": version,
        "embedded_texture": bool(embedded_image),
        "embedded_buffer": bool(buffer_embedded),
        "image_count": int(len(images)),
        "bin_chunk_bytes": int(bin_len),
        "has_normal_attr": "NORMAL" in attrs,
        "has_uv_attr": "TEXCOORD_0" in attrs,
        "packed_roughness": pbr.get("roughnessFactor"),
        "packed_metallic": pbr.get("metallicFactor"),
        "base_color_texture": "baseColorTexture" in pbr,
    }


def _optimize_mesh(mesh: MeshData) -> MeshData:
    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.int32)
    uvs = mesh.uvs
    if uvs.size == 0 or len(uvs) != len(vertices):
        raise GlbExportError("Mesh is missing UV coordinates; run Phase 2/3 first.")
    uvs = np.ascontiguousarray(np.clip(uvs, 0.0, 1.0), dtype=np.float32)
    normals = mesh.normals
    if normals.size == 0 or len(normals) != len(vertices) or not np.any(np.abs(normals) > 1e-8):
        normals = compute_vertex_normals(vertices, faces)
    else:
        length = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.clip(length, 1e-8, None)
    normals = np.ascontiguousarray(normals, dtype=np.float32)
    return MeshData(vertices=vertices, faces=faces, uvs=uvs, normals=normals, colors=mesh.colors)


def _export_with_trimesh(
    mesh: MeshData,
    texture: Image.Image,
    glb_path: Path,
    roughness: float,
    metallic: float,
) -> None:
    try:
        import trimesh
        from trimesh.visual.material import PBRMaterial
        from trimesh.visual.texture import TextureVisuals
    except ImportError as exc:
        raise GlbExportError(
            "trimesh is required for GLB export. Install it with: pip install trimesh"
        ) from exc

    material = PBRMaterial(
        name=MATERIAL_NAME,
        baseColorFactor=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        baseColorTexture=texture,
        metallicFactor=float(metallic),
        roughnessFactor=float(roughness),
        doubleSided=False,
    )
    visual = TextureVisuals(uv=mesh.uvs, material=material)
    tri = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        vertex_normals=mesh.normals,
        visual=visual,
        process=False,
    )
    glb_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        tri.export(glb_path, file_type="glb")
    except Exception as exc:
        raise GlbExportError(f"Failed to export GLB '{glb_path}': {exc}") from exc
    if not glb_path.is_file() or glb_path.stat().st_size < 64:
        raise GlbExportError(f"GLB export produced an empty file: {glb_path}")


def _resolve_texture(obj_path: Path) -> Path | None:
    mtl_path = obj_path.with_suffix(".mtl")
    if mtl_path.is_file():
        for raw in mtl_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line.lower().startswith("map_kd "):
                name = line.split(None, 1)[1].strip().strip('"')
                candidate = (obj_path.parent / name).resolve()
                if candidate.is_file():
                    return candidate
    fallback = obj_path.parent / "shirt_texture.png"
    return fallback if fallback.is_file() else None


def _parse_glb_chunks(data: bytes) -> tuple[dict, int]:
    offset = 12
    json_doc = {}
    bin_len = 0
    while offset + 8 <= len(data):
        chunk_len, chunk_type = struct.unpack_from("<I4s", data, offset)
        offset += 8
        chunk = data[offset : offset + chunk_len]
        offset += chunk_len
        if chunk_type == b"JSON":
            json_doc = json.loads(chunk.decode("utf-8").rstrip("\x20"))
        elif chunk_type == b"BIN\x00":
            bin_len = chunk_len
    if not json_doc:
        raise GlbExportError("GLB is missing a JSON chunk.")
    return json_doc, bin_len
