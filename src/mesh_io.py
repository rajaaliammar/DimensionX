"""Minimal OBJ load/save used by UV mapping and PBR export."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.mesh_generator import MeshGenerationError


@dataclass
class MeshData:
    vertices: np.ndarray
    faces: np.ndarray
    uvs: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    normals: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    colors: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.uint8))


def load_obj_mesh(path: str | Path) -> MeshData:
    path = Path(path)
    if not path.is_file():
        raise MeshGenerationError(f"Mesh file not found: {path}")

    vertices: list[list[float]] = []
    colors: list[list[int]] = []
    uvs: list[list[float]] = []
    normals: list[list[float]] = []
    faces: list[list[int]] = []
    face_uvs: list[list[int]] = []
    face_nrms: list[list[int]] = []

    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("v "):
                    parts = line.split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    if len(parts) >= 7:
                        rgb = [float(parts[4]), float(parts[5]), float(parts[6])]
                        if max(rgb) <= 1.5:
                            rgb = [c * 255.0 for c in rgb]
                        colors.append([int(np.clip(c, 0, 255)) for c in rgb])
                elif line.startswith("vt "):
                    parts = line.split()
                    uvs.append([float(parts[1]), float(parts[2])])
                elif line.startswith("vn "):
                    parts = line.split()
                    normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith("f "):
                    corners = line.split()[1:]
                    vidx, tidx, nidx = [], [], []
                    for corner in corners:
                        bits = corner.split("/")
                        vidx.append(int(bits[0]) - 1)
                        tidx.append(int(bits[1]) - 1 if len(bits) > 1 and bits[1] else -1)
                        nidx.append(int(bits[2]) - 1 if len(bits) > 2 and bits[2] else -1)
                    if len(vidx) < 3:
                        continue
                    for i in range(1, len(vidx) - 1):
                        faces.append([vidx[0], vidx[i], vidx[i + 1]])
                        face_uvs.append([tidx[0], tidx[i], tidx[i + 1]])
                        face_nrms.append([nidx[0], nidx[i], nidx[i + 1]])
    except (OSError, ValueError) as exc:
        raise MeshGenerationError(f"Failed to read mesh '{path}': {exc}") from exc

    if not vertices or not faces:
        raise MeshGenerationError(f"Mesh '{path}' has no vertices or faces.")

    verts = np.asarray(vertices, dtype=np.float32)
    faces_arr = np.asarray(faces, dtype=np.int32)
    uv_arr = np.zeros((len(verts), 2), dtype=np.float32)
    if uvs and face_uvs:
        vt = np.asarray(uvs, dtype=np.float32)
        for tri, uv_tri in zip(faces_arr, face_uvs):
            for vi, ti in zip(tri, uv_tri):
                if 0 <= ti < len(vt):
                    uv_arr[vi] = vt[ti]
    nrm_arr = np.zeros((len(verts), 3), dtype=np.float32)
    if normals and face_nrms:
        vn = np.asarray(normals, dtype=np.float32)
        for tri, n_tri in zip(faces_arr, face_nrms):
            for vi, ni in zip(tri, n_tri):
                if 0 <= ni < len(vn):
                    nrm_arr[vi] = vn[ni]
    elif normals and len(normals) == len(verts):
        nrm_arr = np.asarray(normals, dtype=np.float32)
    col_arr = (
        np.asarray(colors, dtype=np.uint8)
        if len(colors) == len(verts)
        else np.full((len(verts), 3), 200, dtype=np.uint8)
    )
    return MeshData(vertices=verts, faces=faces_arr, uvs=uv_arr, normals=nrm_arr, colors=col_arr)


def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    length = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals = face_normals / np.clip(length, 1e-8, None)
    vertex_normals = np.zeros_like(vertices, dtype=np.float32)
    np.add.at(vertex_normals, faces[:, 0], face_normals)
    np.add.at(vertex_normals, faces[:, 1], face_normals)
    np.add.at(vertex_normals, faces[:, 2], face_normals)
    vlen = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    return vertex_normals / np.clip(vlen, 1e-8, None)


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.clip(length, 1e-8, None)


def write_obj(
    path: str | Path,
    mesh: MeshData,
    mtllib: str | None = None,
    material: str | None = None,
    header: str = "DimensionX mesh",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normals = mesh.normals
    if normals.size == 0 or len(normals) != len(mesh.vertices):
        normals = compute_vertex_normals(mesh.vertices, mesh.faces)
    uvs = mesh.uvs
    if uvs.size == 0 or len(uvs) != len(mesh.vertices):
        uvs = np.zeros((len(mesh.vertices), 2), dtype=np.float32)

    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {header}\n")
        if mtllib:
            handle.write(f"mtllib {mtllib}\n")
        handle.write("o ShirtMesh\n")
        for x, y, z in mesh.vertices:
            handle.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for u, v in uvs:
            handle.write(f"vt {float(u):.6f} {float(v):.6f}\n")
        for nx, ny, nz in normals:
            handle.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")
        if material:
            handle.write(f"usemtl {material}\n")
        handle.write("s off\n")
        for a, b, c in mesh.faces + 1:
            handle.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")
    return path
