"""Single-mesh extrusion from an RGBA cutout and depth map."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from scipy.ndimage import binary_fill_holes

ImageInput = Union[str, Path, Image.Image, np.ndarray]

__all__ = [
    "MeshGenerationError",
    "MeshGenerator",
    "create_point_cloud_and_mesh",
]


class MeshGenerationError(Exception):
    """Raised when a point cloud or mesh cannot be generated or exported."""


class MeshGenerator:
    """Build one front surface, extrude the silhouette, and cap a closed solid."""

    def __init__(
        self,
        z_scale: float = 0.18,
        blur_sigma: float = 1.6,
        min_alpha: float = 0.5,
        min_depth: float = 1.0 / 255.0,
        extrude_depth: float = 0.18,
    ) -> None:
        self.z_scale = float(np.clip(z_scale, 0.1, 0.25))
        self.blur_sigma = blur_sigma
        self.min_alpha = min_alpha
        self.min_depth = min_depth
        self.extrude_depth = float(np.clip(extrude_depth, 0.15, 0.20))
        self.points = np.zeros((0, 3), dtype=np.float32)
        self.colors = np.zeros((0, 3), dtype=np.uint8)
        self.uvs = np.zeros((0, 2), dtype=np.float32)
        self.faces = np.zeros((0, 3), dtype=np.int32)
        self.normals = np.zeros((0, 3), dtype=np.float32)
        self.face_normals = np.zeros((0, 3), dtype=np.float32)
        self.boundary_edge_count = 0

    @property
    def vertex_count(self) -> int:
        return int(self.points.shape[0])

    @property
    def face_count(self) -> int:
        return int(self.faces.shape[0])

    @property
    def point_count(self) -> int:
        return self.vertex_count

    @property
    def is_watertight(self) -> bool:
        if self.face_count == 0:
            return False
        edges = np.sort(
            np.concatenate(
                [self.faces[:, [0, 1]], self.faces[:, [1, 2]], self.faces[:, [2, 0]]],
                axis=0,
            ),
            axis=1,
        )
        _unique, counts = np.unique(edges, axis=0, return_counts=True)
        return bool(counts.size > 0 and np.all(counts == 2))

    @property
    def component_count(self) -> int:
        return self._connected_component_count()

    def create_point_cloud_and_mesh(
        self,
        image_rgba: ImageInput,
        depth_map: ImageInput,
    ) -> "MeshGenerator":
        """Unproject one front mesh, then extrude it into a single closed solid."""
        try:
            color = self._as_rgba(image_rgba)
            depth = self._as_depth(depth_map)
            if depth.size != color.size:
                depth = depth.resize(color.size, Image.Resampling.BILINEAR)

            rgba = np.asarray(color)
            depth_01 = np.asarray(depth, dtype=np.float32) / 255.0
            if depth_01.ndim != 2:
                raise MeshGenerationError(
                    f"Depth map must be grayscale, got shape {depth_01.shape}."
                )

            alpha = rgba[..., 3].astype(np.float32) / 255.0
            mask = alpha >= self.min_alpha
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel).astype(bool)
            mask = binary_fill_holes(mask)
            mask = self._largest_component(mask)
            mask = self._remove_junction_pixels(mask)
            depth_01 = self._prepare_depth(depth_01, mask)
            mask = mask & (depth_01 > self.min_depth)
            mask = binary_fill_holes(mask)
            mask = self._largest_component(mask)
            if not np.any(mask):
                raise MeshGenerationError(
                    "No valid foreground pixels found in the RGBA/depth pair."
                )

            self._height, self._width = depth_01.shape
            self._rgba = rgba
            self._mask = mask

            self.points, self.colors, self.uvs = self._unproject(rgba, depth_01, mask)
            self.faces = self._triangulate(mask)
            if self.faces.size == 0:
                raise MeshGenerationError(
                    "Could not form a front surface from the projected point cloud."
                )
            self._compact_unused_vertices()
            self._extrude_solid()
            self._merge_close_vertices(eps=1e-4)
            self.face_normals, self.normals = self._compute_normals(self.points, self.faces)
        except MeshGenerationError:
            raise
        except Exception as exc:
            raise MeshGenerationError(f"Mesh generation failed: {exc}") from exc
        return self

    def export_mesh(self, file_path: str | Path) -> Path:
        """Save the reconstructed mesh as `.obj` or `.ply`."""
        if self.vertex_count == 0 or self.face_count == 0:
            raise MeshGenerationError("No mesh to export. Run create_point_cloud_and_mesh first.")

        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix not in {".obj", ".ply"}:
            raise MeshGenerationError(
                f"Unsupported mesh format '{path.suffix}'. Use .obj or .ply."
            )

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            exported = self._export_with_trimesh(path)
            if not exported:
                if suffix == ".obj":
                    self._write_obj(path)
                else:
                    self._write_ply(path)
        except MeshGenerationError:
            raise
        except OSError as exc:
            raise MeshGenerationError(f"Failed to write mesh '{path}': {exc}") from exc
        return path

    def _prepare_depth(self, depth_01: np.ndarray, mask: np.ndarray) -> np.ndarray:
        masked = np.where(mask, depth_01, 0.0).astype(np.float32)
        sigma = float(self.blur_sigma)
        blurred = cv2.GaussianBlur(masked, (0, 0), sigmaX=sigma, sigmaY=sigma)
        weight = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
        prepared = np.divide(blurred, np.maximum(weight, 1e-6), dtype=np.float32)
        prepared = np.where(mask, prepared, 0.0)
        foreground = prepared[mask]
        lo = float(foreground.min())
        hi = float(foreground.max())
        if hi > lo:
            prepared[mask] = (foreground - lo) / (hi - lo)
        return np.clip(prepared, 0.0, 1.0)

    def _unproject(
        self,
        rgba: np.ndarray,
        depth_01: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        height, width = depth_01.shape
        ys, xs = np.nonzero(mask)
        span = float(max(height, width) - 1)
        x = (xs.astype(np.float32) - (width - 1) * 0.5) / span
        y = ((height - 1) * 0.5 - ys.astype(np.float32)) / span
        z = depth_01[ys, xs] * self.z_scale
        points = np.stack([x, y, z], axis=1).astype(np.float32)
        colors = rgba[ys, xs, :3].astype(np.uint8)
        uvs = np.stack(
            [
                xs.astype(np.float32) / max(width - 1, 1),
                1.0 - ys.astype(np.float32) / max(height - 1, 1),
            ],
            axis=1,
        ).astype(np.float32)
        return points, colors, uvs

    def _triangulate(self, mask: np.ndarray) -> np.ndarray:
        """Triangulate every foreground quad so the front stays a single disk."""
        index = np.full(mask.shape, -1, dtype=np.int32)
        index[mask] = np.arange(int(mask.sum()), dtype=np.int32)
        quad = mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, :-1] & mask[1:, 1:]
        if not np.any(quad):
            return np.zeros((0, 3), dtype=np.int32)
        i00 = index[:-1, :-1][quad]
        i10 = index[:-1, 1:][quad]
        i01 = index[1:, :-1][quad]
        i11 = index[1:, 1:][quad]
        tri1 = np.stack([i00, i01, i10], axis=1)
        tri2 = np.stack([i10, i01, i11], axis=1)
        return np.concatenate([tri1, tri2], axis=0).astype(np.int32)

    @staticmethod
    def _largest_component(mask: np.ndarray) -> np.ndarray:
        count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        if count <= 2:
            return mask
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        return labels == int(np.argmax(sizes))

    def _compact_unused_vertices(self) -> None:
        used = np.unique(self.faces.ravel())
        if used.size == self.vertex_count:
            return
        remap = np.full(self.vertex_count, -1, dtype=np.int32)
        remap[used] = np.arange(used.size, dtype=np.int32)
        self.points = self.points[used]
        self.colors = self.colors[used]
        self.uvs = self.uvs[used]
        self.faces = remap[self.faces]

    def _extrude_solid(self) -> None:
        """Extrude the front silhouette down in Z and cap with a flat back polygon."""
        boundary = self._oriented_boundary_edges(self.faces)
        loops = self._ordered_perimeter_loops()
        if boundary.size == 0 or not loops:
            raise MeshGenerationError("Could not extract a silhouette loop to extrude.")

        outer = self._dedupe_closed_loop(loops[0])
        if len(outer) < 3:
            raise MeshGenerationError("Silhouette loop is too small to extrude.")

        front_count = self.vertex_count
        z_cap = float(np.min(self.points[:, 2])) - self.extrude_depth

        rim = []
        seen: set[int] = set()
        for loop in loops:
            for vert in self._dedupe_closed_loop(loop):
                if vert not in seen:
                    seen.add(vert)
                    rim.append(vert)
        for a, b in np.atleast_2d(boundary):
            for vert in (int(a), int(b)):
                if vert not in seen:
                    seen.add(vert)
                    rim.append(vert)

        rim = np.asarray(rim, dtype=np.int32)
        front_to_cap = np.full(front_count, -1, dtype=np.int32)
        front_to_cap[rim] = np.arange(rim.size, dtype=np.int32) + front_count

        cap_points = self.points[rim].copy()
        cap_points[:, 2] = z_cap
        self.points = np.concatenate([self.points, cap_points], axis=0)
        self.colors = np.concatenate([self.colors, self.colors[rim]], axis=0)
        self.uvs = np.concatenate([self.uvs, self.uvs[rim]], axis=0)

        walls = self._side_wall_triangles(boundary, front_to_cap)
        cap_faces = self._flat_back_cap(outer, front_to_cap)
        if walls.size == 0 or cap_faces.size == 0:
            raise MeshGenerationError("Failed to extrude side walls or cap the back polygon.")

        self.faces = np.concatenate([self.faces, walls, cap_faces], axis=0).astype(np.int32)
        self.boundary_edge_count = int(len(boundary))
        self._compact_unused_vertices()

    def _side_wall_triangles(self, boundary: np.ndarray, front_to_cap: np.ndarray) -> np.ndarray:
        """Connect each front boundary edge to its extruded cap edge."""
        walls: list[list[int]] = []
        covered: set[tuple[int, int]] = set()
        for a, b in np.atleast_2d(boundary):
            v1, v2 = int(b), int(a)
            if v1 == v2 or front_to_cap[v1] < 0 or front_to_cap[v2] < 0:
                continue
            key = (min(v1, v2), max(v1, v2))
            if key in covered:
                continue
            covered.add(key)
            c1 = int(front_to_cap[v1])
            c2 = int(front_to_cap[v2])
            walls.append([v1, v2, c2])
            walls.append([v1, c2, c1])
        if not walls:
            return np.zeros((0, 3), dtype=np.int32)
        return np.asarray(walls, dtype=np.int32)

    def _flat_back_cap(self, outer: list[int], front_to_cap: np.ndarray) -> np.ndarray:
        """Triangulate the silhouette as one flat, mirrored back polygon."""
        cap_loop = [int(front_to_cap[v]) for v in outer if front_to_cap[v] >= 0]
        cap_loop = self._dedupe_closed_loop(cap_loop)
        if len(cap_loop) < 3:
            raise MeshGenerationError("Back cap loop is empty.")
        xy = self.points[np.asarray(cap_loop, dtype=np.int32), :2]
        local = self._earclip_polygon(xy)
        if local.size == 0:
            local = self._fan_polygon(len(cap_loop))
        cap_idx = np.asarray(cap_loop, dtype=np.int32)
        faces = cap_idx[local]
        # Front faces point +Z; reverse the cap so it points -Z (outward).
        area = self._signed_area(xy)
        if area > 0.0:
            faces = faces[:, ::-1]
        return faces.astype(np.int32)

    def _ordered_perimeter_loops(self) -> list[list[int]]:
        loops = [self._dedupe_closed_loop(loop) for loop in self._walk_boundary_loops()]
        loops = [loop for loop in loops if len(loop) >= 3]
        if not loops:
            return []

        def loop_area(loop: list[int]) -> float:
            xy = self.points[np.asarray(loop, dtype=np.int32), :2]
            return abs(self._signed_area(xy))

        loops.sort(key=loop_area, reverse=True)
        return loops

    def _walk_boundary_loops(self) -> list[list[int]]:
        edges = self._oriented_boundary_edges(self.faces)
        if edges.size == 0:
            return []
        outgoing: dict[int, list[int]] = {}
        for a, b in edges:
            outgoing.setdefault(int(a), []).append(int(b))
        used: set[tuple[int, int]] = set()
        loops: list[list[int]] = []
        for start_a, start_b in edges:
            key = (int(start_a), int(start_b))
            if key in used:
                continue
            loop = [int(start_a)]
            curr, nxt = int(start_a), int(start_b)
            while (curr, nxt) not in used:
                used.add((curr, nxt))
                loop.append(nxt)
                found = None
                for candidate in outgoing.get(nxt, []):
                    if (nxt, candidate) not in used:
                        found = candidate
                        break
                if found is None:
                    break
                curr, nxt = nxt, found
                if nxt == loop[0]:
                    loop.append(nxt)
                    break
            if len(loop) >= 4:
                loops.append(loop[:-1] if loop[0] == loop[-1] else loop)
        return loops

    @staticmethod
    def _oriented_boundary_edges(faces: np.ndarray) -> np.ndarray:
        oriented = np.concatenate(
            [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]],
            axis=0,
        )
        keys = np.sort(oriented, axis=1)
        _unique, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        boundary = oriented[counts[inverse] == 1]
        if boundary.size == 0:
            return np.zeros((0, 2), dtype=np.int32)
        boundary_keys = np.sort(boundary, axis=1)
        _uniq, first = np.unique(boundary_keys, axis=0, return_index=True)
        return boundary[np.sort(first)].astype(np.int32)

    @staticmethod
    def _dedupe_closed_loop(loop: list[int]) -> list[int]:
        cleaned: list[int] = []
        for vert in loop:
            vert = int(vert)
            if not cleaned or cleaned[-1] != vert:
                cleaned.append(vert)
        if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
            cleaned = cleaned[:-1]
        return cleaned

    def _merge_close_vertices(self, eps: float = 1e-4) -> None:
        """Weld coincident vertices, keeping a watertight solid if one already exists."""
        if self.vertex_count == 0 or self.face_count == 0:
            return
        was_tight = self.is_watertight
        backup = (
            self.points.copy(),
            self.colors.copy(),
            self.uvs.copy(),
            self.faces.copy(),
        )
        quantized = np.round(self.points / max(eps, 1e-8), decimals=0)
        _unique, inverse = np.unique(quantized, axis=0, return_inverse=True)
        if int(inverse.max()) + 1 == self.vertex_count:
            return
        count = int(inverse.max() + 1)
        new_points = np.zeros((count, 3), dtype=np.float32)
        new_colors = np.zeros((count, 3), dtype=np.float32)
        new_uvs = np.zeros((count, 2), dtype=np.float32)
        weights = np.zeros((count, 1), dtype=np.float32)
        np.add.at(new_points, inverse, self.points)
        np.add.at(new_colors, inverse, self.colors.astype(np.float32))
        np.add.at(new_uvs, inverse, self.uvs)
        np.add.at(weights, inverse, 1.0)
        weights = np.maximum(weights, 1.0)
        self.points = new_points / weights
        self.colors = np.clip(new_colors / weights, 0, 255).astype(np.uint8)
        self.uvs = (new_uvs / weights).astype(np.float32)
        self.faces = inverse[self.faces]
        unique_ok = (
            (self.faces[:, 0] != self.faces[:, 1])
            & (self.faces[:, 1] != self.faces[:, 2])
            & (self.faces[:, 2] != self.faces[:, 0])
        )
        self.faces = self.faces[unique_ok]
        self._compact_unused_vertices()
        if was_tight and not self.is_watertight:
            self.points, self.colors, self.uvs, self.faces = backup

    def _connected_component_count(self) -> int:
        if self.face_count == 0 or self.vertex_count == 0:
            return 0
        parent = np.arange(self.vertex_count, dtype=np.int32)

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = int(parent[i])
            return i

        for a, b, c in self.faces:
            for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
                ru, rv = find(u), find(v)
                if ru != rv:
                    parent[rv] = ru
        used = np.unique(self.faces.ravel())
        return len({find(int(i)) for i in used.tolist()})

    @staticmethod
    def _signed_area(xy: np.ndarray) -> float:
        x, y = xy[:, 0], xy[:, 1]
        return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5

    @staticmethod
    def _fan_polygon(count: int) -> np.ndarray:
        if count < 3:
            return np.zeros((0, 3), dtype=np.int32)
        i = np.arange(1, count - 1, dtype=np.int32)
        return np.stack([np.zeros_like(i), i, i + 1], axis=1)

    @classmethod
    def _earclip_polygon(cls, xy: np.ndarray) -> np.ndarray:
        count = int(len(xy))
        if count < 3:
            return np.zeros((0, 3), dtype=np.int32)
        ccw = cls._signed_area(xy) > 0.0
        verts = list(range(count))
        faces: list[list[int]] = []
        guard = 0
        max_steps = count * count + 8
        while len(verts) > 3 and guard < max_steps:
            guard += 1
            clipped = False
            n = len(verts)
            for i in range(n):
                prev_i = verts[(i - 1) % n]
                curr_i = verts[i]
                next_i = verts[(i + 1) % n]
                a, b, c = xy[prev_i], xy[curr_i], xy[next_i]
                cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
                is_convex = cross > 1e-12 if ccw else cross < -1e-12
                if not is_convex:
                    continue
                if cls._triangle_contains_any(xy, verts, prev_i, curr_i, next_i):
                    continue
                faces.append([prev_i, curr_i, next_i])
                del verts[i]
                clipped = True
                break
            if not clipped:
                break
        if len(verts) == 3:
            faces.append(verts)
        if len(faces) != count - 2:
            return np.zeros((0, 3), dtype=np.int32)
        return np.asarray(faces, dtype=np.int32)

    @staticmethod
    def _triangle_contains_any(
        xy: np.ndarray,
        verts: list[int],
        i0: int,
        i1: int,
        i2: int,
    ) -> bool:
        a, b, c = xy[i0], xy[i1], xy[i2]
        area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(area) < 1e-16:
            return True
        skip = {i0, i1, i2}
        for idx in verts:
            if idx in skip:
                continue
            p = xy[idx]
            w0 = (b[0] - p[0]) * (c[1] - p[1]) - (b[1] - p[1]) * (c[0] - p[0])
            w1 = (c[0] - p[0]) * (a[1] - p[1]) - (c[1] - p[1]) * (a[0] - p[0])
            w2 = (a[0] - p[0]) * (b[1] - p[1]) - (a[1] - p[1]) * (b[0] - p[0])
            if area > 0:
                inside = w0 >= -1e-12 and w1 >= -1e-12 and w2 >= -1e-12
            else:
                inside = w0 <= 1e-12 and w1 <= 1e-12 and w2 <= 1e-12
            if inside:
                return True
        return False

    @staticmethod
    def _remove_junction_pixels(mask: np.ndarray) -> np.ndarray:
        offsets = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))
        padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
        height, width = mask.shape
        ring = np.stack(
            [padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width] for dy, dx in offsets],
            axis=0,
        )
        transitions = np.sum(ring != np.roll(ring, -1, axis=0), axis=0)
        cleaned = mask.copy()
        cleaned[mask & (transitions >= 4)] = False
        return cleaned

    @staticmethod
    def _compute_normals(
        vertices: np.ndarray, faces: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0)
        face_length = np.linalg.norm(face_normals, axis=1, keepdims=True)
        face_normals = face_normals / np.clip(face_length, 1e-8, None)
        vertex_normals = np.zeros_like(vertices, dtype=np.float32)
        np.add.at(vertex_normals, faces[:, 0], face_normals)
        np.add.at(vertex_normals, faces[:, 1], face_normals)
        np.add.at(vertex_normals, faces[:, 2], face_normals)
        vertex_length = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        vertex_normals = vertex_normals / np.clip(vertex_length, 1e-8, None)
        return face_normals.astype(np.float32), vertex_normals

    def _export_with_trimesh(self, path: Path) -> bool:
        try:
            import trimesh
        except ImportError:
            return False
        mesh = trimesh.Trimesh(
            vertices=self.points,
            faces=self.faces,
            vertex_colors=self.colors,
            vertex_normals=self.normals,
            process=False,
        )
        mesh.export(path)
        return True

    def _write_obj(self, path: Path) -> None:
        colors = self.colors.astype(np.float32) / 255.0
        with path.open("w", encoding="utf-8") as handle:
            handle.write("# DimensionX extruded solid mesh\n")
            for (x, y, z), (r, g, b) in zip(self.points, colors):
                handle.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f}\n")
            for u, v in self.uvs:
                handle.write(f"vt {u:.6f} {v:.6f}\n")
            for nx, ny, nz in self.normals:
                handle.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")
            for a, b, c in self.faces + 1:
                handle.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")

    def _write_ply(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as handle:
            handle.write("ply\nformat ascii 1.0\n")
            handle.write(f"element vertex {self.vertex_count}\n")
            handle.write("property float x\nproperty float y\nproperty float z\n")
            handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            handle.write(f"element face {self.face_count}\n")
            handle.write("property list uchar int vertex_indices\nend_header\n")
            for (x, y, z), (r, g, b) in zip(self.points, self.colors):
                handle.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
            for a, b, c in self.faces:
                handle.write(f"3 {a} {b} {c}\n")

    def _as_rgba(self, image_input: ImageInput) -> Image.Image:
        image = self._as_image(image_input, "RGBA image")
        return image.convert("RGBA")

    def _as_depth(self, image_input: ImageInput) -> Image.Image:
        image = self._as_image(image_input, "depth map")
        return image.convert("L")

    def _as_image(self, image_input: ImageInput, label: str) -> Image.Image:
        if image_input is None:
            raise MeshGenerationError(f"{label} is empty.")
        if isinstance(image_input, Image.Image):
            return self._validate_image(image_input, label)
        if isinstance(image_input, np.ndarray):
            array = image_input
            if array.size == 0:
                raise MeshGenerationError(f"{label} array is empty.")
            if array.ndim == 2:
                return self._validate_image(Image.fromarray(array), label)
            if array.ndim == 3 and array.shape[2] in {3, 4}:
                mode = "RGB" if array.shape[2] == 3 else "RGBA"
                if array.dtype != np.uint8:
                    if np.issubdtype(array.dtype, np.floating) and float(np.nanmax(array)) <= 1.5:
                        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
                    else:
                        array = np.clip(array, 0, 255).astype(np.uint8)
                return self._validate_image(Image.fromarray(array, mode=mode), label)
            raise MeshGenerationError(f"Unsupported {label} array shape {array.shape}.")
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.is_file():
                raise MeshGenerationError(f"{label} file not found: {path}")
            try:
                with Image.open(path) as image:
                    image.load()
                    return self._validate_image(image.copy(), label)
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                raise MeshGenerationError(f"Invalid or corrupted {label} file: {path}") from exc
        raise MeshGenerationError(
            f"Unsupported {label} type: {type(image_input).__name__}."
        )

    @staticmethod
    def _validate_image(image: Image.Image, label: str) -> Image.Image:
        width, height = image.size
        if width <= 0 or height <= 0:
            raise MeshGenerationError(f"{label} has invalid dimensions: {width}x{height}.")
        return image


def create_point_cloud_and_mesh(
    image_rgba: ImageInput,
    depth_map: ImageInput,
) -> MeshGenerator:
    """Build one extruded watertight mesh from RGBA + depth."""
    return MeshGenerator().create_point_cloud_and_mesh(image_rgba, depth_map)
