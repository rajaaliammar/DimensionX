"""Temporary interactive viewer for the Phase 2 initial mesh."""

from __future__ import annotations

from pathlib import Path

import numpy as np

MESH_PATH = Path("data/output/initial_mesh.obj")


def main() -> None:
    if not MESH_PATH.is_file():
        print(
            f"Mesh not found: {MESH_PATH}. "
            "Run main_phase2.py first to generate initial_mesh.obj."
        )
        return

    vertices, faces, colors = load_obj(MESH_PATH)
    print_stats(vertices, faces)

    if show_with_trimesh(MESH_PATH):
        return
    if show_with_open3d(MESH_PATH):
        return
    if show_with_matplotlib(vertices, faces, colors):
        return

    print(
        "No 3D viewer backend found. Install one of: "
        'pip install trimesh  |  pip install open3d  |  pip install matplotlib'
    )


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    vertices: list[list[float]] = []
    colors: list[list[float]] = []
    faces: list[list[int]] = []
    has_color = True

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if len(parts) >= 7:
                    colors.append([float(parts[4]), float(parts[5]), float(parts[6])])
                else:
                    has_color = False
            elif line.startswith("f "):
                corners = []
                for token in line.split()[1:]:
                    corners.append(int(token.split("/")[0]) - 1)
                if len(corners) >= 3:
                    for i in range(1, len(corners) - 1):
                        faces.append([corners[0], corners[i], corners[i + 1]])

    if not vertices or not faces:
        raise RuntimeError(f"No vertices/faces found in {path}")

    color_array = np.asarray(colors, dtype=np.float32) if has_color and colors else None
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32), color_array


def print_stats(vertices: np.ndarray, faces: np.ndarray) -> None:
    bounds = np.stack([vertices.min(axis=0), vertices.max(axis=0)])
    print(f"Mesh path:     {MESH_PATH.resolve()}")
    print(f"Vertex count:  {len(vertices)}")
    print(f"Face count:    {len(faces)}")
    print(f"is_watertight: {is_watertight(faces)}")
    print(f"bounds min:    {bounds[0].tolist()}")
    print(f"bounds max:    {bounds[1].tolist()}")


def is_watertight(faces: np.ndarray) -> bool:
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]],
        axis=0,
    )
    edges.sort(axis=1)
    _unique, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(counts.size > 0 and np.all(counts == 2))


def show_with_trimesh(path: Path) -> bool:
    try:
        import trimesh
    except ImportError:
        return False

    mesh = trimesh.load(path, force="mesh", process=False)
    print(f"Viewer:        trimesh ({type(mesh).__name__})")
    if hasattr(mesh, "is_watertight"):
        print(f"trimesh watertight: {mesh.is_watertight}")
    print("Opening interactive window (rotate / pan / zoom). Close the window to exit.")
    mesh.show()
    return True


def show_with_open3d(path: Path) -> bool:
    try:
        import open3d as o3d
    except ImportError:
        return False

    mesh = o3d.io.read_triangle_mesh(str(path))
    mesh.compute_vertex_normals()
    print("Viewer:        open3d")
    print("Opening interactive window (rotate / pan / zoom). Close the window to exit.")
    o3d.visualization.draw_geometries(
        [mesh],
        window_name="DimensionX initial_mesh.obj",
        mesh_show_back_face=True,
    )
    return True


def show_with_matplotlib(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray | None,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    print("Viewer:        matplotlib 3D")
    print("Controls:      left-drag rotate, right-drag pan, scroll zoom. Close the window to exit.")

    fig = plt.figure("DimensionX initial_mesh.obj", figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf_kwargs = {
        "triangles": faces,
        "linewidth": 0.0,
        "antialiased": False,
        "shade": True,
    }
    if colors is not None and len(colors) == len(vertices):
        ax.plot_trisurf(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            **surf_kwargs,
            cmap="viridis",
        )
    else:
        ax.plot_trisurf(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            **surf_kwargs,
            color="#8ecae6",
        )
    _set_axes_equal(ax)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("initial_mesh.obj")
    plt.tight_layout()
    plt.show()
    return True


def _set_axes_equal(ax) -> None:
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()], dtype=np.float64)
    centers = limits.mean(axis=1)
    radius = 0.5 * float((limits[:, 1] - limits[:, 0]).max())
    ax.set_xlim3d(centers[0] - radius, centers[0] + radius)
    ax.set_ylim3d(centers[1] - radius, centers[1] + radius)
    ax.set_zlim3d(centers[2] - radius, centers[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


if __name__ == "__main__":
    main()
