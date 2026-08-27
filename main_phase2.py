"""DimensionX Phase 2 entry point: point cloud and initial mesh reconstruction."""

from pathlib import Path

from PIL import Image

from src.mesh_generator import MeshGenerationError, create_point_cloud_and_mesh

OUTPUT_DIR = Path("data/output")
CUTOUT_IMAGE = OUTPUT_DIR / "no_bg_sample.png"
DEPTH_IMAGE = OUTPUT_DIR / "depth_map.png"
MESH_PATH = OUTPUT_DIR / "initial_mesh.obj"


def main() -> None:
    missing = [str(path) for path in (CUTOUT_IMAGE, DEPTH_IMAGE) if not path.is_file()]
    if missing:
        print(
            "Missing Phase 1 outputs: "
            + ", ".join(missing)
            + ". Run main_phase1.py first to generate no_bg_sample.png and depth_map.png."
        )
        return

    try:
        with Image.open(CUTOUT_IMAGE) as cutout, Image.open(DEPTH_IMAGE) as depth:
            cutout.load()
            depth.load()
            print(f"Cutout path:  {CUTOUT_IMAGE}")
            print(f"Cutout size:  {cutout.size[0]}x{cutout.size[1]}  mode={cutout.mode}")
            print(f"Depth path:   {DEPTH_IMAGE}")
            print(f"Depth size:   {depth.size[0]}x{depth.size[1]}  mode={depth.mode}")
            rgba = cutout.convert("RGBA").copy()
            depth_map = depth.convert("L").copy()

        mesh = create_point_cloud_and_mesh(rgba, depth_map)
        print(f"Z-scale:      {mesh.z_scale:.2f} (clamped to 0.10-0.25)")
        print(f"Depth blur:   Gaussian sigma={mesh.blur_sigma}")
        print(f"Alpha clip:   drop vertices with alpha < {mesh.min_alpha}")
        print(f"Extrude:      single-mesh downward extrusion depth={mesh.extrude_depth:.3f}")
        print(f"Back cap:     flat mirrored silhouette polygon")
        print(f"Seam edges:   {mesh.boundary_edge_count}")
        print(f"is_watertight: {mesh.is_watertight}")
        print(f"Components:   {mesh.component_count}")
        print(f"Point cloud:  {mesh.point_count} points")
        print(f"Vertices:     {mesh.vertex_count}")
        print(f"Faces:        {mesh.face_count}")

        exported = mesh.export_mesh(MESH_PATH)
    except MeshGenerationError as exc:
        print(f"Failed to reconstruct 3D mesh: {exc}")
        return
    except OSError as exc:
        print(f"Failed to load Phase 1 outputs: {exc}")
        return

    print(f"Saved mesh:   {Path(exported).resolve()}")
    try:
        import trimesh

        loaded = trimesh.load(exported, force="mesh", process=False)
        print(f"trimesh watertight: {bool(loaded.is_watertight)}")
        print(f"trimesh components: {len(loaded.split(only_watertight=False))}")
        print(f"trimesh winding consistent: {bool(getattr(loaded, 'is_winding_consistent', False))}")
    except ImportError:
        print("trimesh not installed; using built-in watertight check only.")
    except Exception as exc:
        print(f"trimesh check skipped: {exc}")
    print("Phase 2 closed volumetric mesh reconstruction completed successfully.")


if __name__ == "__main__":
    main()
