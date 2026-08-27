"""DimensionX Phase 1 entry point."""

from pathlib import Path

from PIL import Image

from src.bg_remover import BackgroundRemovalError, remove_background
from src.depth_estimator import DepthEstimationError, estimate_depth_and_normals
from src.mesh_generator import MeshGenerationError, create_point_cloud_and_mesh
from src.preprocessor import ImageLoadError, preprocess_image

INPUT_DIR = Path("data/input")
OUTPUT_DIR = Path("data/output")
OUTPUT_IMAGE = OUTPUT_DIR / "no_bg_sample.png"
DEPTH_IMAGE = OUTPUT_DIR / "depth_map.png"
NORMAL_IMAGE = OUTPUT_DIR / "normal_map.png"
MESH_PATH = OUTPUT_DIR / "initial_mesh.obj"
DEFAULT_IMAGE = INPUT_DIR / "example.png"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def resolve_sample_image() -> Path | None:
    if DEFAULT_IMAGE.is_file():
        return DEFAULT_IMAGE
    if not INPUT_DIR.is_dir():
        return None
    for path in sorted(INPUT_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            return path
    return None


def main() -> None:
    sample = resolve_sample_image()
    if sample is None:
        print(
            "No test image found. Place a sample file at "
            "data/input/example.png (or any image in data/input/) and run again."
        )
        return

    try:
        with Image.open(sample) as source:
            source.load()
            print(f"Input path:   {sample}")
            print(f"Input mode:   {source.mode}")
            print(f"Input size:   {source.size[0]}x{source.size[1]}")

        result = preprocess_image(sample)
        print(f"Tensor shape: {tuple(result.tensor.shape)}")
        print(f"NumPy shape:  {result.array.shape}")
        print(
            f"Value range:  min={float(result.array.min()):.6f}  "
            f"max={float(result.array.max()):.6f}"
        )

        cutout = remove_background(result.array)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cutout.save(OUTPUT_IMAGE)
        print(f"Saved RGBA cutout: {OUTPUT_IMAGE.resolve()}")
        print(f"Output mode:   {cutout.mode}")
        print(f"Output size:   {cutout.size[0]}x{cutout.size[1]}")
        print(f"Output bands:  {cutout.getbands()}")

        depth_map, normal_map = estimate_depth_and_normals(cutout)
        depth_map.save(DEPTH_IMAGE)
        normal_map.save(NORMAL_IMAGE)

        mesh = create_point_cloud_and_mesh(cutout.convert("RGBA"), depth_map.convert("L"))
        exported = mesh.export_mesh(MESH_PATH)
    except ImageLoadError as exc:
        print(f"Failed to load test image '{sample}': {exc}")
        return
    except BackgroundRemovalError as exc:
        print(f"Failed to remove background from '{sample}': {exc}")
        return
    except DepthEstimationError as exc:
        print(f"Failed to estimate depth/normals from '{sample}': {exc}")
        return
    except MeshGenerationError as exc:
        print(f"Failed to generate extruded mesh from '{sample}': {exc}")
        return
    except OSError as exc:
        print(f"Failed to process test image '{sample}': {exc}")
        return

    print(
        f"Depth map:    size={depth_map.size[0]}x{depth_map.size[1]}  "
        f"mode={depth_map.mode}  shape=({depth_map.size[1]}, {depth_map.size[0]})"
    )
    print(
        f"Normal map:   size={normal_map.size[0]}x{normal_map.size[1]}  "
        f"mode={normal_map.mode}  shape=({normal_map.size[1]}, {normal_map.size[0]}, 3)"
    )
    print(f"Saved depth map:  {DEPTH_IMAGE.resolve()}")
    print(f"Saved normal map: {NORMAL_IMAGE.resolve()}")
    print("Phase 1 preprocess, background removal, and depth estimation completed successfully.")
    print_mesh_summary(mesh, exported)


def print_mesh_summary(mesh, exported: Path) -> None:
    print("--- Mesh output summary ---")
    print(f"Saved mesh:            {Path(exported).resolve()}")
    print(f"Method:                single-mesh extrusion (front + side walls + flat back cap)")
    print(f"Extrude depth:         {mesh.extrude_depth:.3f} (clamped to 0.15-0.20)")
    print(f"Vertices:              {mesh.vertex_count}")
    print(f"Faces:                 {mesh.face_count}")
    print(f"Seam edges:            {mesh.boundary_edge_count}")
    print(f"is_watertight:         {mesh.is_watertight}")
    print(f"connected components:  {mesh.component_count}")
    try:
        import trimesh

        loaded = trimesh.load(exported, force="mesh", process=False)
        bodies = loaded.split(only_watertight=False)
        print(f"trimesh watertight:    {bool(loaded.is_watertight)}")
        print(f"trimesh components:    {len(bodies)}")
        print(f"trimesh winding:       {bool(getattr(loaded, 'is_winding_consistent', False))}")
    except ImportError:
        print("trimesh not installed; using built-in watertight/component checks only.")
    except Exception as exc:
        print(f"trimesh check skipped: {exc}")
    print("Phase 1 single-mesh extrusion completed successfully.")


if __name__ == "__main__":
    main()
