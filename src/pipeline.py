"""End-to-end DimensionX pipeline: image → textured GLB with status callbacks."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from src.bg_remover import remove_background
from src.depth_estimator import estimate_depth_and_normals
from src.glb_exporter import export_textured_glb
from src.mesh_generator import create_point_cloud_and_mesh
from src.pbr_material import FABRIC_METALLIC, FABRIC_ROUGHNESS, export_textured_mesh
from src.preprocessor import preprocess_image
from src.uv_mapper import map_mesh_from_files

StatusCallback = Callable[[str], None]

OUTPUT_DIR = Path("data/output")
INPUT_DIR = Path("data/input")
CUTOUT_PATH = OUTPUT_DIR / "no_bg_sample.png"
DEPTH_PATH = OUTPUT_DIR / "depth_map.png"
NORMAL_PATH = OUTPUT_DIR / "normal_map.png"
MESH_PATH = OUTPUT_DIR / "initial_mesh.obj"
TEXTURE_PATH = OUTPUT_DIR / "shirt_texture.png"
TEXTURED_OBJ = OUTPUT_DIR / "textured_mesh.obj"
GLB_PATH = OUTPUT_DIR / "final_garment.glb"


def run_full_pipeline(
    image_path: str | Path,
    on_status: StatusCallback | None = None,
) -> dict:
    """Run Phase 1–4 on a garment image and return paths plus mesh stats."""

    def status(message: str) -> None:
        if on_status is not None:
            on_status(message)

    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    status("Preprocessing image...")
    prepared = preprocess_image(image_path)

    status("Removing background...")
    cutout = remove_background(prepared.array)
    cutout.save(CUTOUT_PATH)

    status("Estimating depth...")
    depth_map, normal_map = estimate_depth_and_normals(cutout)
    depth_map.save(DEPTH_PATH)
    normal_map.save(NORMAL_PATH)

    status("Extruding Mesh...")
    mesh = create_point_cloud_and_mesh(cutout.convert("RGBA"), depth_map.convert("L"))
    mesh.export_mesh(MESH_PATH)

    status("UV Mapping...")
    mapped, texture, uv_stats = map_mesh_from_files(MESH_PATH, CUTOUT_PATH)
    texture.save(TEXTURE_PATH)

    status("Applying PBR material...")
    export_textured_mesh(
        mapped,
        TEXTURED_OBJ,
        texture_filename=TEXTURE_PATH.name,
        roughness=FABRIC_ROUGHNESS,
        metallic=FABRIC_METALLIC,
    )

    status("Packing GLB...")
    report = export_textured_glb(
        TEXTURED_OBJ,
        GLB_PATH,
        texture_path=TEXTURE_PATH,
        roughness=FABRIC_ROUGHNESS,
        metallic=FABRIC_METALLIC,
    )

    status("Ready")
    return {
        "glb_path": str(GLB_PATH.as_posix()),
        "glb_url": f"/data/output/{GLB_PATH.name}",
        "texture_url": f"/data/output/{TEXTURE_PATH.name}",
        "vertex_count": report.get("vertex_count", mesh.vertex_count),
        "face_count": report.get("face_count", mesh.face_count),
        "file_size_mb": report.get("file_size_mb"),
        "uv_stats": uv_stats,
        "is_watertight": mesh.is_watertight,
    }
