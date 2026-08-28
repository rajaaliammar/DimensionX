"""DimensionX Phase 3: fabric PBR material and textured OBJ export."""

from pathlib import Path

from src.mesh_generator import MeshGenerationError
from src.pbr_material import (
    FABRIC_METALLIC,
    FABRIC_ROUGHNESS,
    MATERIAL_NAME,
    export_textured_mesh,
)
from src.uv_mapper import UVMappingError, map_mesh_from_files

OUTPUT_DIR = Path("data/output")
CUTOUT_IMAGE = OUTPUT_DIR / "no_bg_sample.png"
SOURCE_MESH = OUTPUT_DIR / "initial_mesh.obj"
TEXTURE_PATH = OUTPUT_DIR / "shirt_texture.png"
TEXTURED_OBJ = OUTPUT_DIR / "textured_mesh.obj"


def main() -> None:
    missing = [str(path) for path in (SOURCE_MESH, CUTOUT_IMAGE) if not path.is_file()]
    if missing:
        print(
            "Missing Phase 1 outputs: "
            + ", ".join(missing)
            + ". Run main_phase1.py first to generate initial_mesh.obj and no_bg_sample.png."
        )
        return

    print("=== Phase 3: PBR Material Setup ===")
    print(f"Source mesh:   {SOURCE_MESH}")
    print(f"T-shirt image: {CUTOUT_IMAGE}")
    try:
        mesh, texture, stats = map_mesh_from_files(SOURCE_MESH, CUTOUT_IMAGE)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        texture.save(TEXTURE_PATH)
        obj_path, mtl_path = export_textured_mesh(
            mesh,
            TEXTURED_OBJ,
            texture_filename=TEXTURE_PATH.name,
            roughness=FABRIC_ROUGHNESS,
            metallic=FABRIC_METALLIC,
        )
    except (UVMappingError, MeshGenerationError) as exc:
        print(f"Failed to export textured mesh: {exc}")
        return
    except OSError as exc:
        print(f"Failed to write PBR outputs: {exc}")
        return

    print(f"Material:      {MATERIAL_NAME}")
    print(f"Roughness:     {FABRIC_ROUGHNESS:.2f}")
    print(f"Metallic:      {FABRIC_METALLIC:.2f}")
    print(f"Vertices:      {stats['vertex_count']}")
    print(f"Faces:         {stats['face_count']}")
    print(f"Saved texture: {TEXTURE_PATH.resolve()}  size={texture.size[0]}x{texture.size[1]}")
    print(f"Saved OBJ:     {obj_path.resolve()}")
    print(f"Saved MTL:     {mtl_path.resolve()}")

    try:
        import trimesh

        loaded = trimesh.load(obj_path, force="mesh", process=False)
        print(f"trimesh faces: {int(len(getattr(loaded, 'faces', [])))}")
        print(f"trimesh has UV: {bool(getattr(getattr(loaded, 'visual', None), 'uv', None) is not None)}")
        print(f"trimesh watertight: {bool(getattr(loaded, 'is_watertight', False))}")
    except ImportError:
        print("trimesh not installed; skipped textured-mesh verification.")
    except Exception as exc:
        print(f"trimesh check skipped: {exc}")
    print("Phase 3 PBR material setup completed successfully.")


if __name__ == "__main__":
    main()
