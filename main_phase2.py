"""DimensionX Phase 2: dynamic UV mapping on the Phase 1 extruded mesh."""

from pathlib import Path

from src.mesh_generator import MeshGenerationError
from src.uv_mapper import UVMappingError, map_mesh_from_files, save_uv_mesh

OUTPUT_DIR = Path("data/output")
CUTOUT_IMAGE = OUTPUT_DIR / "no_bg_sample.png"
SOURCE_MESH = OUTPUT_DIR / "initial_mesh.obj"
UV_MESH = OUTPUT_DIR / "uv_mapped_mesh.obj"
TEXTURE_PATH = OUTPUT_DIR / "shirt_texture.png"


def main() -> None:
    missing = [str(path) for path in (SOURCE_MESH, CUTOUT_IMAGE) if not path.is_file()]
    if missing:
        print(
            "Missing Phase 1 outputs: "
            + ", ".join(missing)
            + ". Run main_phase1.py first to generate initial_mesh.obj and no_bg_sample.png."
        )
        return

    print("=== Phase 2: Dynamic UV Mapping ===")
    print(f"Source mesh:   {SOURCE_MESH}")
    print(f"T-shirt image: {CUTOUT_IMAGE}")
    try:
        mesh, texture, stats = map_mesh_from_files(SOURCE_MESH, CUTOUT_IMAGE)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        texture.save(TEXTURE_PATH)
        exported = save_uv_mesh(mesh, UV_MESH)
    except (UVMappingError, MeshGenerationError) as exc:
        print(f"Failed to apply UV mapping: {exc}")
        return
    except OSError as exc:
        print(f"Failed to write UV outputs: {exc}")
        return

    print(f"Vertices:      {stats['vertex_count']}")
    print(f"Faces:         {stats['face_count']}")
    print(f"Front verts:   {stats['front_vertices']}  (planar map onto T-shirt image)")
    print(f"Cap verts:     {stats['cap_vertices']}  (outward UV offset into edge-color band)")
    print(f"Front faces:   {stats['front_faces']}")
    print(f"Side faces:    {stats['side_faces']}")
    print(f"Back faces:    {stats['back_faces']}")
    print(
        f"UV range:      u=[{stats['uv_min'][0]:.3f}, {stats['uv_max'][0]:.3f}]  "
        f"v=[{stats['uv_min'][1]:.3f}, {stats['uv_max'][1]:.3f}]"
    )
    print(f"Texture pad:   {stats['pad']}px edge-color dilation")
    print(f"Saved texture: {TEXTURE_PATH.resolve()}  size={texture.size[0]}x{texture.size[1]}")
    print(f"Saved UV mesh: {Path(exported).resolve()}")
    print("Phase 2 UV mapping completed successfully.")


if __name__ == "__main__":
    main()
