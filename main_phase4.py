"""DimensionX Phase 4: standalone GLB export and asset validation."""

from pathlib import Path

from src.glb_exporter import GlbExportError, export_textured_glb
from src.pbr_material import FABRIC_METALLIC, FABRIC_ROUGHNESS

OUTPUT_DIR = Path("data/output")
TEXTURED_OBJ = OUTPUT_DIR / "textured_mesh.obj"
TEXTURED_MTL = OUTPUT_DIR / "textured_mesh.mtl"
TEXTURE_PATH = OUTPUT_DIR / "shirt_texture.png"
GLB_PATH = OUTPUT_DIR / "final_garment.glb"


def main() -> None:
    missing = [
        str(path)
        for path in (TEXTURED_OBJ, TEXTURED_MTL, TEXTURE_PATH)
        if not path.is_file()
    ]
    if missing:
        print(
            "Missing Phase 3 outputs: "
            + ", ".join(missing)
            + ". Run main_phase3.py first to generate textured_mesh.obj, "
            ".mtl, and shirt_texture.png."
        )
        return

    print("=== Phase 4: Multi-Format Export & Asset Optimization ===")
    print(f"Source OBJ:    {TEXTURED_OBJ}")
    print(f"Source MTL:    {TEXTURED_MTL}")
    print(f"Source texture:{TEXTURE_PATH}")
    try:
        report = export_textured_glb(
            TEXTURED_OBJ,
            GLB_PATH,
            texture_path=TEXTURE_PATH,
            roughness=FABRIC_ROUGHNESS,
            metallic=FABRIC_METALLIC,
        )
    except GlbExportError as exc:
        print(f"Failed to export GLB: {exc}")
        return
    except OSError as exc:
        print(f"Failed to write GLB: {exc}")
        return

    print("--- GLB verification ---")
    print(f"Output GLB:           {report['glb_path']}")
    print(f"Vertices:             {report['vertex_count']}")
    print(f"Faces:                {report['face_count']}")
    print(f"File size:            {report['file_size_mb']:.3f} MB ({report['file_size_bytes']} bytes)")
    print(f"GLB version:          {report['gltf_version']}")
    print(f"Packed normals:       {report['has_normal_attr']}")
    print(f"Packed UVs:           {report['has_uv_attr']}")
    print(f"Embedded texture:     {report['embedded_texture']}  images={report['image_count']}")
    print(f"Base color texture:   {report['base_color_texture']}")
    print(f"Embedded BIN buffer:  {report['embedded_buffer']}  ({report['bin_chunk_bytes']} bytes)")
    print(f"Material:             {report['material']}")
    print(f"Roughness:            {report.get('packed_roughness', FABRIC_ROUGHNESS)}")
    print(f"Metallic:             {report.get('packed_metallic', FABRIC_METALLIC)}")
    print(f"Texture size:         {report['texture_size'][0]}x{report['texture_size'][1]}")
    print("Phase 4 GLB export completed successfully.")


if __name__ == "__main__":
    main()
