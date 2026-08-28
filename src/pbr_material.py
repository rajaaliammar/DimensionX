"""Basic fabric PBR material and textured OBJ/MTL export."""

from __future__ import annotations

from pathlib import Path

from src.mesh_io import MeshData, write_obj

__all__ = [
    "FABRIC_METALLIC",
    "FABRIC_ROUGHNESS",
    "MATERIAL_NAME",
    "export_textured_mesh",
    "write_pbr_mtl",
]

MATERIAL_NAME = "ShirtFabric"
FABRIC_ROUGHNESS = 0.7
FABRIC_METALLIC = 0.0


def write_pbr_mtl(
    path: str | Path,
    texture_filename: str,
    roughness: float = FABRIC_ROUGHNESS,
    metallic: float = FABRIC_METALLIC,
) -> Path:
    """Write a Wavefront MTL with fabric PBR roughness/metallic."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ns = float(max((1.0 - roughness) ** 2 * 250.0, 1.0))
    specular = 0.04 * (1.0 - metallic)
    text = (
        "# DimensionX fabric PBR material\n"
        f"newmtl {MATERIAL_NAME}\n"
        "Ka 1.000000 1.000000 1.000000\n"
        "Kd 1.000000 1.000000 1.000000\n"
        f"Ks {specular:.6f} {specular:.6f} {specular:.6f}\n"
        "Ke 0.000000 0.000000 0.000000\n"
        f"Ns {ns:.4f}\n"
        "Ni 1.450000\n"
        "d 1.000000\n"
        "illum 2\n"
        f"map_Kd {texture_filename}\n"
        f"Pr {roughness:.4f}\n"
        f"Pm {metallic:.4f}\n"
    )
    path.write_text(text, encoding="utf-8")
    return path


def export_textured_mesh(
    mesh: MeshData,
    obj_path: str | Path,
    texture_filename: str = "shirt_texture.png",
    roughness: float = FABRIC_ROUGHNESS,
    metallic: float = FABRIC_METALLIC,
) -> tuple[Path, Path]:
    """Export OBJ + MTL referencing the UV shirt texture."""
    obj_path = Path(obj_path)
    mtl_path = obj_path.with_suffix(".mtl")
    write_pbr_mtl(mtl_path, texture_filename, roughness=roughness, metallic=metallic)
    write_obj(
        obj_path,
        mesh,
        mtllib=mtl_path.name,
        material=MATERIAL_NAME,
        header="DimensionX textured fabric mesh",
    )
    return obj_path, mtl_path
