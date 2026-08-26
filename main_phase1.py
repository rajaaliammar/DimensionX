"""DimensionX Phase 1 entry point."""

from pathlib import Path

from PIL import Image

from src.bg_remover import BackgroundRemovalError, remove_background
from src.preprocessor import ImageLoadError, preprocess_image

INPUT_DIR = Path("data/input")
OUTPUT_DIR = Path("data/output")
OUTPUT_IMAGE = OUTPUT_DIR / "no_bg_sample.png"
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
    except ImageLoadError as exc:
        print(f"Failed to load test image '{sample}': {exc}")
        return
    except BackgroundRemovalError as exc:
        print(f"Failed to remove background from '{sample}': {exc}")
        return
    except OSError as exc:
        print(f"Failed to process test image '{sample}': {exc}")
        return

    print(f"Saved RGBA cutout: {OUTPUT_IMAGE.resolve()}")
    print(f"Output mode:   {cutout.mode}")
    print(f"Output size:   {cutout.size[0]}x{cutout.size[1]}")
    print(f"Output bands:  {cutout.getbands()}")
    print("Phase 1 preprocess + background removal completed successfully.")


if __name__ == "__main__":
    main()
