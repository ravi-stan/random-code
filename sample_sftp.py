from pathlib import Path
from typing import Iterable, Union
from PIL import Image, TiffImagePlugin
import warnings
import math

def _bytes_per_pixel(mode: str) -> float:
    """Return the raw bytes per pixel for common Pillow modes."""
    # 8-bit channels unless noted
    table = {
        "1": 1 / 8,     # 1-bit bilevel
        "L": 1,         # 8-bit greyscale
        "P": 1,         # 8-bit palette
        "RGB": 3,
        "RGBA": 4,
        "CMYK": 4,
        "I;16": 2,      # 16-bit unsigned integer
        "I": 4,         # 32-bit signed integer
        "F": 4,         # 32-bit float
    }
    if mode not in table:
        raise ValueError(f"Unsupported mode {mode!r} for size estimation")
    return table[mode]

def combine_tiffs_preview_safe(
    input_files: Iterable[Union[str, Path]],
    output_path: Union[str, Path],
    *,
    mode: str = "RGB",
    dpi: tuple[int, int] = (300, 300),
    size_limit_gb: int = 4,
) -> Path:
    """
    Merge single-page TIFFs into a multi-page baseline-strip TIFF that
    macOS Preview can open (classic header, LZW strips).
    """
    paths = [Path(p) for p in input_files]
    if not paths:
        raise ValueError("No input files provided.")

    pages = []
    w = h = None
    for p in paths:
        with Image.open(p) as im:
            if im.mode != mode:
                im = im.convert(mode)
            im.load()               # fully detach from source file
            if w is None:
                w, h = im.size
            elif im.size != (w, h):
                raise ValueError(
                    f"Page {p.name} is {im.size}, "
                    f"first page is {(w, h)} – Preview needs uniform size."
                )
            im.info["dpi"] = dpi
            pages.append(im.copy())

    # ---- 4 GB guard ---------------------------------------------------
    bpp   = _bytes_per_pixel(mode)
    est   = math.ceil(w * h * bpp * len(pages))        # raw byte estimate
    limit = size_limit_gb * 1024 ** 3
    compress = "tiff_lzw"
    if est >= limit:
        warnings.warn(
            "Estimated uncompressed size exceeds classic TIFF limit. "
            "Falling back to uncompressed strips so we can still write "
            "a Preview-readable classic TIFF (file will be large)."
        )
        compress = "raw"  # Pillow keyword for “none”

    # ---- write with Pillow’s baseline-strip encoder -------------------
    TiffImagePlugin.WRITE_LIBTIFF = False  # force baseline writer
    pages[0].save(
        output_path,
        save_all=True,
        append_images=pages[1:],
        compression=compress,
        dpi=dpi,
    )

    for im in pages:
        im.close()
    return Path(output_path)
