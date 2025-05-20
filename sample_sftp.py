from pathlib import Path
from typing import Iterable, Union
from PIL import Image, TiffImagePlugin
import warnings
import math

def _bytes_per_pixel(mode: str) -> float:
    table = {
        "1": 1 / 8, "L": 1, "P": 1,
        "RGB": 3, "RGBA": 4, "CMYK": 4,
        "I;16": 2, "I": 4, "F": 4,
    }
    if mode not in table:
        raise ValueError(f"Unsupported mode {mode!r}")
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
    Merge **all frames** from **all** TIFFs into a single
    macOS-Preview-friendly multi-page TIFF (baseline strips, LZW).
    """
    paths = [Path(p) for p in input_files]
    if not paths:
        raise ValueError("No input files provided.")

    pages = []
    w = h = None
    for p in paths:
        with Image.open(p) as im:
            for i in range(getattr(im, "n_frames", 1)):
                im.seek(i)
                frame = im.convert(mode) if im.mode != mode else im.copy()
                frame.load()
                if w is None:
                    w, h = frame.size
                elif frame.size != (w, h):
                    raise ValueError(
                        f"Frame {i} in {p.name} is {frame.size}, "
                        f"first frame is {(w, h)} – Preview needs uniform size."
                    )
                frame.info["dpi"] = dpi
                pages.append(frame)

    # ---- 4 GB classic-TIFF guard ------------------------------------
    bpp   = _bytes_per_pixel(mode)
    est   = math.ceil(w * h * bpp * len(pages))
    limit = size_limit_gb * 1024 ** 3
    compression = "tiff_lzw"
    if est >= limit:
        warnings.warn(
            "Estimated raw size exceeds classic-TIFF 4 GB limit; "
            "falling back to uncompressed strips (file will be large)."
        )
        compression = "raw"             # Pillow keyword for ‘none’

    # ---- write with Pillow baseline writer --------------------------
    TiffImagePlugin.WRITE_LIBTIFF = False
    pages[0].save(
        output_path,
        save_all=True,
        append_images=pages[1:],
        compression=compression,
        dpi=dpi,
    )

    for im in pages:
        im.close()
    return Path(output_path)
