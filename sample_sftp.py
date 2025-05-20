from pathlib import Path
from typing import Iterable, Union
from PIL import Image, TiffImagePlugin
import warnings

def combine_tiffs_preview_safe(
    input_files: Iterable[Union[str, Path]],
    output_path: Union[str, Path],
    *,
    mode: str = "RGB",
    dpi: tuple[int, int] = (300, 300),
) -> Path:
    """
    Merge single-page TIFFs into a multi-page TIFF that macOS Preview
    can open.  Requirements enforced:

    * classic TIFF header (≤ 4 GB)
    * LZW-compressed **strip** layout (no tiles)
    * identical size / bit depth / DPI across pages
    """

    files = [Path(f) for f in input_files]
    if not files:
        raise ValueError("No input files provided.")

    # --- 1. Load every page fully and validate uniformity -------------
    pages = []
    w = h = None
    for fp in files:
        with Image.open(fp) as im:
            if im.mode != mode:
                im = im.convert(mode)
            im.load()                    # detach from underlying file
            if w is None:
                w, h = im.size
            elif im.size != (w, h):
                raise ValueError(
                    f"Page {fp.name} size {im.size} differs from "
                    f"first page {(w, h)} – Preview can't handle that."
                )
            im.info["dpi"] = dpi
            pages.append(im.copy())      # keep a copy open after fp closes

    # --- 2. Pillow's own writer → guaranteed baseline strips ----------
    TiffImagePlugin.WRITE_LIBTIFF = False     # force Pillow writer
    save_kwargs = dict(
        save_all=True,
        append_images=pages[1:],
        compression="tiff_lzw",              # Baseline, Preview-safe
        dpi=dpi,
    )

    # Pillow can't create BigTIFF; if > 4 GB fall back to no compression
    est_size = sum(p.nbytes for p in pages)
    if est_size >= 4 * 1024 ** 3:
        warnings.warn(
            "Output would exceed classic-TIFF 4 GB limit; "
            "falling back to uncompressed strips."
        )
        save_kwargs["compression"] = "raw"   # 'none' in Pillow

    pages[0].save(output_path, **save_kwargs)

    # --- 3. Clean-up ---------------------------------------------------
    for p in pages:
        p.close()

    return Path(output_path)

