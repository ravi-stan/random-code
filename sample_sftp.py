#!/usr/bin/env python3
"""
pdf_to_tiff.py — High-quality, small-size PDF → TIFF converter using PyMuPDF + Pillow.

Why this exists
---------------
Directly saving PyMuPDF Pixmaps to TIFF often produces *uncompressed* or suboptimal TIFFs.
This script renders each PDF page at an appropriate DPI, detects the page's content type,
and applies the most size-efficient (and standards-friendly) TIFF compression that preserves readability.

Key ideas
---------
1) Correct DPI: Too-high DPI inflates size; too-low hurts legibility. Default is 300 for text/line art,
   300 for grayscale, and 240 for photographic/color pages.
2) Content-aware compression:
   - Text/line art (bilevel-friendly) → 1-bit + CCITT Group 4 (lossless for bilevel)  → tiny files.
   - Grayscale or light color pages → Lossless Deflate (a.k.a. Adobe Deflate / zlib) or LZW.
   - Photographic pages → Optionally JPEG-in-TIFF with high quality (visually lossless), otherwise Deflate.
3) No alpha channels. Color management preserved when possible.

Dependencies
------------
    pip install --upgrade pymupdf pillow numpy

Usage
-----
    python pdf_to_tiff.py input.pdf output.tiff \
        [--dpi-text 300] [--dpi-gray 300] [--dpi-color 240] \
        [--lossless {adobe_deflate,lzw}] \
        [--enable-jpeg-for-color] [--jpeg-quality 90] [--no-bilevel] \
        [--pages 1-3,7,10-] [--big-tiff]

Notes
-----
- Group 4 requires a *bilevel* image (mode '1'). We use Otsu thresholding + sanity checks.
- If you prefer strictly lossless for *all* pages, do not pass --enable-jpeg-for-color.
- TIFF supports per-page compression. We leverage Pillow's AppendingTiffWriter to set
  compression individually per frame.

Author: (c) 2025
License: MIT
"""

from __future__ import annotations

import argparse
import io
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Third-party
try:
    import fitz  # PyMuPDF
except ImportError as e:
    print("ERROR: PyMuPDF (fitz) not installed. Run: pip install pymupdf", file=sys.stderr)
    raise

try:
    from PIL import Image, ImageOps, TiffImagePlugin
except ImportError as e:
    print("ERROR: Pillow not installed. Run: pip install pillow", file=sys.stderr)
    raise

try:
    import numpy as np
except ImportError as e:
    print("ERROR: numpy not installed. Run: pip install numpy", file=sys.stderr)
    raise


def _parse_pages_arg(pages: str, page_count: int) -> List[int]:
    """
    Parse pages like '1-3,7,10-' into a zero-based list of indices.
    '10-' means from 10 to end (1-based in user input).
    """
    result = set()
    tokens = [t.strip() for t in pages.split(",") if t.strip()]
    for tk in tokens:
        if "-" in tk:
            start_str, end_str = tk.split("-", 1)
            start = int(start_str) if start_str else 1
            end = int(end_str) if end_str else page_count
            # clamp and convert to 0-based
            start = max(1, start)
            end = min(page_count, end)
            for p in range(start, end + 1):
                result.add(p - 1)
        else:
            p = int(tk)
            if 1 <= p <= page_count:
                result.add(p - 1)
    return sorted(result)


def _colorfulness_hasler_suesstrunk(rgb: Image.Image) -> float:
    """Compute colorfulness index (Hasler & Suesstrunk).
    Returns 0 for grayscale-ish, larger for colorful images.
    """
    if rgb.mode != "RGB":
        rgb = rgb.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return 0.0
    R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    std_rg, std_yb = rg.std(), yb.std()
    mean_rg, mean_yb = rg.mean(), yb.mean()
    return float(math.sqrt(std_rg**2 + std_yb**2) + 0.3 * math.sqrt(mean_rg**2 + mean_yb**2))


def _otsu_threshold(gray: np.ndarray) -> int:
    """Compute Otsu threshold on uint8 grayscale array -> int [0..255]."""
    hist = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
    total = hist.sum()
    prob = hist / (total + 1e-12)
    cum_prob = np.cumsum(prob)
    cum_mean = np.cumsum(prob * np.arange(256))
    global_mean = cum_mean[-1]
    denom = cum_prob * (1.0 - cum_prob)
    denom[denom == 0] = 1e-12
    bc_var = ((global_mean * cum_prob - cum_mean) ** 2) / denom
    t = int(np.nanargmax(bc_var))
    return t


def _bilevel_suitability(gray: Image.Image) -> Tuple[bool, int]:
    """
    Decide if page is "bilevel-friendly" using a quick check:
    - compute Otsu threshold
    - compute fraction of pixels near extremes (<= 16 or >= 239)
    If >= 0.94 (94%), we consider it bilevel-friendly.
    """
    if gray.mode != "L":
        gray = gray.convert("L")
    arr = np.asarray(gray, dtype=np.uint8)
    t = _otsu_threshold(arr)
    near_extremes = ((arr <= 16) | (arr >= 239)).mean()
    return (near_extremes >= 0.94), t


def _render_page_image(page: "fitz.Page", dpi: int, mode: str) -> Image.Image:
    """
    Render a PyMuPDF page to a PIL Image at the requested dpi.
    mode: 'RGB' or 'L'
    """
    # Since PyMuPDF 1.19.2, page.get_pixmap(dpi=...) sets dpi metadata and is convenient.
    colorspace = fitz.csRGB if mode == "RGB" else fitz.csGRAY
    pix = page.get_pixmap(dpi=dpi, colorspace=colorspace, alpha=False)
    # Convert to PIL image without touching disk.
    if pix.alpha:  # Just in case
        pix = fitz.Pixmap(fitz.csRGB, pix)  # drop alpha
    if pix.n == 1:
        pil = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    else:
        pil = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        if mode == "L":
            pil = pil.convert("L")
    return pil


@dataclass
class PageDecision:
    kind: str            # 'bilevel' | 'gray' | 'color'
    dpi: int
    compression: str     # Pillow compression token, e.g., 'group4', 'tiff_adobe_deflate', 'tiff_lzw', 'tiff_jpeg'
    jpeg_quality: Optional[int] = None


def choose_strategy(preview_rgb: Image.Image,
                    args) -> PageDecision:
    """Heuristics to choose per-page save strategy."""
    cf = _colorfulness_hasler_suesstrunk(preview_rgb)
    is_color = cf >= args.colorfulness_threshold

    # Work in grayscale to test bilevel suitability
    preview_gray = preview_rgb.convert("L")
    is_bilevel, _ = _bilevel_suitability(preview_gray)

    if is_bilevel and not args.no_bilevel:
        return PageDecision(kind="bilevel",
                            dpi=args.dpi_text,
                            compression="group4",
                            jpeg_quality=None)

    if is_color:
        if args.enable_jpeg_for_color:
            return PageDecision(kind="color",
                                dpi=args.dpi_color,
                                compression="tiff_jpeg",
                                jpeg_quality=args.jpeg_quality)
        else:
            return PageDecision(kind="color",
                                dpi=args.dpi_color if args.dpi_color else args.dpi_gray,
                                compression=("tiff_adobe_deflate" if args.lossless == "adobe_deflate" else "tiff_lzw"),
                                jpeg_quality=None)
    # grayscale
    return PageDecision(kind="gray",
                        dpi=args.dpi_gray,
                        compression=("tiff_adobe_deflate" if args.lossless == "adobe_deflate" else "tiff_lzw"),
                        jpeg_quality=None)


def save_append_tiff(fp, image: Image.Image, compression: str, dpi_xy: Tuple[int, int],
                     jpeg_quality: Optional[int] = None, bigtiff: bool = False):
    """
    Append a single frame to an open TiffImagePlugin.AppendingTiffWriter.
    """
    # Pillow expects compression names like: "group4", "tiff_adobe_deflate", "tiff_lzw", "tiff_jpeg"
    encoderinfo = {}
    if compression == "tiff_jpeg" and jpeg_quality is not None:
        # as of Pillow, use 'quality' for JPEG-in-TIFF
        encoderinfo["quality"] = int(jpeg_quality)
        # and avoid chroma subsampling for crisp text in images
        encoderinfo["subsampling"] = 0

    # Some Pillow/libtiff combos want float DPI
    dpi_xy = (float(dpi_xy[0]), float(dpi_xy[1]))

    image.save(fp,
               format="TIFF",
               compression=compression,
               dpi=dpi_xy,
               save_all=False,
               **encoderinfo)


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                description="Convert PDF to size-efficient TIFF without compromising readability.")
    p.add_argument("pdf", help="Input PDF path")
    p.add_argument("tiff", help="Output TIFF path")
    p.add_argument("--pages", default=None, help="Pages to include, e.g., '1-3,7,10-'. Default: all")
    p.add_argument("--dpi-text", type=int, default=300, help="DPI for bilevel (text/line art) pages")
    p.add_argument("--dpi-gray", type=int, default=300, help="DPI for grayscale pages")
    p.add_argument("--dpi-color", type=int, default=240, help="DPI for color/photographic pages")
    p.add_argument("--lossless", choices=["adobe_deflate", "lzw"], default="adobe_deflate",
                   help="Lossless compression for non-bilevel pages")
    p.add_argument("--enable-jpeg-for-color", action="store_true",
                   help="Use JPEG-in-TIFF for color/photographic pages (smaller, slightly lossy)")
    p.add_argument("--jpeg-quality", type=int, default=92,
                   help="JPEG quality if --enable-jpeg-for-color is set")
    p.add_argument("--no-bilevel", action="store_true", help="Disable 1-bit + Group4 for text pages")
    p.add_argument("--colorfulness-threshold", type=float, default=15.0,
                   help="Colorfulness threshold separating grayscale vs color pages")
    p.add_argument("--preview-dpi", type=int, default=150, help="DPI used only for content classification preview")
    p.add_argument("--big-tiff", action="store_true", help="(Advisory only) If output may exceed 4GB, consider using a BigTIFF-capable build of Pillow or split output.")

    args = p.parse_args()

    if not os.path.exists(args.pdf):
        print(f"ERROR: Input PDF not found: {args.pdf}", file=sys.stderr)
        sys.exit(2)

    # Open PDF
    doc = fitz.open(args.pdf)
    page_indices = list(range(doc.page_count))
    if args.pages:
        page_indices = _parse_pages_arg(args.pages, doc.page_count)
        if not page_indices:
            print("ERROR: --pages resolved to an empty set.", file=sys.stderr)
            sys.exit(2)

    # Prepare TIFF writer
    # If file exists, overwrite
    if os.path.exists(args.tiff):
        os.remove(args.tiff)

    # NOTE: Pillow exposes an "AppendingTiffWriter" to append frames with per-frame options.
    # 'bigtiff' is supported when creating a new file via mode="w" and setting 'bigtiff=True' in Pillow>=10.
    # NOTE: If you truly need BigTIFF (>4GB) and your libtiff build supports it,
    # you may need to toggle Pillow internals (e.g., TiffImagePlugin.WRITE_LIBTIFF = True)
    # or use the 'tifffile' library. Here we keep maximum compatibility.
    with TiffImagePlugin.AppendingTiffWriter(args.tiff) as tf:
        for i, pno in enumerate(page_indices, start=1):
            page = doc.load_page(pno)

            # Quick preview render at low DPI for classification
            preview = _render_page_image(page, dpi=args.preview_dpi, mode="RGB")
            decision = choose_strategy(preview, args)

            # Final render per decision
            render_mode = "RGB" if decision.kind == "color" else "L"
            final_img = _render_page_image(page, dpi=decision.dpi, mode=render_mode)

            # If bilevel: binarize and switch to mode '1'
            if decision.kind == "bilevel":
                gray = final_img.convert("L") if final_img.mode != "L" else final_img
                arr = np.asarray(gray, dtype=np.uint8)
                t = _otsu_threshold(arr)
                bin_arr = (arr > t).astype(np.uint8) * 255
                final_img = Image.fromarray(bin_arr, mode="L").convert("1")  # 1-bit pixels

            # Append with per-page compression and DPI
            save_append_tiff(tf, final_img, decision.compression, (decision.dpi, decision.dpi),
                             jpeg_quality=decision.jpeg_quality)

            # After each save, advance to a new frame
            tf.newFrame()

            print(f"[{i}/{len(page_indices)}] page={pno+1} kind={decision.kind} dpi={decision.dpi} comp={decision.compression}")

    doc.close()
    print(f"Done. Wrote: {args.tiff}")


if __name__ == "__main__":
    main()

