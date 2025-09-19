
import fitz   # pymupdf
from PIL import Image

def pdf_to_tiff(pdf_path: str, tiff_path: str, dpi: int = 300, compress: str = "tiff_lzw"):
    """
    Convert a PDF to a multi-page TIFF.

    - pdf_path: input PDF file
    - tiff_path: output TIFF file
    - dpi: rendering DPI (higher => larger images/files)
    - compress: TIFF compression (e.g., "tiff_lzw", "tiff_adobe_deflate", "group4" for B/W)
    """
    doc = fitz.open(pdf_path)
    pil_pages = []

    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)  # scale from default 72 DPI

    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)  # alpha=False -> RGB or grayscale
        mode = "L" if pix.n == 1 else "RGB"
        img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
        pil_pages.append(img)

    if not pil_pages:
        raise RuntimeError("No pages found in PDF")

    # Save first image and append the rest as multi-page TIFF
    first, rest = pil_pages[0], pil_pages[1:]
    save_kwargs = dict(save_all=True, append_images=rest, compression=compress)
    first.save(tiff_path, format="TIFF", **save_kwargs)

    doc.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python pdf_to_tiff.py input.pdf output.tiff [dpi]")
    else:
        pdf_to_tiff(sys.argv[1], sys.argv[2], dpi=int(sys.argv[3]) if len(sys.argv) > 3 else 300)
        print("Done.")
