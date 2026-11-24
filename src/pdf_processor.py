from pathlib import Path
import fitz
from PIL import Image
from src.logger import logger
from src.config import settings

def render_pdf(pdf_path: Path, dpi: int = None, output_dir: Path = None):
    """Render every PDF page to an image. No OCR or text extraction is performed."""
    dpi = dpi or settings.render_dpi
    output_dir = output_dir or settings.image_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    document = fitz.open(pdf_path)
    scale = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)
    pages = []
    safe_stem = pdf_path.stem.replace(" ", "_")
    logger.info("PDF_RENDER_START file=%s pages=%s dpi=%s", pdf_path.name, len(document), dpi)
    for page_index, page in enumerate(document):
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        image_path = output_dir / f"{safe_stem}_page_{page_index + 1:04d}.jpg"
        pix.save(str(image_path), jpg_quality=90)
        # Verify the rendered image is readable before it enters the index.
        with Image.open(image_path) as image:
            image.verify()
        pages.append({
            "filename": pdf_path.name,
            "page_number": page_index + 1,
            "image_path": str(image_path.resolve()),
        })
    document.close()
    logger.info("PDF_RENDER_COMPLETE file=%s rendered_pages=%s", pdf_path.name, len(pages))
    return pages
