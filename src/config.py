from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

# Ensure Hugging Face Hub uses stable HTTP downloads and respects custom cache directory
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
if os.getenv("HF_HOME"):
    os.environ["HF_HOME"] = os.getenv("HF_HOME")

@dataclass(frozen=True)
class Settings:
    root_dir: Path = ROOT_DIR
    pdf_dir: Path = ROOT_DIR / "data" / "pdfs"
    image_dir: Path = ROOT_DIR / "data" / "page_images"
    index_dir: Path = ROOT_DIR / "data" / "embeddings"
    log_dir: Path = ROOT_DIR / "logs"
    model_name: str = os.getenv("COLPALI_MODEL", "vidore/colqwen2-v1.0")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    render_dpi: int = int(os.getenv("RENDER_DPI", "100"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "1"))

settings = Settings()
for folder in (settings.pdf_dir, settings.image_dir, settings.index_dir, settings.log_dir):
    folder.mkdir(parents=True, exist_ok=True)
