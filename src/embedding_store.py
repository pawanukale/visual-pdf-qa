import json
from pathlib import Path
import torch
from src.config import settings
from src.logger import logger

INDEX_FILE = settings.index_dir / "colpali_page_embeddings.pt"
METADATA_FILE = settings.index_dir / "page_metadata.json"

def save_index(embeddings, metadata):
    if embeddings.shape[0] != len(metadata):
        raise ValueError("Embedding count and page metadata count do not match.")
    payload = {"embeddings": embeddings, "model_name": settings.model_name}
    torch.save(payload, INDEX_FILE)
    METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("INDEX_SAVED embeddings=%s path=%s", len(embeddings), INDEX_FILE)

def load_index():
    if not index_exists():
        raise FileNotFoundError("No saved index found. Index PDFs first.")
    payload = torch.load(INDEX_FILE, map_location="cpu", weights_only=False)
    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    if payload.get("model_name") != settings.model_name:
        raise ValueError("Saved index model differs from configured model. Re-index the PDFs.")
    return {"embeddings": payload["embeddings"], "metadata": metadata}

def index_exists():
    return INDEX_FILE.exists() and METADATA_FILE.exists()
