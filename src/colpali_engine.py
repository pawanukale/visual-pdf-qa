import time
import torch
from PIL import Image
from src.config import settings
from src.logger import logger

class ColPaliEngine:
    """CPU-first ColQwen2 visual retriever using the ColPali Engine API."""
    def __init__(self):
        try:
            from colpali_engine.models import ColQwen2, ColQwen2Processor
        except ImportError as exc:
            raise RuntimeError("Could not import colpali_engine. Install requirements.txt and restart.") from exc
        self.device = torch.device("cpu")
        logger.info("COLPALI_LOAD_START model=%s device=%s", settings.model_name, self.device)
        self.model = ColQwen2.from_pretrained(
            settings.model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
        ).eval()
        self.processor = ColQwen2Processor.from_pretrained(settings.model_name)
        self.last_index_seconds = 0.0
        logger.info("COLPALI_LOAD_COMPLETE model=%s", settings.model_name)

    def embed_pages(self, image_paths):
        start = time.perf_counter()
        embeddings = []
        for start_idx in range(0, len(image_paths), max(1, settings.batch_size)):
            batch_paths = image_paths[start_idx:start_idx + max(1, settings.batch_size)]
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            try:
                batch = self.processor.process_images(images).to(self.device)
                with torch.inference_mode():
                    batch_embeddings = self.model(**batch)
                embeddings.extend([item.detach().cpu().contiguous() for item in batch_embeddings])
            finally:
                for image in images:
                    image.close()
            logger.info("COLPALI_EMBED_PROGRESS completed=%s total=%s", min(start_idx + len(batch_paths), len(image_paths)), len(image_paths))
        self.last_index_seconds = time.perf_counter() - start
        logger.info("COLPALI_EMBED_COMPLETE pages=%s elapsed_seconds=%.3f", len(embeddings), self.last_index_seconds)
        # score_multi_vector expects a batched tensor. ColQwen2 uses a fixed
        # processor output shape for a given model; fail clearly if pages differ.
        try:
            return torch.stack(embeddings, dim=0)
        except RuntimeError as exc:
            raise RuntimeError("Page embedding shapes differ; use a consistent PDF page size or render configuration.") from exc

    def embed_query(self, query: str):
        batch = self.processor.process_queries([query]).to(self.device)
        with torch.inference_mode():
            embeddings = self.model(**batch)
        return embeddings.detach().cpu()

    def score(self, query_embedding, page_embeddings):
        # The processor performs late-interaction MaxSim scoring.
        return self.processor.score_multi_vector(query_embedding, page_embeddings)
