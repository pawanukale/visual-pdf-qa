import mimetypes
import time
from pathlib import Path
from google import genai
from google.genai import types
from src.config import settings
from src.logger import logger

NOT_FOUND = "Not found in the provided documents."
SYSTEM_INSTRUCTION = """You are a careful visual document question-answering assistant. Answer using ONLY the supplied PDF page images. Do not use outside knowledge or infer facts that are not visible. Every factual claim must end with a citation exactly like [filename.pdf, Page 2]. If the requested information cannot be clearly verified from the supplied pages, respond with exactly: Not found in the provided documents. Do not mention pages that were not supplied. Be concise and preserve units, labels, dates, and values as shown."""

def answer_question(question: str, retrieved_pages):
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured in .env")
    client = genai.Client(api_key=settings.gemini_api_key)
    parts = [types.Part.from_text(text=f"Question: {question}\n\nUse only the following retrieved PDF pages. Cite every factual claim.")]
    for page in retrieved_pages:
        path = Path(page["image_path"])
        mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        parts.append(types.Part.from_text(text=f"Source page: {page['filename']}, Page {page['page_number']}"))
        parts.append(types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type))
    start = time.perf_counter()
    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.1,
        ),
    )
    elapsed = time.perf_counter() - start
    answer = (response.text or "").strip()
    if not answer:
        answer = NOT_FOUND
    logger.info("VLM_ANSWER_COMPLETE model=%s elapsed_seconds=%.3f", settings.gemini_model, elapsed)
    return answer, elapsed
