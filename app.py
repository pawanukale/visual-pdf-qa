from pathlib import Path
import streamlit as st
from src.config import settings
from src.logger import setup_logger
from src.pdf_processor import render_pdf
from src.colpali_engine import ColPaliEngine
from src.embedding_store import save_index, load_index, index_exists
from src.retriever import retrieve
from src.vlm_generator import answer_question
from src.performance_monitor import memory_mb, PeakMemorySampler
from src.logger import logger

st.set_page_config(page_title="Visual PDF QA", page_icon="📄", layout="wide")
setup_logger()
st.title("📄 Visual PDF Question Answering")
st.caption("Visual RAG with ColPali retrieval and Gemini answer generation")

@st.cache_resource(show_spinner="Loading ColPali model (first run downloads model weights)...")
def get_engine():
    return ColPaliEngine()

with st.sidebar:
    st.header("Document indexing")
    uploads = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)
    dpi = st.select_slider("Rendering DPI", options=[100, 110, 120, 130, 140, 150], value=settings.render_dpi)
    if st.button("Index uploaded PDFs", type="primary", disabled=not uploads):
        pdf_dir = settings.pdf_dir
        pdf_dir.mkdir(parents=True, exist_ok=True)
        all_pages = []
        progress = st.progress(0, text="Saving and rendering PDFs...")
        for i, uploaded in enumerate(uploads):
            target = pdf_dir / Path(uploaded.name).name
            target.write_bytes(uploaded.getvalue())
            all_pages.extend(render_pdf(target, dpi=dpi, output_dir=settings.image_dir))
            progress.progress((i + 1) / len(uploads), text=f"Rendered {uploaded.name}")
        if not all_pages:
            st.error("No pages were rendered.")
        else:
            try:
                engine = get_engine()
                with st.spinner(f"Generating visual embeddings for {len(all_pages)} pages..."):
                    with PeakMemorySampler() as sampler:
                        embeddings = engine.embed_pages([p["image_path"] for p in all_pages])
                        elapsed_pages = len(all_pages)
                        save_index(embeddings, all_pages)
                    st.session_state["last_index_seconds"] = engine.last_index_seconds
                    st.session_state["last_index_pages"] = elapsed_pages
                    st.session_state["last_index_ram"] = sampler.peak_mb
                    logger.info("INDEX_METRICS pages=%s indexing_seconds=%.3f peak_rss_mb=%.1f", elapsed_pages, engine.last_index_seconds, sampler.peak_mb)
                st.success(f"Indexed {len(all_pages)} pages successfully.")
            except Exception as exc:
                st.exception(exc)

st.subheader("Ask a question")
question = st.text_input("Your question", placeholder="What does the sales chart on page 3 show?")
col1, col2 = st.columns([1, 4])
with col1:
    top_k = st.selectbox("Pages to retrieve", [2, 3], index=0)
with col2:
    st.caption("The answer is generated only from the retrieved page images. Each factual claim must include a source citation.")

if st.button("Get answer", type="primary", disabled=not question.strip()):
    if not index_exists():
        st.warning("Please upload and index PDF documents first.")
    elif not settings.gemini_api_key:
        st.error("GEMINI_API_KEY is missing. Add it to your .env file and restart Streamlit.")
    else:
        try:
            logger.info("QUESTION_SUBMITTED question=%r", question)
            engine = get_engine()
            index = load_index()
            with PeakMemorySampler() as sampler:
                with st.spinner("Retrieving relevant pages with MaxSim..."):
                    results, query_seconds = retrieve(question, index, engine, top_k=top_k)
                with st.spinner("Generating a grounded answer with Gemini..."):
                    answer, generation_seconds = answer_question(question, results)
            logger.info("QUESTION_DELIVERED retrieval_seconds=%.3f generation_seconds=%.3f peak_rss_mb=%.1f", query_seconds, generation_seconds, sampler.peak_mb)
            st.markdown("### Answer")
            st.write(answer)
            st.markdown("### Retrieved pages")
            for item in results:
                with st.expander(f"{item['filename']} — Page {item['page_number']} | MaxSim: {item['score']:.4f}", expanded=True):
                    st.image(item["image_path"], use_container_width=True)
            st.markdown("### Performance")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Retrieval latency", f"{query_seconds:.2f} s")
            m2.metric("Answer generation", f"{generation_seconds:.2f} s")
            m3.metric("Total query latency", f"{query_seconds + generation_seconds:.2f} s")
            m4.metric("Peak observed RAM", f"{sampler.peak_mb:.1f} MB")
        except Exception as exc:
            st.exception(exc)

if index_exists():
    st.sidebar.success("A saved page index is available.")
if "last_index_seconds" in st.session_state:
    st.sidebar.metric("Last indexing time", f"{st.session_state['last_index_seconds']:.1f} s")
    st.sidebar.metric("Peak observed RAM", f"{st.session_state['last_index_ram']:.1f} MB")
    st.sidebar.caption(f"Pages indexed: {st.session_state['last_index_pages']}")

st.divider()
st.caption("Privacy note: PDF processing and retrieval run locally. Only the top retrieved page images and question are sent to the configured Gemini API.")
