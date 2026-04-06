import logging
from collections import deque
from pathlib import Path

import streamlit as st

from backend.api.rag_service import RAGService
from backend.config.logger import setup_logging
from backend.config.settings import get_settings
from backend.embedding.pipeline import EmbeddingPipeline
from backend.emotion.detector import EmotionDetector
from backend.llm.huggingface_client import HuggingFaceInferenceClient
from backend.llm.prompt_builder import PromptBuilder
from backend.memory.manager import MemoryManager
from backend.retrieval.retriever import Retriever
from backend.retrieval.vector_store import FaissVectorStore


@st.cache_resource(show_spinner="Loading models and vector store (once)...")
def _get_shared_components():
    settings = get_settings()
    embedding_pipeline = EmbeddingPipeline(
        model_name=settings.embedding_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    vector_store = FaissVectorStore(
        persist_dir=settings.persist_dir,
        embedding_pipeline=embedding_pipeline,
        ltm_max_entries=settings.ltm_max_entries,
    )
    # Load or build exactly once for the whole process.
    if vector_store.exists():
        vector_store.load()
    else:
        from backend.ingestion.loaders import load_all_documents

        docs = load_all_documents(settings.data_dir)
        vector_store.build_from_documents(docs)

    emotion_detector = EmotionDetector(model_name=settings.emotion_model)
    retriever = Retriever(
        vector_store,
        semantic_weight=settings.retrieval_semantic_weight,
        emotion_weight=settings.retrieval_emotion_weight,
        recency_weight=settings.retrieval_recency_weight,
        half_life_hours=settings.retrieval_half_life_hours,
        candidate_pool=settings.retrieval_candidate_pool,
    )
    llm_client = HuggingFaceInferenceClient(
        model_name=settings.hf_model,
        api_token=settings.hf_api_token,
        max_new_tokens=settings.hf_max_new_tokens,
        timeout_s=settings.hf_timeout_s,
        temperature=settings.hf_temperature,
    )
    prompt_builder = PromptBuilder()

    return {
        "vector_store": vector_store,
        "emotion_detector": emotion_detector,
        "retriever": retriever,
        "llm_client": llm_client,
        "prompt_builder": prompt_builder,
    }


def _init_session_state() -> None:
    settings = get_settings()
    st.session_state.setdefault("chat_history", [])
    # STM must be session-local; it is used to seed prompt continuity.
    st.session_state.setdefault("stm_entries", deque(maxlen=settings.session_stm_size))
    st.session_state.setdefault("show_retrieved", False)
    st.session_state.setdefault("debug_mode", False)


def _append_chat(role: str, content: str) -> None:
    st.session_state["chat_history"].append({"role": role, "content": content})


def _save_uploaded_file(uploaded_file) -> str:
    save_directory = Path("data") / "text_files"
    save_directory.mkdir(parents=True, exist_ok=True)
    save_path = save_directory / uploaded_file.name
    save_path.write_bytes(uploaded_file.getbuffer())
    return str(save_path)


if __name__ == "__main__":
    setup_logging(logging.INFO)
    _init_session_state()
    st.title("AI Journaling Assistant")
    shared = _get_shared_components()

    # Create a per-session memory manager + service wrapper.
    if "service" not in st.session_state:
        memory_manager = MemoryManager(
            vector_store=shared["vector_store"],
            emotion_detector=shared["emotion_detector"],
            stm=st.session_state["stm_entries"],
        )
        st.session_state["service"] = RAGService(
            vector_store=shared["vector_store"],
            emotion_detector=shared["emotion_detector"],
            retriever=shared["retriever"],
            prompt_builder=shared["prompt_builder"],
            llm_client=shared["llm_client"],
            memory_manager=memory_manager,
        )
    service: RAGService = st.session_state["service"]

    col1, col2 = st.columns([3, 1])
    with col1:
        st.session_state["show_retrieved"] = st.checkbox(
            "Show retrieved memories (top context)",
            value=st.session_state["show_retrieved"],
        )
    with col2:
        st.session_state["debug_mode"] = st.checkbox(
            "Debug mode (emotion + prompt)",
            value=st.session_state["debug_mode"],
        )

    if st.button("Reset Session"):
        st.session_state["chat_history"] = []
        settings = get_settings()
        st.session_state["stm_entries"] = deque(maxlen=settings.session_stm_size)
        st.session_state.pop("service", None)
        st.rerun()

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_text = st.chat_input("Write your journal update...")
    if user_text:
        _append_chat("user", user_text)
        with st.chat_message("user"):
            st.markdown(user_text)

        try:
            output = service.run_pipeline(
                text=user_text,
                chat_history=st.session_state["chat_history"],
                top_k=3,
            )
            emotion = output["emotion"]
            response = output["response"]
            prompt = output["prompt"]
            retrieved = output["retrieved_memories"]

            assistant_text = (
                f"Emotion detected: **{emotion['emotion']}** ({emotion['confidence']:.2f})\n\n"
                f"{response}"
            )
            _append_chat("assistant", assistant_text)

            with st.chat_message("assistant"):
                st.markdown(assistant_text)
                if st.session_state["show_retrieved"]:
                    st.caption("Retrieved memories used:")
                    st.json(retrieved)
                if st.session_state["debug_mode"]:
                    st.subheader("Debug: detected emotion payload")
                    st.json(emotion)
                    with st.expander("Debug: retrieved memories (top context)"):
                        st.json(retrieved)
                    with st.expander("Debug: final constructed prompt"):
                        st.code(prompt)
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")

    uploaded_file = st.file_uploader("Upload a file")
    if uploaded_file is not None:
        try:
            save_path = _save_uploaded_file(uploaded_file)
            st.success(f"File saved at: {save_path}")
        except Exception as exc:
            st.error(f"Failed to save file: {exc}")
