import logging
from typing import Any

from backend.config.debug import log_stage
from backend.config.settings import get_settings
from backend.emotion.detector import EmotionDetector
from backend.ingestion.loaders import load_all_documents
from backend.llm.huggingface_client import HuggingFaceInferenceClient
from backend.llm.prompt_builder import PromptBuilder
from backend.memory.manager import MemoryManager
from backend.retrieval.retriever import Retriever
from backend.retrieval.vector_store import FaissVectorStore

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(
        self,
        *,
        vector_store: FaissVectorStore | None = None,
        emotion_detector: EmotionDetector | None = None,
        retriever: Retriever | None = None,
        prompt_builder: PromptBuilder | None = None,
        llm_client: HuggingFaceInferenceClient | None = None,
        memory_manager: MemoryManager | None = None,
    ) -> None:
        """
        Uses injected components when provided, to avoid reloading models
        across Streamlit reruns while keeping STM session-local.
        """

        settings = get_settings()

        self.vector_store = vector_store
        if self.vector_store is None:
            # Lazy import: embedding pipeline depends on heavy model loading.
            from backend.embedding.pipeline import EmbeddingPipeline

            embedding_pipeline = EmbeddingPipeline(
                model_name=settings.embedding_model,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            self.vector_store = FaissVectorStore(
                persist_dir=settings.persist_dir,
                embedding_pipeline=embedding_pipeline,
                ltm_max_entries=settings.ltm_max_entries,
            )
            self._load_or_build_store(data_dir=settings.data_dir)

        self.emotion_detector = (
            emotion_detector
            if emotion_detector is not None
            else EmotionDetector(model_name=settings.emotion_model)
        )

        self.retriever = (
            retriever
            if retriever is not None
            else Retriever(
                self.vector_store,
                semantic_weight=settings.retrieval_semantic_weight,
                emotion_weight=settings.retrieval_emotion_weight,
                recency_weight=settings.retrieval_recency_weight,
                half_life_hours=settings.retrieval_half_life_hours,
                candidate_pool=settings.retrieval_candidate_pool,
            )
        )

        self.prompt_builder = prompt_builder if prompt_builder is not None else PromptBuilder()

        self.llm = (
            llm_client
            if llm_client is not None
            else HuggingFaceInferenceClient(
                model_name=settings.hf_model,
                api_token=settings.hf_api_token,
                max_new_tokens=settings.hf_max_new_tokens,
                timeout_s=settings.hf_timeout_s,
                temperature=settings.hf_temperature,
            )
        )

        self.memory = (
            memory_manager
            if memory_manager is not None
            else MemoryManager(
                vector_store=self.vector_store,
                emotion_detector=self.emotion_detector,
                stm_size=settings.memory_stm_size,
            )
        )

    def _load_or_build_store(self, data_dir: str) -> None:
        try:
            if self.vector_store.exists():
                self.vector_store.load()
                return
            logger.info("Vector store not found, building from data directory.")
            docs = load_all_documents(data_dir)
            self.vector_store.build_from_documents(docs)
        except Exception as exc:
            logger.exception("Failed to load/build vector store: %s", exc)
            raise

    def run_pipeline(
        self,
        text: str,
        chat_history: list[dict[str, str]] | None = None,
        top_k: int = 5,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        emotion = self.emotion_detector.detect(text)
        log_stage("emotion", {"input": text[:120], "emotion": emotion})

        stored_entry = self.memory.store_entry(
            text=text,
            tags=tags,
            emotion_signal=emotion,
        )
        # Retrieval uses long-term memory (FAISS). We also inject STM for
        # conversational continuity (recent entries from this session).
        retrieved_ltm = self.retriever.retrieve(
            query=text,
            query_emotion=emotion.get("emotion", "neutral"),
            top_k=top_k,
        )

        recent_memories = self.memory.get_recent_memory(limit=3)
        stm_items = [{"metadata": m, "scores": {"combined": None}} for m in recent_memories]

        # Keep the ordering stable: STM first, then LTM.
        merged = stm_items + retrieved_ltm

        log_stage(
            "retrieval",
            {
                "top_k": top_k,
                "results": [
                    {
                        "text": r.get("metadata", {}).get("text", "")[:120],
                        "emotion": r.get("metadata", {}).get("emotion"),
                        "timestamp": r.get("metadata", {}).get("timestamp"),
                        "scores": r.get("scores"),
                    }
                    for r in merged[: top_k + 3]
                ],
            },
        )

        prompt = self.prompt_builder.build(
            user_text=text,
            current_emotion=emotion,
            retrieved_memories=merged,
            recent_history=chat_history or [],
        )
        log_stage("prompt", {"prompt_preview": prompt[:800]})
        response = self.llm.generate(prompt)

        return {
            "emotion": emotion,
            "stored_entry": stored_entry,
            "retrieved_memories": merged,
            "prompt": prompt,
            "response": response,
        }
