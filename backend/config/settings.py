from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    data_dir: str = os.getenv("DATA_DIR", "data")
    persist_dir: str = os.getenv("VECTOR_STORE_DIR", "faiss_store")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    hf_model: str = os.getenv("HF_MODEL", "google/flan-t5-small")
    hf_api_token: str | None = os.getenv("HF_API_TOKEN")
    hf_max_new_tokens: int = int(os.getenv("HF_MAX_NEW_TOKENS", "220"))
    hf_timeout_s: int = int(os.getenv("HF_TIMEOUT_S", "30"))
    hf_temperature: float = float(os.getenv("HF_TEMPERATURE", "0.2"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    memory_stm_size: int = int(os.getenv("STM_SIZE", "10"))
    session_stm_size: int = int(os.getenv("SESSION_STM_SIZE", "8"))
    emotion_model: str = os.getenv(
        "EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base"
    )
    retrieval_semantic_weight: float = float(os.getenv("RETRIEVAL_SEMANTIC_WEIGHT", "0.6"))
    retrieval_emotion_weight: float = float(os.getenv("RETRIEVAL_EMOTION_WEIGHT", "0.25"))
    retrieval_recency_weight: float = float(os.getenv("RETRIEVAL_RECENCY_WEIGHT", "0.15"))
    retrieval_half_life_hours: float = float(os.getenv("RETRIEVAL_HALF_LIFE_HOURS", "72"))
    retrieval_candidate_pool: int = int(os.getenv("RETRIEVAL_CANDIDATE_POOL", "20"))
    ltm_max_entries: int = int(os.getenv("LTM_MAX_ENTRIES", "0"))


def get_settings() -> Settings:
    return Settings()
