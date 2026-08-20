import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    data_dir: str = os.getenv("DATA_DIR", "data")
    persist_dir: str = os.getenv("VECTOR_STORE_DIR", "faiss_store")
    sqlite_path: str = os.getenv("SQLITE_PATH", "data/journal.db")
    latency_log_path: str = os.getenv("LATENCY_LOG_PATH", "data/latency_log.jsonl")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    hf_model: str = os.getenv("MISTRAL_MODEL", os.getenv("HF_MODEL", "mistral-small"))
    hf_api_token: str | None = os.getenv("MISTRAL_API_KEY") or os.getenv("HF_API_TOKEN")
    hf_max_new_tokens: int = int(os.getenv("HF_MAX_NEW_TOKENS", "220"))
    hf_timeout_s: int = int(os.getenv("HF_TIMEOUT_S", "30"))
    hf_temperature: float = float(os.getenv("HF_TEMPERATURE", "0.2"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    memory_stm_size: int = int(os.getenv("STM_SIZE", "10"))
    session_stm_size: int = int(os.getenv("SESSION_STM_SIZE", "8"))
    emotion_model: str = os.getenv(
        "EMOTION_MODEL", "SamLowe/roberta-base-go_emotions"
    )
    retrieval_semantic_weight: float = float(os.getenv("RETRIEVAL_SEMANTIC_WEIGHT", "0.6"))
    retrieval_emotion_weight: float = float(os.getenv("RETRIEVAL_EMOTION_WEIGHT", "0.25"))
    retrieval_recency_weight: float = float(os.getenv("RETRIEVAL_RECENCY_WEIGHT", "0.15"))
    retrieval_half_life_hours: float = float(os.getenv("RETRIEVAL_HALF_LIFE_HOURS", "72"))
    retrieval_candidate_pool: int = int(os.getenv("RETRIEVAL_CANDIDATE_POOL", "20"))
    ltm_max_entries: int = int(os.getenv("LTM_MAX_ENTRIES", "0"))
    dashboard_min_insight_confidence: float = float(
        os.getenv("DASHBOARD_MIN_INSIGHT_CONFIDENCE", "0.5")
    )
    dashboard_min_mention_count: int = int(os.getenv("DASHBOARD_MIN_MENTION_COUNT", "3"))
    dashboard_min_entry_count: int = int(os.getenv("DASHBOARD_MIN_ENTRY_COUNT", "5"))


def get_settings() -> Settings:
    return Settings()
