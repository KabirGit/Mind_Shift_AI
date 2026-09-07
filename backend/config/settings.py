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
    llm_backend: str = os.getenv("LLM_BACKEND", "huggingface").strip().lower()
    hf_model: str = os.getenv("HF_MODEL", "arsoban/ocd-therapist-27b-v0.3")
    hf_inference_provider: str = os.getenv(
        "HF_INFERENCE_PROVIDER", "featherless-ai"
    )
    hf_api_token: str | None = (
        os.getenv("HF_TOKEN")
        or os.getenv("HF_API_TOKEN")
        or os.getenv("HUGGINGFACE_API_KEY")
    )
    hf_max_new_tokens: int = int(os.getenv("HF_MAX_NEW_TOKENS", "220"))
    hf_connect_timeout_s: float = float(os.getenv("HF_CONNECT_TIMEOUT_S", "5"))
    hf_read_timeout_s: float = float(os.getenv("HF_READ_TIMEOUT_S", "30"))
    hf_temperature: float = float(os.getenv("HF_TEMPERATURE", "0.2"))
    hf_max_attempts: int = int(os.getenv("HF_MAX_ATTEMPTS", "2"))
    hf_retry_backoff_s: float = float(os.getenv("HF_RETRY_BACKOFF_S", "0.25"))
    mistral_model: str = os.getenv("MISTRAL_MODEL", "mistral-small")
    mistral_api_key: str | None = os.getenv("MISTRAL_API_KEY")
    mistral_max_tokens: int = int(
        os.getenv("MISTRAL_MAX_TOKENS", os.getenv("HF_MAX_NEW_TOKENS", "220"))
    )
    mistral_connect_timeout_s: float = float(
        os.getenv("MISTRAL_CONNECT_TIMEOUT_S", "5")
    )
    mistral_read_timeout_s: float = float(
        os.getenv("MISTRAL_READ_TIMEOUT_S", os.getenv("HF_TIMEOUT_S", "30"))
    )
    mistral_temperature: float = float(
        os.getenv("MISTRAL_TEMPERATURE", os.getenv("HF_TEMPERATURE", "0.2"))
    )
    mistral_max_attempts: int = int(os.getenv("MISTRAL_MAX_ATTEMPTS", "2"))
    mistral_retry_backoff_s: float = float(
        os.getenv("MISTRAL_RETRY_BACKOFF_S", "0.25")
    )
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
    guidance_context_token_budget: int = int(
        os.getenv("GUIDANCE_CONTEXT_TOKEN_BUDGET", "1500")
    )
    guidance_context_characters_per_token: int = int(
        os.getenv("GUIDANCE_CONTEXT_CHARACTERS_PER_TOKEN", "4")
    )
    ltm_max_entries: int = int(os.getenv("LTM_MAX_ENTRIES", "0"))
    dashboard_min_insight_confidence: float = float(
        os.getenv("DASHBOARD_MIN_INSIGHT_CONFIDENCE", "0.5")
    )
    dashboard_min_mention_count: int = int(os.getenv("DASHBOARD_MIN_MENTION_COUNT", "3"))
    dashboard_min_entry_count: int = int(os.getenv("DASHBOARD_MIN_ENTRY_COUNT", "5"))

def get_settings() -> Settings:
    return Settings()
