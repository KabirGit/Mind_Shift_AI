import json
import logging
import os
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from backend.config.debug import log_stage
from backend.config.settings import get_settings
from backend.emotion.detector import EmotionDetector
from backend.ingestion.loaders import load_all_documents
from backend.llm.huggingface_client import HuggingFaceInferenceClient
from backend.llm.prompt_builder import PromptBuilder
from backend.memory.manager import MemoryManager
from backend.retrieval.retriever import Retriever
from backend.retrieval.vector_store import FaissVectorStore
from backend.storage.db import JournalDB
from backend.storage.models import JournalRecord

if TYPE_CHECKING:
    from backend.nlp.text_processor import TextProcessor

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
        journal_db: JournalDB | None = None,
        text_processor: "TextProcessor | None" = None,
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

        # Structured metadata store (additive; never affects FAISS path).
        self.journal_db = (
            journal_db if journal_db is not None else JournalDB(settings.sqlite_path)
        )

        # Local-only NLP enrichment (Phase 2). Lazy-loads spaCy/VADER on use.
        if text_processor is not None:
            self.text_processor = text_processor
        else:
            from backend.nlp.text_processor import TextProcessor

            self.text_processor = TextProcessor()

        # Phase 3/4/7/8: deterministic analytics + insight generation (no LLM).
        from backend.analytics.habit_engine import HabitEngine
        from backend.analytics.insight_engine import InsightEngine
        from backend.analytics.pattern_engine import PatternEngine
        from backend.analytics.relationship_engine import RelationshipEngine

        self.pattern_engine = PatternEngine(self.journal_db)
        self.habit_engine = HabitEngine(self.journal_db)
        self.relationship_engine = RelationshipEngine(self.journal_db)
        self.insight_engine = InsightEngine(
            self.pattern_engine,
            habit_engine=self.habit_engine,
            relationship_engine=self.relationship_engine,
        )

        # Phase 6: rule-based crisis detector (local, runs first in pipeline).
        from backend.safety.crisis_detector import CrisisDetector

        self.crisis_detector = CrisisDetector()

        # Phase 9: per-message reflective-question generator (rule-based, no LLM).
        from backend.analytics.reflection_engine import ReflectionEngine

        self.reflection_engine = ReflectionEngine()

        # Phase 10: local PDF report generator (no API). Enhanced in Phase 16
        # with growth + prediction engines (wired after they are constructed).
        from backend.reports.report_generator import ReportGenerator

        self._ReportGenerator = ReportGenerator

        # Phase 12: user profile + orchestrator (deterministic, no LLM).
        from backend.orchestrator.orchestrator import Orchestrator
        from backend.profile.profile_manager import ProfileManager

        self.profile_manager = ProfileManager(
            self.journal_db,
            self.pattern_engine,
            self.habit_engine,
            self.relationship_engine,
        )
        self.orchestrator = Orchestrator()
        self.current_profile = self.profile_manager.load()

        # Phase 13: temporal + causal + proactive alerts (deterministic, no LLM).
        from backend.analytics.alert_engine import AlertEngine
        from backend.analytics.causal_engine import CausalEngine
        from backend.analytics.temporal_engine import TemporalEngine

        self.temporal_engine = TemporalEngine(self.journal_db)
        self.causal_engine = CausalEngine(self.journal_db)
        self.alert_engine = AlertEngine(self.journal_db, self.pattern_engine)

        # Phase 14: prediction + goal tracking (deterministic, no LLM).
        from backend.analytics.goal_engine import GoalEngine
        from backend.analytics.prediction_engine import PredictionEngine

        self.prediction_engine = PredictionEngine(self.journal_db)
        self.goal_engine = GoalEngine(self.journal_db)

        # Phase 15: memory replay + knowledge graph (deterministic, no LLM).
        from backend.memory.replay_engine import ReplayEngine

        self.replay_engine = ReplayEngine(self.vector_store, self.journal_db)

        from backend.graph.knowledge_graph import KnowledgeGraph

        self.knowledge_graph = KnowledgeGraph(self.journal_db)

        # Phase 16: timeline + growth tracking (deterministic, no LLM).
        from backend.analytics.growth_tracker import GrowthTracker
        from backend.analytics.timeline_engine import TimelineEngine

        self.timeline_engine = TimelineEngine(self.journal_db)
        self.growth_tracker = GrowthTracker(self.journal_db)

        # Phase 17: evaluation metrics + latency logging (local, developer-facing).
        self._latency_log_path = settings.latency_log_path
        from backend.evaluation.eval_engine import EvalEngine

        self.eval_engine = EvalEngine(
            self.vector_store, self.journal_db, settings.latency_log_path
        )

        # Now that growth + prediction engines exist, build the report generator.
        self.report_generator = self._ReportGenerator(
            self.pattern_engine,
            self.habit_engine,
            self.relationship_engine,
            self.insight_engine,
            growth_tracker=self.growth_tracker,
            prediction_engine=self.prediction_engine,
        )

    def _persist_journal_record(
        self, text: str, emotion: dict[str, Any], extracted: dict[str, Any] | None = None
    ) -> None:
        """Write a structured JournalRecord. Must never raise into the pipeline."""
        try:
            extracted = extracted or {}
            record = JournalRecord(
                id=self.vector_store._hash_entry(text, None),
                text=text,
                timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                emotion=emotion.get("emotion", "neutral"),
                emotion_confidence=float(emotion.get("confidence", 0.0)),
                entities_people=extracted.get("entities_people", []),
                entities_places=extracted.get("entities_places", []),
                entities_orgs=extracted.get("entities_orgs", []),
                keywords=extracted.get("keywords", []),
                topics=extracted.get("topics", []),
                habits=extracted.get("habits", []),
                person_relationship_types=extracted.get("person_relationship_types", {}),
                sentiment_compound=float(extracted.get("sentiment_compound", 0.0)),
                sentiment_valence=float(extracted.get("sentiment_valence", 0.0)),
            )
            self.journal_db.insert(record)
        except Exception as exc:
            logger.exception("Failed to persist journal record: %s", exc)

    def _safe_insights(self, lookback_days: int = 30) -> list[str]:
        """Generate insights for the prompt; never raise into the pipeline.

        Filters the 'not enough history' placeholder so the prompt section is
        omitted entirely when there is nothing meaningful to say.
        """
        try:
            insights = self.insight_engine.generate(lookback_days=lookback_days)
            return [s for s in insights if s and "Not enough journal history" not in s]
        except Exception as exc:
            logger.exception("Failed to generate insights: %s", exc)
            return []

    def _safe_reflection(self, text: str, replay: dict | None = None) -> list[str]:
        """Per-message reflective questions; never raise into the pipeline."""
        try:
            return self.reflection_engine.detect(
                text, replay=replay, context=self._reflection_context()
            )
        except Exception as exc:
            logger.exception("Failed to generate reflection prompts: %s", exc)
            return []

    def _reflection_context(self) -> dict[str, Any] | None:
        candidates: list[tuple[float, str, str]] = []
        try:
            summary = self.pattern_engine.analyze(lookback_days=30)
            for trig in summary.triggers:
                candidates.append((trig.confidence, "trigger", trig.topic))
        except Exception:
            pass
        try:
            for habit in self.habit_engine.analyze(lookback_days=30):
                candidates.append((habit.confidence, "habit", habit.habit))
        except Exception:
            pass
        try:
            for person in self.relationship_engine.analyze(lookback_days=30):
                candidates.append((person.confidence, "person", person.person))
        except Exception:
            pass
        if not candidates:
            return None
        confidence, kind, name = max(candidates, key=lambda c: c[0])
        if confidence <= 0:
            return None
        return {"kind": kind, "name": name, "confidence": confidence}

    def _store_memory_entry(
        self,
        *,
        text: str,
        tags: list[str] | None,
        emotion: dict[str, Any],
        extracted: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            return self.memory.store_entry(
                text=text,
                tags=tags,
                emotion_signal=emotion,
                topics=extracted.get("topics", []),
                person_relationship_types=extracted.get("person_relationship_types", {}),
            )
        except TypeError:
            return self.memory.store_entry(
                text=text,
                tags=tags,
                emotion_signal=emotion,
                topics=extracted.get("topics", []),
            )

    @staticmethod
    def _try(fn, default):
        try:
            return fn()
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("packet sub-step failed: %s", exc)
            return default

    def _assemble_packet(
        self,
        *,
        text: str,
        emotion: dict[str, Any],
        sentiment: float,
        insights: list[str],
        reflection_prompts: list[str],
        memory_replay: dict | None = None,
    ):
        """Gather already-cheap engine outputs into an IntelligencePacket.

        Engines added in later phases are invoked via getattr so this method
        works whether or not they have been wired yet, and degrades gracefully.
        """
        profile = self._try(self.profile_manager.update, self.current_profile)
        self.current_profile = profile

        summary = self._try(self.pattern_engine.analyze, None)
        triggers = summary.triggers if summary else []
        habits = self._try(self.habit_engine.analyze, [])
        relationships = self._try(self.relationship_engine.analyze, [])

        # Phase 13/14/15 engines (optional until wired).
        alerts = []
        if getattr(self, "alert_engine", None):
            alerts = self._try(self.alert_engine.check, [])
        temporal = []
        if getattr(self, "temporal_engine", None):
            temporal = self._try(self.temporal_engine.analyze, [])
        causal = []
        if getattr(self, "causal_engine", None):
            causal = self._try(self.causal_engine.analyze, [])
        predictions: dict[str, Any] = {}
        goals: list[Any] = []
        if getattr(self, "prediction_engine", None):
            predictions = {
                "sentiment_forecast": self._try(
                    self.prediction_engine.forecast_sentiment, None
                ),
                "burnout_risk": self._try(
                    self.prediction_engine.assess_burnout_risk, None
                ),
            }
        if getattr(self, "goal_engine", None):
            goals = self._try(self.goal_engine.analyze, [])

        return self.orchestrator.assemble(
            text=text,
            emotion_result=emotion,
            sentiment=sentiment,
            insights=insights,
            reflection_prompts=reflection_prompts,
            triggers=triggers,
            habits=habits,
            relationships=relationships,
            user_profile=profile,
            proactive_alerts=[a.message for a in alerts] if alerts else [],
            temporal_patterns=temporal,
            causal_links=causal,
            predictions={k: v for k, v in predictions.items() if v is not None},
            goals=goals,
            memory_replay=memory_replay,
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
        # Phase 6: crisis check runs FIRST. If flagged, we still run the full
        # pipeline (so memory/continuity is preserved) and prepend a calm,
        # non-diagnostic safety message to the final response.
        t0 = time.perf_counter()
        crisis = self.crisis_detector.check(text)
        log_stage("crisis", crisis)

        emotion = self.emotion_detector.detect(text)
        log_stage("emotion", {"input": text[:120], "emotion": emotion})

        # Phase 2: local NLP enrichment (entities/keywords/topics/sentiment).
        extracted = self.text_processor.extract(text)

        stored_entry = self._store_memory_entry(
            text=text,
            tags=tags,
            emotion=emotion,
            extracted=extracted,
        )
        # Phase 1+2: write structured record with enrichment fields.
        self._persist_journal_record(text=text, emotion=emotion, extracted=extracted)
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

        # Phase 12+: compute engine outputs once, update profile, assemble packet.
        # Phase 15: find replay first so reflection can personalize from it.
        replay_obj = None
        if getattr(self, "replay_engine", None):
            replay_obj = self._try(
                lambda: self.replay_engine.find_replay(
                    text, emotion.get("emotion", "neutral")
                ),
                None,
            )
        replay_dict = replay_obj.model_dump() if replay_obj is not None else None

        insights = self._safe_insights()
        reflection_prompts = self._safe_reflection(text, replay=replay_dict)
        sentiment = float(extracted.get("sentiment_compound", 0.0))
        packet = self._assemble_packet(
            text=text,
            emotion=emotion,
            sentiment=sentiment,
            insights=insights,
            reflection_prompts=reflection_prompts,
            memory_replay=replay_dict,
        )

        prompt = self.prompt_builder.build(
            user_text=text,
            current_emotion=emotion,
            retrieved_memories=merged,
            recent_history=chat_history or [],
            insights=insights,
            reflection_prompts=reflection_prompts,
            packet=packet,
        )
        log_stage("prompt", {"prompt_preview": prompt[:800]})
        response = self.llm.generate(prompt)

        # Prepend the safety message when crisis language was detected.
        if crisis.get("flagged"):
            from backend.safety.crisis_detector import CRISIS_MESSAGE

            response = f"{CRISIS_MESSAGE}\n\n{response}"

        self._log_latency((time.perf_counter() - t0) * 1000.0)

        return {
            "emotion": emotion,
            "stored_entry": stored_entry,
            "retrieved_memories": merged,
            "prompt": prompt,
            "response": response,
            "crisis": crisis,
            "packet": packet,
        }

    def _log_latency(self, elapsed_ms: float) -> None:
        """Append a latency sample; never let logging failure crash the pipeline."""
        try:
            path = getattr(self, "_latency_log_path", None)
            if not path:
                return
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            line = json.dumps(
                {
                    "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    "elapsed_ms": round(elapsed_ms, 2),
                }
            )
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as exc:
            logger.exception("latency logging failed: %s", exc)
