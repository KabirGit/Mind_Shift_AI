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

    from backend.storage.db import JournalDB

    journal_db = JournalDB(settings.sqlite_path)

    from backend.nlp.text_processor import TextProcessor

    text_processor = TextProcessor()

    return {
        "vector_store": vector_store,
        "emotion_detector": emotion_detector,
        "retriever": retriever,
        "llm_client": llm_client,
        "prompt_builder": prompt_builder,
        "journal_db": journal_db,
        "text_processor": text_processor,
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


def _format_secondary_emotions(emotion: dict) -> str:
    """Render the next-strongest emotions (beyond the top one) as a caption."""
    spread = emotion.get("all_emotions") or []
    others = [
        e for e in spread if e.get("emotion") != emotion.get("emotion")
    ][:3]
    if not others:
        return ""
    parts = ", ".join(f"{e['emotion']} ({e['score']:.2f})" for e in others)
    return f"_Also sensing: {parts}_\n\n"


def _render_memory_replay(packet) -> None:
    """Show a subtle Memory Replay expander if the packet carries one."""
    if packet is None:
        return
    replay = getattr(packet, "memory_replay", None)
    if not replay:
        return
    days_ago = replay.get("days_ago", "?")
    emotion = replay.get("similar_entry_emotion", "neutral")
    nxt = replay.get("what_happened_next")
    hint = replay.get("recovery_hint", "")
    with st.expander("🔁 Memory Replay"):
        st.markdown(
            f"**{days_ago} days ago** you wrote something similar "
            f"(feeling _{emotion}_)."
        )
        st.caption(f'"{str(replay.get("similar_entry_text", ""))[:200]}"')
        if nxt:
            st.markdown(f"**What happened next:** {str(nxt)[:200]}")
        st.markdown(f"_{hint}_")


def _save_uploaded_file(uploaded_file) -> str:
    save_directory = Path("data") / "text_files"
    save_directory.mkdir(parents=True, exist_ok=True)
    save_path = save_directory / uploaded_file.name
    save_path.write_bytes(uploaded_file.getbuffer())
    return str(save_path)


_LOOKBACK_OPTIONS = {
    "Last 7 days": 7,
    "Last 30 days": 30,
    "All time": 36500,
}


def _emotion_over_time_df(records):
    """Build a date x emotion count DataFrame for charting. Pure data, no st."""
    import pandas as pd

    rows = []
    for r in records:
        ts = (r.timestamp or "")[:10]  # YYYY-MM-DD
        if ts:
            rows.append({"date": ts, "emotion": r.emotion or "neutral"})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    pivot = (
        df.groupby(["date", "emotion"]).size().unstack(fill_value=0).sort_index()
    )
    return pivot


def _render_dashboard(service: "RAGService") -> None:
    db = service.journal_db
    records = db.get_all()

    if not records:
        st.info("Write a few journal entries to see your patterns here.")
        return

    # Proactive alerts at the very top of the dashboard.
    try:
        alerts = service.alert_engine.check()
    except Exception:
        alerts = []
    for alert in alerts:
        if alert.severity == "watch":
            st.warning(f"⚠ {alert.message}")
        else:
            st.info(f"ℹ {alert.message}")

    label = st.selectbox(
        "Time range", list(_LOOKBACK_OPTIONS.keys()), index=1, key="dash_range"
    )
    lookback = _LOOKBACK_OPTIONS[label]

    summary = service.pattern_engine.analyze(lookback_days=lookback)
    insights = service.insight_engine.generate(lookback_days=lookback)

    st.subheader("Emotion frequency over time")
    emo_df = _emotion_over_time_df(records)
    if emo_df.empty:
        st.info("Not enough dated entries yet.")
    else:
        st.area_chart(emo_df)

    st.subheader("Top topics")
    if summary.recurring_topics:
        import pandas as pd

        top = sorted(
            summary.recurring_topics.items(), key=lambda kv: kv[1], reverse=True
        )[:5]
        topic_df = pd.DataFrame(top, columns=["topic", "count"]).set_index("topic")
        st.bar_chart(topic_df)
    else:
        st.info("No topics detected in this range yet.")

    st.subheader("Triggers")
    if summary.triggers:
        trig_rows = [t.model_dump() for t in summary.triggers]
        st.dataframe(trig_rows, use_container_width=True)
    else:
        st.info("No recurring triggers detected in this range yet.")

    st.subheader("Habits & Mood")
    habit_corrs = service.habit_engine.analyze(lookback_days=lookback)
    if habit_corrs:
        st.dataframe([c.model_dump() for c in habit_corrs], use_container_width=True)
    else:
        st.info("No habit correlations detected in this range yet.")

    st.subheader("People")
    profiles = service.relationship_engine.analyze(lookback_days=lookback)
    if profiles:
        st.dataframe([p.model_dump() for p in profiles], use_container_width=True)
    else:
        st.info("No recurring people detected in this range yet.")

    st.subheader("Goals")
    goals = service.goal_engine.analyze(lookback_days=max(lookback, 90))
    if goals:
        for g in goals:
            st.markdown(f"**{g.goal_keyword.replace('_', ' ').title()}**")
            st.progress(g.estimated_progress)
            st.caption(
                f"{g.explanation} (Based on sentiment around related entries, "
                "not actual completion.)"
            )
    else:
        st.info("No goal-related entries detected yet.")

    st.subheader("Stress Pattern")
    try:
        risk = service.prediction_engine.assess_burnout_risk()
        forecast = service.prediction_engine.forecast_sentiment()
        st.markdown(f"**{risk.risk_level.upper()} ({risk.score:.0%})**")
        st.markdown(f"_{risk.explanation}_")
        st.caption(
            f"Mood forecast: {forecast.direction} over the next "
            f"{forecast.horizon_days} days. {forecast.explanation}"
        )
    except Exception:
        st.info("Not enough data to assess stress patterns yet.")

    st.subheader("Insights")
    for line in insights:
        st.markdown(f"- {line}")
    st.caption(
        "These reflections are generated from your own entries and are not "
        "medical or psychological advice."
    )

    st.subheader("Knowledge Graph")
    gq = st.text_input(
        "Search your personal knowledge graph (e.g. 'career', 'mom', 'exercise')",
        key="kg_query",
    )
    if gq.strip():
        graph = service.knowledge_graph.build(lookback_days=max(lookback, 90))
        node = gq.strip()
        result = service.knowledge_graph.query(graph, node)
        if not result["neighbors"]:
            st.info(f"No connections found for '{node}'.")
        else:
            st.markdown(service.knowledge_graph.summarize_node(graph, node))
            st.dataframe(result["edge_data"], use_container_width=True)

    st.subheader("My Journey")
    timeline = service.timeline_engine.build(lookback_days=max(lookback, 90))
    if timeline:
        _emoji = {"positive_peak": "🟢", "negative_peak": "🔴", "normal": "⚪"}
        shown = timeline[:15]
        for ev in shown:
            date = (ev.timestamp or "")[:10]
            st.markdown(f"{_emoji.get(ev.event_type, '⚪')} **{date}** — {ev.title}")
        if len(timeline) > 15:
            st.caption(f"...and {len(timeline) - 15} more")
    else:
        st.info("Your journey timeline will appear as you journal.")

    st.subheader("Growth Over Time")
    snapshots = service.growth_tracker.compute_snapshots()
    if snapshots:
        import pandas as pd

        gdf = pd.DataFrame(
            [{"period": s.period_label, "avg_sentiment": s.avg_sentiment} for s in snapshots]
        ).set_index("period")
        st.line_chart(gdf)
        narrative = service.growth_tracker.narrative()
        deltas = service.growth_tracker.compute_growth_deltas()
        if deltas and deltas[-1]["sentiment_delta"] > 0.1:
            st.success(narrative)
        else:
            st.info(narrative)
    else:
        st.info("Growth trends appear once you have entries across time.")

    st.subheader("Weekly Report")
    if st.button("Generate Report", key="gen_report"):
        report_bytes = service.report_generator.generate(lookback_days=lookback)
        st.download_button(
            "Download Weekly Report (PDF)",
            data=report_bytes,
            file_name="weekly_report.pdf",
            mime="application/pdf",
            key="dl_report",
        )

    with st.expander("⚙ System Diagnostics", expanded=False):
        try:
            prec = service.eval_engine.retrieval_precision_at_k(k=3)
            emo = service.eval_engine.emotion_confidence_stats()
            lat = service.eval_engine.latency_summary()
            st.markdown(f"**Retrieval Precision@3:** {prec['precision_at_k']:.1%}")
            st.caption(prec["note"])
            st.markdown(
                f"**Emotion confidence:** mean {emo['mean_confidence']:.2f} "
                f"(min {emo['min']:.2f}, max {emo['max']:.2f}); "
                f"low-confidence ratio {emo['low_confidence_ratio']:.1%}"
            )
            st.markdown(
                f"**Pipeline latency:** avg {lat['avg_ms']:.0f} ms, "
                f"p95 {lat['p95_ms']:.0f} ms ({lat['sample_count']} samples)"
            )
            st.caption(
                "These metrics help evaluate system quality. Higher precision and "
                "confidence = better retrieval and emotion detection."
            )
        except Exception:
            st.info("Diagnostics unavailable.")


def _render_chat(service: "RAGService") -> None:
    # Show watch-level proactive alerts once per session as a subtle banner.
    if not st.session_state.get("alerts_shown"):
        try:
            watch = [a for a in service.alert_engine.check() if a.severity == "watch"]
        except Exception:
            watch = []
        for a in watch:
            st.info(a.message)
        st.session_state["alerts_shown"] = True

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
                f"{_format_secondary_emotions(emotion)}"
                f"{response}"
            )
            _append_chat("assistant", assistant_text)

            with st.chat_message("assistant"):
                st.markdown(assistant_text)
                _render_memory_replay(output.get("packet"))
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
            journal_db=shared["journal_db"],
            text_processor=shared["text_processor"],
        )
    service: RAGService = st.session_state["service"]

    chat_tab, dashboard_tab = st.tabs(["Chat", "Insights Dashboard"])
    with chat_tab:
        _render_chat(service)
    with dashboard_tab:
        _render_dashboard(service)
