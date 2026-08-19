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


def _inject_app_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 1.5rem;
            max-width: 1180px;
        }
        .insight-card {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 1rem 1.05rem;
            background: #ffffff;
            min-height: 138px;
        }
        .insight-card h3 {
            font-size: 0.9rem;
            margin: 0 0 .35rem 0;
            color: #475569;
            font-weight: 700;
        }
        .insight-card .big {
            font-size: 1.65rem;
            line-height: 1.2;
            font-weight: 760;
            color: #111827;
            margin-bottom: .3rem;
        }
        .insight-card p {
            margin: 0;
            color: #475569;
            font-size: .92rem;
            line-height: 1.45;
        }
        .action-box {
            border-left: 4px solid #2563eb;
            background: #eff6ff;
            padding: .85rem 1rem;
            border-radius: 6px;
            margin-bottom: .65rem;
        }
        .action-box strong {
            color: #1e3a8a;
        }
        .quiet-note {
            color: #64748b;
            font-size: .9rem;
        }
        div[data-testid="stVerticalBlock"]:has(.chat-history-marker) {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: .75rem .85rem;
            background: #f8fafc;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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


def _mood_label(score: float) -> str:
    if score >= 68:
        return "Mostly steady"
    if score >= 52:
        return "Mixed but manageable"
    if score >= 38:
        return "Under pressure"
    return "Needs attention"


def _sentiment_score(records) -> int:
    if not records:
        return 50
    avg = sum(r.sentiment_compound for r in records) / len(records)
    return int(round((avg + 1) * 50))


def _window_records(records, lookback_days: int):
    from backend.analytics._stats_utils import filter_window

    return filter_window(records, lookback_days)


def _momentum(records) -> tuple[str, str]:
    if len(records) < 4:
        return "Still learning", "Add a few more entries before reading a trend."
    ordered = sorted(records, key=lambda r: r.timestamp or "")
    mid = len(ordered) // 2
    first = ordered[:mid]
    second = ordered[mid:]
    first_avg = sum(r.sentiment_compound for r in first) / len(first)
    second_avg = sum(r.sentiment_compound for r in second) / len(second)
    delta = second_avg - first_avg
    if delta > 0.08:
        return "Improving", "Your recent entries are more positive than earlier ones."
    if delta < -0.08:
        return "Dropping", "Recent entries are heavier than the earlier part of this period."
    return "Stable", "Your emotional tone has stayed fairly consistent."


def _balance_text(records) -> tuple[str, str]:
    if not records:
        return "No entries yet", "Write a few entries to see your balance."
    positive = sum(1 for r in records if r.sentiment_compound > 0.1)
    difficult = sum(1 for r in records if r.sentiment_compound < -0.1)
    neutral = max(0, len(records) - positive - difficult)
    if difficult > positive:
        headline = "More heavy days than light ones"
    elif positive > difficult:
        headline = "More light days than heavy ones"
    else:
        headline = "Evenly mixed"
    detail = f"{positive} supportive, {neutral} neutral, {difficult} difficult entries."
    return headline, detail


def _plain_topic(topic: str) -> str:
    return str(topic).replace("_", " ").title()


def _topic_guidance(topic: str) -> str:
    topic_l = str(topic).lower()
    if topic_l == "career":
        return "Turn the pressure into one next action: one application, one message, or one focused work block."
    if topic_l == "money":
        return "Make the worry concrete: list the next bill, amount, and one step you control today."
    if topic_l == "health":
        return "Protect the basics first: sleep, food, movement, and one low-friction recovery habit."
    if topic_l == "relationship":
        return "Name the person and the need clearly; decide whether this needs a conversation or a boundary."
    if topic_l == "education":
        return "Convert study stress into a small plan: what to revise, when, and how you will check progress."
    return "Write one sentence about what you can control, then choose a small next step."


def _top_pressure(summary) -> tuple[str, str]:
    if summary.triggers:
        trigger = sorted(
            summary.triggers,
            key=lambda t: (t.avg_sentiment, -t.frequency),
        )[0]
        topic = _plain_topic(trigger.topic)
        return topic, _topic_guidance(trigger.topic)
    if summary.recurring_topics:
        topic = max(summary.recurring_topics.items(), key=lambda kv: kv[1])[0]
        return _plain_topic(topic), _topic_guidance(topic)
    return "No clear pressure source", "Keep journaling naturally; a pattern will appear after more entries."


def _best_stabilizer(habits) -> tuple[str, str]:
    positives = [h for h in habits if h.delta > 0.05]
    if not positives:
        return "No reliable stabilizer yet", "Mention habits as you journal so the app can learn what helps."
    best = max(positives, key=lambda h: h.delta)
    return (
        _plain_topic(best.habit),
        "Your mood tends to be higher on days this appears. Try scheduling it before stressful blocks.",
    )


def _relationship_guidance(profiles) -> tuple[str, str]:
    if not profiles:
        return "No relationship pattern yet", "When people show up repeatedly, this will summarize support or strain."
    strongest = min(profiles, key=lambda p: p.avg_sentiment)
    if strongest.avg_sentiment < -0.1:
        return (
            strongest.person,
            "This connection is often emotionally heavy. Consider what boundary, clarity, or support you need.",
        )
    strongest = max(profiles, key=lambda p: p.avg_sentiment)
    return (
        strongest.person,
        "This connection appears supportive. It may be worth leaning into it when the week feels heavy.",
    )


def _build_next_steps(summary, habits, profiles, risk, forecast, goals) -> list[tuple[str, str]]:
    pressure, pressure_action = _top_pressure(summary)
    stabilizer, stabilizer_action = _best_stabilizer(habits)
    person, person_action = _relationship_guidance(profiles)
    steps = [(f"Work with {pressure}", pressure_action)]

    if "No reliable" not in stabilizer:
        steps.append((f"Repeat {stabilizer}", stabilizer_action))
    else:
        steps.append(("Create one recovery anchor", stabilizer_action))

    if risk.risk_level in {"medium", "high"}:
        steps.append(
            (
                "Lower the load this week",
                "Your recent pattern suggests pressure is building. Reduce one optional task or add one recovery window.",
            )
        )
    elif forecast.direction == "declining":
        steps.append(
            (
                "Catch the dip early",
                "Recent mood is trending down. Plan one small action today instead of waiting for it to become urgent.",
            )
        )
    elif goals:
        goal = goals[0]
        steps.append(
            (
                f"Move {goal.goal_keyword.replace('_', ' ')} forward",
                "Choose one visible next step so progress feels concrete, not just mental.",
            )
        )
    else:
        steps.append((f"Use {person} wisely", person_action))

    return steps[:3]


def _render_insight_card(title: str, big: str, detail: str) -> None:
    st.markdown(
        f"""
        <div class="insight-card">
            <h3>{title}</h3>
            <div class="big">{big}</div>
            <p>{detail}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_action(title: str, detail: str) -> None:
    st.markdown(
        f"""
        <div class="action-box">
            <strong>{title}</strong><br>
            {detail}
        </div>
        """,
        unsafe_allow_html=True,
    )


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


def _render_dashboard_v2(service: "RAGService") -> None:
    db = service.journal_db
    records = db.get_all()

    if not records:
        st.info("Write a few journal entries to see your personal summary here.")
        return

    label = st.selectbox(
        "Time range", list(_LOOKBACK_OPTIONS.keys()), index=1, key="dash_range_v2"
    )
    lookback = _LOOKBACK_OPTIONS[label]
    window_records = _window_records(records, lookback)

    summary = service.pattern_engine.analyze(lookback_days=lookback)
    habit_corrs = service.habit_engine.analyze(lookback_days=lookback)
    profiles = service.relationship_engine.analyze(lookback_days=lookback)
    goals = service.goal_engine.analyze(lookback_days=max(lookback, 90))

    try:
        risk = service.prediction_engine.assess_burnout_risk()
        forecast = service.prediction_engine.forecast_sentiment()
    except Exception:
        from backend.analytics.prediction_engine import BurnoutRisk, SentimentForecast

        risk = BurnoutRisk(explanation="Not enough data yet.")
        forecast = SentimentForecast(explanation="Not enough data yet.")

    mood_score = _sentiment_score(window_records)
    balance_title, balance_detail = _balance_text(window_records)
    momentum_title, momentum_detail = _momentum(window_records)
    pressure_title, pressure_detail = _top_pressure(summary)
    stabilizer_title, stabilizer_detail = _best_stabilizer(habit_corrs)
    person_title, person_detail = _relationship_guidance(profiles)
    next_steps = _build_next_steps(
        summary, habit_corrs, profiles, risk, forecast, goals
    )

    st.subheader("Your Journal Summary")
    st.caption(
        "A simple read of your recent entries: what is happening, what is driving it, "
        "and what to try next. This is reflection support, not medical advice."
    )

    try:
        alerts = service.alert_engine.check()
    except Exception:
        alerts = []
    for alert in alerts[:2]:
        if alert.severity == "watch":
            st.warning(alert.message)
        else:
            st.info(alert.message)

    c1, c2, c3 = st.columns(3)
    with c1:
        _render_insight_card(
            "Overall state",
            f"{mood_score}/100",
            f"{_mood_label(mood_score)}. {balance_detail}",
        )
    with c2:
        _render_insight_card("Direction", momentum_title, momentum_detail)
    with c3:
        factors = ", ".join(risk.contributing_factors) or "No strong risk factors yet."
        _render_insight_card(
            "Pressure load",
            risk.risk_level.title(),
            f"{risk.score:.0%} signal. {factors}",
        )

    st.subheader("What This Means")
    c1, c2, c3 = st.columns(3)
    with c1:
        _render_insight_card("Main pressure", pressure_title, pressure_detail)
    with c2:
        _render_insight_card("Best stabilizer", stabilizer_title, stabilizer_detail)
    with c3:
        _render_insight_card("People pattern", person_title, person_detail)

    st.subheader("Try This Next")
    for title, detail in next_steps:
        _render_action(title, detail)

    st.subheader("Progress Signals")
    p1, p2 = st.columns([1, 1])
    with p1:
        st.markdown("**Mood forecast**")
        st.progress(min(1.0, max(0.0, (forecast.predicted_sentiment + 1.0) / 2.0)))
        st.caption(
            f"{forecast.direction.title()} over the next {forecast.horizon_days} days. "
            f"Confidence {forecast.confidence:.0%}."
        )
    with p2:
        st.markdown("**Goal momentum**")
        if goals:
            for goal in goals[:3]:
                st.markdown(f"**{goal.goal_keyword.replace('_', ' ').title()}**")
                st.progress(goal.estimated_progress)
                st.caption(goal.explanation)
        else:
            st.caption("No goal has enough repeated evidence yet.")

    with st.expander("Evidence behind these conclusions", expanded=False):
        emo_df = _emotion_over_time_df(window_records)
        if not emo_df.empty:
            st.markdown("**Emotional rhythm**")
            st.area_chart(emo_df)

        if summary.triggers:
            st.markdown("**Pressure sources**")
            trig_rows = [
                {
                    "area": _plain_topic(t.topic),
                    "pattern": t.explanation,
                    "confidence": f"{t.confidence:.0%}",
                }
                for t in summary.triggers
            ]
            st.dataframe(trig_rows, use_container_width=True, hide_index=True)

        if habit_corrs:
            st.markdown("**Habits that seem to move mood**")
            habit_rows = [
                {
                    "habit": _plain_topic(c.habit),
                    "effect": (
                        "helps"
                        if c.delta > 0.05
                        else "hurts"
                        if c.delta < -0.05
                        else "unclear"
                    ),
                    "summary": c.explanation,
                    "confidence": f"{c.confidence:.0%}",
                }
                for c in habit_corrs
            ]
            st.dataframe(habit_rows, use_container_width=True, hide_index=True)

        if profiles:
            st.markdown("**Relationship signals**")
            profile_rows = [
                {
                    "person": p.person,
                    "tone": (
                        "supportive"
                        if p.avg_sentiment > 0.1
                        else "heavy"
                        if p.avg_sentiment < -0.1
                        else "mixed"
                    ),
                    "trend": p.trend,
                    "summary": p.explanation,
                }
                for p in profiles
            ]
            st.dataframe(profile_rows, use_container_width=True, hide_index=True)

    with st.expander("Explore a topic or person", expanded=False):
        st.caption("Use this when you want to inspect one specific theme.")
        gq = st.text_input(
            "Search your personal knowledge graph",
            placeholder="career, money, a person's name...",
            key="kg_query_v2",
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

    with st.expander("Journey timeline", expanded=False):
        timeline = service.timeline_engine.build(lookback_days=max(lookback, 90))
        if timeline:
            shown = timeline[:12]
            for ev in shown:
                date = (ev.timestamp or "")[:10]
                tone = (
                    "High point"
                    if ev.event_type == "positive_peak"
                    else "Low point"
                    if ev.event_type == "negative_peak"
                    else "Entry"
                )
                st.markdown(f"**{date} - {tone}:** {ev.title}")
                st.caption(ev.description)
            if len(timeline) > 12:
                st.caption(f"...and {len(timeline) - 12} more")
        else:
            st.info("Your journey timeline will appear as you journal.")

    with st.expander("Growth over time", expanded=False):
        snapshots = service.growth_tracker.compute_snapshots()
        if snapshots:
            import pandas as pd

            gdf = pd.DataFrame(
                [
                    {"period": s.period_label, "avg_sentiment": s.avg_sentiment}
                    for s in snapshots
                ]
            ).set_index("period")
            st.line_chart(gdf)
            st.info(service.growth_tracker.narrative())
        else:
            st.info("Growth trends appear once you have entries across time.")

    st.subheader("Weekly Report")
    if st.button("Generate Report", key="gen_report_v2"):
        report_bytes = service.report_generator.generate(lookback_days=lookback)
        st.download_button(
            "Download Weekly Report (PDF)",
            data=report_bytes,
            file_name="weekly_report.pdf",
            mime="application/pdf",
            key="dl_report_v2",
        )

    with st.expander("System Diagnostics", expanded=False):
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


def _render_chat_v2(service: "RAGService") -> None:
    if not st.session_state.get("alerts_shown"):
        try:
            watch = [a for a in service.alert_engine.check() if a.severity == "watch"]
        except Exception:
            watch = []
        for alert in watch:
            st.info(alert.message)
        st.session_state["alerts_shown"] = True

    top_left, top_right = st.columns([5, 1])
    with top_left:
        st.caption("Write naturally. The chat remembers this session; scroll the history without moving the composer.")
    with top_right:
        if st.button("Reset", use_container_width=True):
            st.session_state["chat_history"] = []
            settings = get_settings()
            st.session_state["stm_entries"] = deque(maxlen=settings.session_stm_size)
            st.session_state.pop("service", None)
            st.session_state.pop("last_pipeline_debug", None)
            st.rerun()

    with st.expander("Chat options", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.session_state["show_retrieved"] = st.checkbox(
                "Show retrieved memories",
                value=st.session_state["show_retrieved"],
            )
        with c2:
            st.session_state["debug_mode"] = st.checkbox(
                "Debug mode",
                value=st.session_state["debug_mode"],
            )
        uploaded_file = st.file_uploader("Attach a journal file")
        if uploaded_file is not None:
            try:
                save_path = _save_uploaded_file(uploaded_file)
                st.success(f"File saved at: {save_path}")
            except Exception as exc:
                st.error(f"Failed to save file: {exc}")

    with st.container(height=560):
        st.markdown('<span class="chat-history-marker"></span>', unsafe_allow_html=True)
        if not st.session_state["chat_history"]:
            st.info("No entries in this session yet. Start with what happened today.")
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    debug_payload = st.session_state.get("last_pipeline_debug")
    if debug_payload and (
        st.session_state["show_retrieved"] or st.session_state["debug_mode"]
    ):
        with st.expander("Latest pipeline details", expanded=False):
            if st.session_state["show_retrieved"]:
                st.caption("Retrieved memories used")
                st.json(debug_payload.get("retrieved", []))
            if st.session_state["debug_mode"]:
                st.caption("Detected emotion")
                st.json(debug_payload.get("emotion", {}))
                st.caption("Constructed prompt")
                st.code(debug_payload.get("prompt", ""))

    user_text = st.chat_input("Write your current journal entry...")
    if not user_text:
        return

    _append_chat("user", user_text)
    try:
        with st.spinner("Reflecting on your entry..."):
            output = service.run_pipeline(
                text=user_text,
                chat_history=st.session_state["chat_history"],
                top_k=3,
            )
        emotion = output["emotion"]
        response = output["response"]
        assistant_text = (
            f"Detected mood: **{emotion['emotion']}** ({emotion['confidence']:.2f})\n\n"
            f"{_format_secondary_emotions(emotion)}"
            f"{response}"
        )
        _append_chat("assistant", assistant_text)
        st.session_state["last_pipeline_debug"] = {
            "emotion": emotion,
            "retrieved": output.get("retrieved_memories", []),
            "prompt": output.get("prompt", ""),
        }
    except Exception as exc:
        _append_chat("assistant", f"I could not process that entry yet: {exc}")
    st.rerun()


if __name__ == "__main__":
    setup_logging(logging.INFO)
    _init_session_state()
    _inject_app_css()
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
        _render_chat_v2(service)
    with dashboard_tab:
        _render_dashboard_v2(service)
