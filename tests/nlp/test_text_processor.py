from __future__ import annotations

import socket

import pytest

from backend.nlp.text_processor import TextProcessor


@pytest.fixture(scope="module")
def processor():
    return TextProcessor()


def test_entities_extracted(processor):
    out = processor.extract("I met Barack Obama in Paris last week.")
    assert "Barack Obama" in out["entities_people"]
    assert "Paris" in out["entities_places"]


def test_topic_career(processor):
    out = processor.extract("I had a job interview and I'm hoping for a promotion.")
    assert "career" in out["topics"]


def test_empty_string_defaults(processor):
    out = processor.extract("")
    assert out["entities_people"] == []
    assert out["topics"] == []
    assert out["keywords"] == []
    assert out["sentiment_compound"] == 0.0


def test_long_text_no_crash(processor):
    long_text = ("I feel happy and grateful about work today. " * 200)[:6000]
    out = processor.extract(long_text)
    assert isinstance(out["keywords"], list)
    assert isinstance(out["sentiment_compound"], float)


def test_no_network_calls(processor, monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("network call attempted in TextProcessor.extract")

    monkeypatch.setattr(socket.socket, "connect", _boom)
    out = processor.extract("My friend Alice helped me study for the exam.")
    assert "relationship" in out["topics"] or "education" in out["topics"]


def test_relationship_type_classifier_examples():
    examples = [
        ("My mom Sarah called me today.", "Sarah", "family"),
        ("My sister Priya helped me study.", "Priya", "family"),
        ("My wife Nina made dinner.", "Nina", "partner"),
        ("My friend Alice checked in.", "Alice", "friend"),
        ("My coworker Omar reviewed the project.", "Omar", "colleague"),
        ("My boss Morgan was supportive.", "Morgan", "colleague"),
        ("I talked with Jordan after class.", "Jordan", "unknown"),
    ]
    for text, person, expected in examples:
        start = text.index(person)
        out = TextProcessor._relationship_types(
            text, [person], [(person, start, start + len(person))]
        )
        assert out[person] == expected


def test_relationship_cue_extraction_adds_people_when_ner_misses():
    out = TextProcessor().extract(
        "My wife Meera found a therapist named Dr. Anjali, and my old friend Sameer called."
    )

    assert "Meera" in out["entities_people"]
    assert "Anjali" in out["entities_people"]
    assert "Sameer" in out["entities_people"]
    assert out["person_relationship_types"]["Meera"] == "partner"
    assert out["person_relationship_types"]["Anjali"] == "other"
    assert out["person_relationship_types"]["Sameer"] == "friend"


def test_relationship_aliases_are_treated_as_people():
    out = TextProcessor().extract("Ma asked if Papa and my younger brother Rohan had eaten.")

    assert out["person_relationship_types"]["Ma"] == "family"
    assert out["person_relationship_types"]["Papa"] == "family"
    assert out["person_relationship_types"]["Rohan"] == "family"


def test_topics_are_ranked_by_evidence_not_static_bucket_order():
    processor = TextProcessor()

    health = processor.extract(
        "My father's follow-up appointment was today. The asthma medication change "
        "worked, the medical test looked good, and the doctor sounded relieved. "
        "I took one quick work call after dinner."
    )
    relationship = processor.extract(
        "Long honest conversation with Elena on the balcony. We talked about how "
        "distant and lonely the marriage had felt during my work stress, and how "
        "much the relationship needed real attention."
    )

    assert health["topics"][0] == "health"
    assert relationship["topics"][0] == "relationship"


def test_person_action_cues_recover_names_spacy_can_miss():
    out = TextProcessor().extract(
        "Marcus texted after work. Run with Elena felt steady, and Omar called later."
    )

    assert "Marcus" in out["entities_people"]
    assert "Elena" in out["entities_people"]
    assert "Omar" in out["entities_people"]


def test_friendship_context_types_people_without_name_specific_rules():
    out = TextProcessor().extract(
        "Nina mentioned today that she has known me since college. "
        "Later, Carlos and I grabbed coffee after work."
    )

    assert "Nina" in out["entities_people"]
    assert "Carlos" in out["entities_people"]
    assert out["person_relationship_types"]["Nina"] == "friend"
