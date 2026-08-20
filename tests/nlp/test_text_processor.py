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
