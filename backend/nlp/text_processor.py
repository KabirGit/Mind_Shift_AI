from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Fixed keyword buckets for lightweight topic matching (deterministic, local).
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "career": [
        "job", "work", "career", "boss", "manager", "office", "promotion",
        "interview", "colleague", "project", "deadline", "meeting", "salary",
    ],
    "health": [
        "health", "sick", "ill", "doctor", "hospital", "sleep", "tired",
        "exercise", "gym", "diet", "pain", "anxiety", "stress", "therapy",
    ],
    "relationship": [
        "friend", "family", "partner", "wife", "husband", "girlfriend",
        "boyfriend", "mother", "father", "mom", "dad", "love", "relationship",
        "breakup", "marriage", "date",
    ],
    "money": [
        "money", "rent", "bills", "debt", "loan", "savings", "budget",
        "expensive", "afford", "finance", "financial", "pay", "cost",
    ],
    "education": [
        "school", "college", "university", "exam", "study", "studying",
        "class", "course", "degree", "homework", "assignment", "grade",
        "teacher", "professor", "student",
    ],
}

_EMPTY: dict[str, Any] = {
    "entities_people": [],
    "entities_places": [],
    "entities_orgs": [],
    "keywords": [],
    "topics": [],
    "habits": [],
    "sentiment_compound": 0.0,
    "sentiment_valence": 0.0,
}


HABIT_KEYWORDS: dict[str, list[str]] = {
    "exercise": ["gym", "workout", "run", "running", "exercise"],
    "sleep": ["sleep", "slept", "nap", "insomnia"],
    "reading": ["read", "reading", "book"],
    "meditation": ["meditate", "meditation", "mindfulness"],
    "social_media": ["instagram", "twitter", "tiktok", "scrolling"],
    "coffee": ["coffee", "caffeine"],
    "cooking": ["cook", "cooking", "recipe"],
    "coding": ["code", "coding", "programming"],
}


class TextProcessor:
    """Local-only text enrichment: spaCy NER + noun-chunk keywords + topic
    bucketing + VADER sentiment. Lazy-loads heavy resources on first use.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:
        self.spacy_model = spacy_model
        self._nlp = None
        self._vader = None

    def _get_nlp(self):
        if self._nlp is None:
            import spacy

            self._nlp = spacy.load(self.spacy_model)
        return self._nlp

    def _get_vader(self):
        if self._vader is None:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            self._vader = SentimentIntensityAnalyzer()
        return self._vader

    def extract(self, text: str) -> dict[str, Any]:
        if not text or not text.strip():
            return dict(_EMPTY)

        try:
            nlp = self._get_nlp()
            doc = nlp(text)

            people, places, orgs = [], [], []
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    people.append(ent.text)
                elif ent.label_ in {"GPE", "LOC"}:
                    places.append(ent.text)
                elif ent.label_ == "ORG":
                    orgs.append(ent.text)

            # Keywords: deduped lowercased noun chunks, max 10.
            keywords: list[str] = []
            seen = set()
            for chunk in doc.noun_chunks:
                kw = chunk.text.strip().lower()
                if kw and kw not in seen:
                    seen.add(kw)
                    keywords.append(kw)
                if len(keywords) >= 10:
                    break

            lowered = text.lower()
            kw_blob = " ".join(keywords)
            topics: list[str] = []
            for topic, bucket in TOPIC_KEYWORDS.items():
                if any(word in lowered or word in kw_blob for word in bucket):
                    topics.append(topic)

            habits: list[str] = []
            for habit, bucket in HABIT_KEYWORDS.items():
                if any(word in lowered or word in kw_blob for word in bucket):
                    habits.append(habit)

            vader = self._get_vader()
            compound = float(vader.polarity_scores(text).get("compound", 0.0))

            return {
                "entities_people": self._dedupe(people),
                "entities_places": self._dedupe(places),
                "entities_orgs": self._dedupe(orgs),
                "keywords": keywords,
                "topics": topics,
                "habits": habits,
                "sentiment_compound": compound,
                "sentiment_valence": compound,
            }
        except Exception as exc:
            logger.exception("TextProcessor.extract failed, using defaults: %s", exc)
            return dict(_EMPTY)

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        out: list[str] = []
        seen = set()
        for item in items:
            key = item.strip()
            low = key.lower()
            if key and low not in seen:
                seen.add(low)
                out.append(key)
        return out
