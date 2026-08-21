from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Fixed keyword buckets for lightweight topic matching (deterministic, local).
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "career": [
        "job", "work", "career", "boss", "manager", "office", "promotion",
        "interview", "colleague", "project", "deadline", "meeting", "salary",
        "leadership", "slack", "sprint", "migration", "code", "coding",
    ],
    "health": [
        "health", "sick", "ill", "doctor", "hospital", "sleep", "tired",
        "exercise", "gym", "diet", "pain", "anxiety", "stress", "therapy",
        "cardiology", "checkup", "check-up", "follow-up", "appointment",
        "medication", "dosage", "medical", "test", "chest", "headache",
        "fatigue", "run", "running", "meditation",
    ],
    "relationship": [
        "friend", "family", "partner", "wife", "husband", "girlfriend",
        "boyfriend", "mother", "father", "mom", "dad", "love", "relationship",
        "breakup", "marriage", "date", "conversation", "talk", "talked",
        "apologize", "distant", "lonely", "support", "together", "cousin",
        "brother", "sister", "hand", "hurt", "worried",
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
    "person_relationship_types": {},
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


RELATIONSHIP_TYPE_KEYWORDS: dict[str, list[str]] = {
    "family": [
        "mom", "mother", "dad", "father", "sister", "brother", "parent",
        "parents", "son", "daughter", "child", "cousin", "aunt", "uncle",
        "grandma", "grandmother", "grandpa", "grandfather", "family",
    ],
    "friend": ["friend", "best friend", "roommate"],
    "colleague": [
        "coworker", "co-worker", "colleague", "boss", "manager", "teammate",
        "mentor", "client", "professor", "teacher",
    ],
    "partner": [
        "wife", "husband", "partner", "girlfriend", "boyfriend", "spouse",
        "fiance", "fiancee",
    ],
    "other": ["neighbor", "therapist", "doctor", "coach"],
}

RELATIONSHIP_UNKNOWN = "unknown"

_PERSON_LEADING_NOISE = {
    "asked",
    "called",
    "felt",
    "met",
    "saw",
    "spent",
    "talked",
    "told",
}
_PERSON_STOPWORDS = {
    "api",
    "friday",
    "felt",
    "he",
    "instagram",
    "monday",
    "pune",
    "saturday",
    "slack",
    "she",
    "sunday",
    "thursday",
    "tiktok",
    "today",
    "tuesday",
    "twitter",
    "wednesday",
    "we",
}
_NAME_PATTERN = r"(?:Dr\.\s*)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"
_INTERACTION_VERBS = (
    r"texted|called|asked|said|mentioned|noticed|replied|agreed|suggested|"
    r"listened|visited|came|joined|told|helped|checked|met|wanted|found"
)
_RELATIONSHIP_CUE_PATTERN = (
    r"best friend|old friend|close friend|friend|wife|husband|partner|"
    r"girlfriend|boyfriend|spouse|fiancee?|mom|mother|dad|father|"
    r"younger brother|older brother|younger sister|older sister|brother|sister|"
    r"cousin|manager|boss|coworker|co-worker|colleague|teammate|mentor|"
    r"professor|teacher|therapist|doctor|coach|neighbor"
)
_RELATIONSHIP_ALIAS_TYPES = {
    "ma": "family",
    "papa": "family",
    "mom": "family",
    "dad": "family",
    "mum": "family",
    "mama": "family",
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
            person_spans: list[tuple[str, int, int]] = []
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    person = self._normalize_person_name(ent.text)
                    if person:
                        people.append(person)
                        person_spans.append((person, ent.start_char, ent.end_char))
                elif ent.label_ in {"GPE", "LOC"}:
                    places.append(ent.text)
                elif ent.label_ == "ORG":
                    orgs.append(ent.text)

            cue_spans, direct_relationship_types = self._relationship_cues(text)
            for person, spans in cue_spans.items():
                people.append(person)
                person_spans.extend((person, start, end) for start, end in spans)
            action_spans = self._person_action_cues(text)
            for person, spans in action_spans.items():
                people.append(person)
                person_spans.extend((person, start, end) for start, end in spans)

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

            habits: list[str] = []
            lowered = text.lower()
            kw_blob = " ".join(keywords)
            for habit, bucket in HABIT_KEYWORDS.items():
                if any(word in lowered or word in kw_blob for word in bucket):
                    habits.append(habit)

            vader = self._get_vader()
            compound = float(vader.polarity_scores(text).get("compound", 0.0))
            deduped_people = self._dedupe(people)
            relationship_types = self._relationship_types(
                text, deduped_people, person_spans, direct_relationship_types
            )
            topics = self._topics(text, keywords, deduped_people, relationship_types)

            return {
                "entities_people": deduped_people,
                "entities_places": self._dedupe(places),
                "entities_orgs": self._dedupe(orgs),
                "keywords": keywords,
                "topics": topics,
                "habits": habits,
                "person_relationship_types": relationship_types,
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

    @staticmethod
    def _normalize_person_name(name: str) -> str:
        clean = re.sub(r"^[\"'(\[]+|[\"'),.;:\]]+$", "", name.strip())
        clean = re.sub(r"^(?:Dr\.?|Mr\.?|Mrs\.?|Ms\.?|Miss)\s+", "", clean).strip()
        parts = clean.split()
        if len(parts) > 1 and parts[0].lower() in _PERSON_LEADING_NOISE:
            clean = " ".join(parts[1:])
            parts = clean.split()
        if not parts:
            return ""
        if len(parts) == 1 and parts[0].lower() in _PERSON_STOPWORDS:
            return ""
        return " ".join(part[:1].upper() + part[1:] for part in parts)

    @classmethod
    def _relationship_cues(
        cls,
        text: str,
    ) -> tuple[dict[str, list[tuple[int, int]]], dict[str, str]]:
        spans: dict[str, list[tuple[int, int]]] = {}
        types: dict[str, str] = {}

        before_name = re.compile(
            rf"\b(?:my|our|his|her|their)?\s*(?P<cue>{_RELATIONSHIP_CUE_PATTERN})"
            rf"\s+(?:named\s+|called\s+)?(?P<name>{_NAME_PATTERN})\b"
        )
        after_name = re.compile(
            rf"\b(?P<name>{_NAME_PATTERN})\s*,?\s+"
            rf"(?:my|our|his|her|their)\s+(?P<cue>{_RELATIONSHIP_CUE_PATTERN})\b"
        )
        alias = re.compile(r"\b(?P<name>Ma|Papa|Mom|Dad|Mum|Mama)\b")

        for pattern in (before_name, after_name):
            for match in pattern.finditer(text):
                person = cls._normalize_person_name(match.group("name"))
                rel_type = cls._relationship_type_from_cue(match.group("cue"))
                if not person or rel_type == RELATIONSHIP_UNKNOWN:
                    continue
                spans.setdefault(person, []).append(match.span("name"))
                types.setdefault(person, rel_type)

        for match in alias.finditer(text):
            person = cls._normalize_person_name(match.group("name"))
            if not person:
                continue
            spans.setdefault(person, []).append(match.span("name"))
            types.setdefault(person, _RELATIONSHIP_ALIAS_TYPES[person.lower()])

        return spans, types

    @classmethod
    def _person_action_cues(cls, text: str) -> dict[str, list[tuple[int, int]]]:
        spans: dict[str, list[tuple[int, int]]] = {}
        subject_action = re.compile(
            rf"\b(?P<name>{_NAME_PATTERN})\s+(?P<verb>{_INTERACTION_VERBS})\b"
        )
        preposition_name = re.compile(
            rf"\b(?:with|from|to|about|for)\s+(?P<name>{_NAME_PATTERN})\b"
        )
        for pattern in (subject_action, preposition_name):
            for match in pattern.finditer(text):
                person = cls._normalize_person_name(match.group("name"))
                if not person:
                    continue
                spans.setdefault(person, []).append(match.span("name"))
        return spans

    @classmethod
    def _topics(
        cls,
        text: str,
        keywords: list[str],
        people: list[str],
        relationship_types: dict[str, str],
    ) -> list[str]:
        lowered = text.lower()
        kw_blob = " ".join(keywords)
        scores: dict[str, int] = {}
        for topic, bucket in TOPIC_KEYWORDS.items():
            score = 0
            for word in bucket:
                score += cls._phrase_count(lowered, word)
                score += cls._phrase_count(kw_blob, word)
            if topic == "relationship" and people:
                relation_hits = sum(
                    1 for rel_type in relationship_types.values()
                    if rel_type != RELATIONSHIP_UNKNOWN
                )
                score += min(4, relation_hits)
            if score > 0:
                scores[topic] = score

        order = {topic: idx for idx, topic in enumerate(TOPIC_KEYWORDS)}
        return sorted(scores, key=lambda topic: (-scores[topic], order[topic]))

    @staticmethod
    def _phrase_count(text: str, phrase: str) -> int:
        escaped = re.escape(phrase.lower())
        pattern = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
        return len(re.findall(pattern, text))

    @staticmethod
    def _relationship_type_from_cue(cue: str) -> str:
        cue_lower = cue.lower()
        for rel_type, keywords in RELATIONSHIP_TYPE_KEYWORDS.items():
            for keyword in keywords:
                if re.search(rf"\b{re.escape(keyword)}\b", cue_lower):
                    return rel_type
        return RELATIONSHIP_UNKNOWN

    @classmethod
    def _relationship_types(
        cls,
        text: str,
        people: list[str],
        person_spans: list[tuple[str, int, int]],
        direct_relationship_types: dict[str, str] | None = None,
        window_chars: int = 64,
    ) -> dict[str, str]:
        if not people:
            return {}

        lowered = text.lower()
        direct_relationship_types = direct_relationship_types or {}
        spans_by_person: dict[str, list[tuple[int, int]]] = {p: [] for p in people}
        for ent_text, start, end in person_spans:
            for person in people:
                if ent_text.strip().lower() == person.strip().lower():
                    spans_by_person[person].append((start, end))

        for person in people:
            if spans_by_person[person]:
                continue
            pattern = re.compile(rf"\b{re.escape(person.lower())}\b")
            spans_by_person[person] = [m.span() for m in pattern.finditer(lowered)]

        relationships: dict[str, str] = {}
        for person in people:
            direct = direct_relationship_types.get(person)
            if direct and direct != RELATIONSHIP_UNKNOWN:
                relationships[person] = direct
            else:
                relationships[person] = cls._classify_person_relationship(
                    lowered, spans_by_person[person], window_chars
                )
        return relationships

    @staticmethod
    def _classify_person_relationship(
        lowered_text: str,
        spans: list[tuple[int, int]],
        window_chars: int,
    ) -> str:
        best: tuple[int, str] | None = None
        for start, end in spans:
            left = max(0, start - window_chars)
            right = min(len(lowered_text), end + window_chars)
            window = lowered_text[left:right]
            entity_mid = start - left + max(1, end - start) // 2
            for rel_type, keywords in RELATIONSHIP_TYPE_KEYWORDS.items():
                for keyword in keywords:
                    for match in re.finditer(rf"\b{re.escape(keyword)}\b", window):
                        distance = abs(match.start() - entity_mid)
                        if best is None or distance < best[0]:
                            best = (distance, rel_type)
        return best[1] if best is not None else RELATIONSHIP_UNKNOWN
