from __future__ import annotations

from backend.analytics.models import TriggerStat
from backend.orchestrator.orchestrator import Orchestrator
from backend.orchestrator.packet import IntelligencePacket
from backend.profile.models import UserProfile


def test_assemble_well_formed():
    pkt = Orchestrator().assemble(
        text="hi",
        emotion_result={"emotion": "joy", "confidence": 0.8},
        sentiment=0.5,
        insights=["insight a"],
        triggers=[TriggerStat(topic="career", frequency=3, avg_sentiment=0.1,
                              dominant_emotion="joy", trend="stable",
                              confidence=0.3, explanation="x")],
        user_profile=UserProfile(entry_count=5),
    )
    assert isinstance(pkt, IntelligencePacket)
    assert pkt.current_entry_emotion == "joy"
    assert pkt.current_entry_sentiment == 0.5
    assert pkt.insights == ["insight a"]
    assert pkt.user_profile.entry_count == 5
    assert pkt.triggers[0].topic == "career"


def test_assemble_handles_none_defaults():
    pkt = Orchestrator().assemble(
        text="hi",
        emotion_result={"emotion": "neutral", "confidence": 0.0},
        sentiment=0.0,
    )
    assert pkt.insights == []
    assert pkt.triggers == []
    assert pkt.user_profile is None
    assert pkt.proactive_alerts == []
    assert pkt.predictions == {}
