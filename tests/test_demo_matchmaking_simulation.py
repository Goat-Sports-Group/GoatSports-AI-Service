"""Kèo mô phỏng phải đủ và đủ tín hiệu để logistic regression thay luật viết tay."""

from datetime import datetime, timezone

from app.application.service.matchmaking_model import MIN_AUC, MIN_SAMPLES, label, train
from app.domain.enums import MatchSessionStatus
from app.infrastructure.demo.matchmaking_simulation import build_sessions

NOW = datetime(2026, 11, 12, 5, 0, tzinfo=timezone.utc)


def test_sessions_are_deterministic_and_never_matched_in_the_future():
    sessions = build_sessions(NOW)
    again = build_sessions(NOW)
    assert [s.session_id for s in sessions] == [s.session_id for s in again]
    assert len({s.session_id for s in sessions}) == len(sessions)
    assert all(s.matched_at <= NOW for s in sessions)
    assert all(s.participants[0].participant_id != s.participants[1].participant_id for s in sessions)
    # Kèo sắp tới chưa có nhãn; kèo đã qua thì có.
    assert any(s.status == MatchSessionStatus.CONFIRMED for s in sessions)
    assert any(s.status == MatchSessionStatus.COMPLETED for s in sessions)


def test_model_learns_from_the_simulated_outcomes():
    sessions = build_sessions(NOW)
    labels = [label(s) for s in sessions]
    assert sum(1 for value in labels if value is not None) >= MIN_SAMPLES
    outcome = train(sessions, now=NOW)
    assert outcome.model is not None, outcome.reason
    assert outcome.auc is not None and outcome.auc >= MIN_AUC + 0.1
    # "Sự thật ngầm" đặt khung giờ nặng nhất: mô hình học ra như vậy, khác luật viết tay (ELO nặng nhất).
    weights = {name: outcome.model.metrics[f"w_{name}"] for name in ("schedule_score", "elo_score")}
    assert weights["schedule_score"] > weights["elo_score"]
