"""Kèo ghép trận mô phỏng (dev / demo) cho cộng đồng ``population``: đủ kết quả để logistic regression học được.

Mỗi ngày từ ``HISTORY_DAYS`` ngày trước tới ``COVER_UNTIL`` có vài kèo giữa người chơi mô phỏng cùng môn (phần lớn
cùng tỉnh). Điểm 7 tiêu chí tính từ hồ sơ thật của hai người (ELO, trình độ, giờ rảnh, khoảng cách...). Kèo thành công
hay hỏng được quyết định bởi một "sự thật ngầm" khác trọng số viết tay — khung giờ và khoảng cách quan trọng hơn ELO —
nên mô hình học ra trọng số khác luật cũ và AUC kiểm chứng chéo vượt ngưỡng.

Tất định (id và kết quả theo ngày + thứ tự), chỉ tạo kèo đã được ghép tới thời điểm chạy; chạy lại thì cập nhật trạng
thái các kèo vừa qua ngày chơi. Bật bằng ``DEMO_SIMULATION_ENABLED=true``.
"""

from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta, timezone

from app.application.port.persistence import MatchmakingSessionRepository
from app.application.service.matchmaking_scoring import SCORE_WEIGHTS
from app.domain.entity.models import (
    MatchAcceptance,
    MatchmakingSession,
    MatchResult,
    MatchScoreBreakdown,
    OpponentFeedback,
    PlayerProfile,
    SideMember,
)
from app.domain.enums import AcceptanceDecision, MatchSessionStatus, SkillLevel, SportType
from app.infrastructure.demo.population import (
    COVER_UNTIL,
    HISTORY_DAYS,
    SKILLS,
    JavaRandom,
    Player,
    mixed_random,
    name_uuid,
    players,
)

VIETNAM = timezone(timedelta(hours=7))
_SPORT_WEIGHTS = (35, 30, 15, 10, 5, 5)
_SPORTS = ("FOOTBALL", "BADMINTON", "PICKLEBALL", "TENNIS", "BASKETBALL", "VOLLEYBALL")
_STYLES = ("BALANCED", "ATTACKING", "DEFENSIVE", "CONSISTENT")
_COMMENTS_GOOD = ("Chơi đẹp, đúng giờ.", "Trình ngang nhau, rất vui.", "Sẽ hẹn kèo tiếp.", None)
_COMMENTS_BAD = ("Tới trễ gần 30 phút.", "Trình chênh quá nhiều.", "Thái độ chưa tốt.")


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _weighted(random: JavaRandom, weights: tuple[int, ...]) -> int:
    roll = random.next_int(sum(weights))
    for index, weight in enumerate(weights):
        roll -= weight
        if roll < 0:
            return index
    return len(weights) - 1


def _distance_km(first: Player, second: Player) -> float:
    lat = math.radians((first.latitude + second.latitude) / 2)
    dx = (first.longitude - second.longitude) * 111.32 * math.cos(lat)
    dy = (first.latitude - second.latitude) * 110.57
    return math.hypot(dx, dy)


def _overlap_hours(first: Player, second: Player, weekday: int) -> tuple[int, int, int]:
    """Số giờ cả hai cùng rảnh trong ngày (thứ 1..7) và khung giờ chung; không có thì (0, 18, 20)."""
    best = (0, 18, 20)
    for a in first.availability:
        for b in second.availability:
            if a.day == b.day == weekday:
                start, end = max(a.start_hour, b.start_hour), min(a.end_hour, b.end_hour)
                if end - start > best[0]:
                    best = (end - start, start, end)
    return best


def _breakdown(first: Player, second: Player, play_date: date, random: JavaRandom):
    overlap, start, end = _overlap_hours(first, second, play_date.isoweekday())
    distance = _distance_km(first, second)
    skill_gap = abs(SKILLS.index(first.skill) - SKILLS.index(second.skill))
    elo_gap = abs(first.elo - second.elo)
    scores = {
        "elo": max(0.0, 100 - elo_gap / 4),
        "skill": max(0.0, 100 - 35 * skill_gap),
        "schedule": min(100.0, 30 + 25 * overlap) if overlap else 15 + random.next_int(25),
        "distance": max(0.0, 100 * (1 - distance / 15)),
        "play_style": 50 + random.next_int(46),
        "position": 60 + random.next_int(36),
        "experience": max(0.0, 100 - 25 * abs(first.bookings_per_week - second.bookings_per_week)),
    }
    total = sum(scores[key] * weight for key, weight in SCORE_WEIGHTS.items())
    breakdown = MatchScoreBreakdown(
        elo_score=round(scores["elo"], 1),
        skill_score=round(scores["skill"], 1),
        schedule_score=round(scores["schedule"], 1),
        distance_score=round(scores["distance"], 1),
        play_style_score=float(scores["play_style"]),
        position_score=float(scores["position"]),
        experience_score=round(scores["experience"], 1),
        total_score=round(total, 1),
        reasons=_reasons(scores, distance, overlap),
    )
    start = start if overlap else 18
    end = min(start + 2, end if overlap else 20)
    return breakdown, distance, elo_gap, time(start), time(max(end, start + 1))


def _reasons(scores: dict[str, float], distance: float, overlap: int) -> list[str]:
    reasons = []
    if scores["elo"] >= 70:
        reasons.append("ELO gần nhau, trận dễ cân bằng")
    if overlap >= 2:
        reasons.append(f"Cùng rảnh {overlap} giờ trong ngày chơi")
    if distance <= 5:
        reasons.append(f"Ở gần nhau (~{distance:.1f} km)")
    return reasons or ["Cùng môn và cùng khu vực"]


def success_probability(breakdown: MatchScoreBreakdown) -> float:
    """"Sự thật ngầm" của mô phỏng: khung giờ và khoảng cách quyết định kèo có diễn ra hay không nhiều hơn ELO."""
    z = (
        -5.4
        + 3.2 * breakdown.schedule_score / 100
        + 2.6 * breakdown.distance_score / 100
        + 1.3 * breakdown.elo_score / 100
        + 0.7 * breakdown.skill_score / 100
        + 0.4 * breakdown.play_style_score / 100
        + 0.3 * breakdown.experience_score / 100
    )
    return _sigmoid(z)


def _profile(player: Player, sport: str) -> PlayerProfile:
    return PlayerProfile(
        participant_profile_id=name_uuid(f"goatsports:sim-match-profile:{player.email}|{sport}"),
        participant_id=player.user_id,
        name=player.full_name,
        sport_type=SportType(sport),
        elo_rating=player.elo,
        latitude=player.latitude,
        longitude=player.longitude,
        skill_level=SkillLevel(player.skill),
        play_style=_STYLES[player.index % len(_STYLES)],
        match_count=int(round(player.bookings_per_week * 14)),
        members=[SideMember(user_id=player.user_id, name=player.full_name, elo_rating=player.elo)],
    )


def _at(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime.combine(day, time(hour, minute), tzinfo=VIETNAM).astimezone(timezone.utc)


def _score(sport: str, random: JavaRandom) -> tuple[int, int]:
    if sport == "FOOTBALL":
        win = 1 + random.next_int(5)
        return win, random.next_int(win)
    if sport in ("TENNIS", "VOLLEYBALL"):
        return 2, random.next_int(2)
    if sport == "BASKETBALL":
        win = 15 + random.next_int(7)
        return win, max(5, win - 1 - random.next_int(9))
    if sport == "PICKLEBALL":
        return 11, 3 + random.next_int(8)
    return 21, 8 + random.next_int(12)


def build_sessions(now: datetime) -> list[MatchmakingSession]:
    """Mọi kèo mô phỏng đã được ghép tới ``now`` (giờ UTC), với trạng thái tại thời điểm đó."""
    today = now.astimezone(VIETNAM).date()
    first = today - timedelta(days=HISTORY_DAYS)
    last = max(COVER_UNTIL, today)
    population = players()
    by_sport = {sport: [p for p in population if sport in p.sports] for sport in _SPORTS}
    sessions = []
    day = first
    while day <= last:
        random = mixed_random(0x3A7C, day.toordinal())
        progress = min(1.0, (day - first).days / HISTORY_DAYS)
        weekend = day.isoweekday() >= 6
        count = max(1, round((2 + 4 * progress) * (1.4 if weekend else 1.0) + random.next_gaussian()))
        for index in range(count):
            session = _session(day, index, random, by_sport, today, now)
            if session is not None:
                sessions.append(session)
        day += timedelta(days=1)
    return sessions


def _session(day, index, random, by_sport, today, now):
    sport = _SPORTS[_weighted(random, _SPORT_WEIGHTS)]
    pool = by_sport[sport]
    first = pool[random.next_int(len(pool))]
    local = [p for p in pool if p is not first and p.city == first.city]
    others = [p for p in pool if p is not first]
    # 88% ghép người cùng tỉnh.
    candidates = local if local and random.next_int(100) < 88 else others
    second = candidates[random.next_int(len(candidates))]
    lead = 1 + random.next_int(4)
    matched_at = _at(day - timedelta(days=lead), 8 + random.next_int(14), random.next_int(60))
    roll = random.next_double()
    outcome = random.next_int(100)
    breakdown, distance, elo_gap, start, end = _breakdown(first, second, day, random)
    score = _score(sport, random)
    first_wins = random.next_double() < 1 / (1 + 10 ** ((second.elo - first.elo) / 400))
    ratings = (3 + random.next_int(3), 3 + random.next_int(3), 1 + random.next_int(2), 1 + random.next_int(2))
    if matched_at > now:
        return None

    profiles = [_profile(first, sport), _profile(second, sport)]
    session = MatchmakingSession(
        session_id=name_uuid(f"goatsports:sim-match:{day}|{index}"),
        sport_type=SportType(sport),
        play_date=day,
        start_time=start,
        end_time=end,
        timezone="Asia/Ho_Chi_Minh",
        compatibility_score=breakdown.total_score,
        distance_km=round(distance, 2),
        elo_difference=elo_gap,
        score_breakdown=breakdown,
        status=MatchSessionStatus.PROPOSED,
        matched_at=matched_at,
        expires_at=matched_at + timedelta(hours=12),
        participants=profiles,
    )
    accepted = roll < success_probability(breakdown)
    decided_at = matched_at + timedelta(hours=2)
    played = day < today
    if not played:
        # Kèo sắp tới: còn chờ người thứ hai hoặc đã chốt.
        if accepted and decided_at <= now:
            session.status = MatchSessionStatus.CONFIRMED
            session.acceptances = [_accept(p, AcceptanceDecision.ACCEPTED, decided_at) for p in profiles]
            session.accepted_at = session.confirmed_at = decided_at
        elif matched_at + timedelta(hours=1) <= now:
            session.status = MatchSessionStatus.ACCEPTED_BY_ONE
            session.acceptances = [_accept(profiles[0], AcceptanceDecision.ACCEPTED, matched_at + timedelta(hours=1))]
        return session

    if not accepted:
        # Hỏng trước giờ chơi: bị từ chối, hết hạn hoặc hủy; một phần đã đá nhưng chấm nhau rất thấp.
        if outcome < 40:
            session.status = MatchSessionStatus.REJECTED
            session.acceptances = [_accept(profiles[1], AcceptanceDecision.REJECTED, decided_at)]
            return session
        if outcome < 65:
            session.status = MatchSessionStatus.EXPIRED
            return session
        if outcome < 80:
            session.status = MatchSessionStatus.CANCELLED
            session.acceptances = [_accept(p, AcceptanceDecision.ACCEPTED, decided_at) for p in profiles]
            session.accepted_at = decided_at
            return session
        good = False
    else:
        good = True
    session.acceptances = [_accept(p, AcceptanceDecision.ACCEPTED, decided_at) for p in profiles]
    session.accepted_at = session.confirmed_at = decided_at
    finished = _at(day, end.hour)
    session.checked_in_at = finished - timedelta(hours=2)
    session.completed_at = finished + timedelta(minutes=40)
    session.status = MatchSessionStatus.COMPLETED
    home, away = score if first_wins else (score[1], score[0])
    winner = first.user_id if home > away else second.user_id if away > home else None
    session.result = MatchResult(
        participant_one_score=home, participant_two_score=away, winner_id=winner,
        confirmed_at=session.completed_at, elo_applied=True,
    )
    rating_a, rating_b = (ratings[0], ratings[1]) if good else (ratings[2], ratings[3])
    session.feedback = [
        _feedback(first, second, rating_a, good, session.completed_at, ratings[0]),
        _feedback(second, first, rating_b, good, session.completed_at, ratings[1]),
    ]
    return session


def _accept(profile: PlayerProfile, decision: AcceptanceDecision, at: datetime) -> MatchAcceptance:
    return MatchAcceptance(
        acceptance_id=name_uuid(f"goatsports:sim-acceptance:{profile.participant_profile_id}|{at.isoformat()}"),
        participant_profile_id=profile.participant_profile_id,
        decision=decision,
        decided_at=at,
    )


def _feedback(reviewer: Player, reviewee: Player, rating: int, good: bool, at: datetime, salt: int) -> OpponentFeedback:
    pool = _COMMENTS_GOOD if good else _COMMENTS_BAD
    return OpponentFeedback(
        feedback_id=name_uuid(f"goatsports:sim-feedback:{reviewer.email}|{reviewee.email}|{at.isoformat()}"),
        reviewer_id=reviewer.user_id,
        reviewee_id=reviewee.user_id,
        rating=rating,
        fair_play_rating=rating if not good else min(5, rating + salt % 2),
        comment=pool[salt % len(pool)],
        created_at=at + timedelta(hours=1),
    )


def seed(repository: MatchmakingSessionRepository, now: datetime | None = None) -> int:
    sessions = build_sessions(now or datetime.now(timezone.utc))
    for session in sessions:
        repository.save(session)
    return len(sessions)
