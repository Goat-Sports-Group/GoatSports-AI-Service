"""Cộng đồng mô phỏng — bản Python của ``DemoPopulation.java`` (auth/venue/social/club-service).

Sinh tất định theo chỉ số bằng đúng thuật toán của ``java.util.Random``, nên ra cùng người, cùng tên, cùng id
(``UUID.nameUUIDFromBytes("goatsports:user:" + email)``) như các service Java mà không cần gọi chúng.
Sửa bên Java thì sửa cả ở đây; ``tests/test_demo_population.py`` chốt cùng các giá trị với ``DemoPopulationTest``.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from datetime import date

PLAYERS = 300
DOMAIN = "demo.goatsports.vn"
HISTORY_DAYS = 120
COVER_UNTIL = date(2026, 12, 31)

_MASK48 = (1 << 48) - 1
_MASK64 = (1 << 64) - 1


def _int32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value >= 1 << 31 else value


def _long(value: int) -> int:
    value &= _MASK64
    return value - (1 << 64) if value >= 1 << 63 else value


class JavaRandom:
    """``java.util.Random``: LCG 48 bit, cùng nextInt/nextDouble/nextBoolean/nextGaussian."""

    def __init__(self, seed: int) -> None:
        self._seed = (seed ^ 0x5DEECE66D) & _MASK48
        self._next_gaussian: float | None = None

    def _next(self, bits: int) -> int:
        self._seed = (self._seed * 0x5DEECE66D + 0xB) & _MASK48
        return _int32(self._seed >> (48 - bits))

    def next_int(self, bound: int) -> int:
        if bound & -bound == bound:
            return _int32((bound * self._next(31)) >> 31)
        while True:
            bits = self._next(31)
            value = bits % bound
            if _int32(bits - value + (bound - 1)) >= 0:
                return value

    def next_double(self) -> float:
        return ((self._next(26) << 27) + self._next(27)) * (1.0 / (1 << 53))

    def next_boolean(self) -> bool:
        return self._next(1) != 0

    def next_gaussian(self) -> float:
        import math

        if self._next_gaussian is not None:
            value, self._next_gaussian = self._next_gaussian, None
            return value
        while True:
            v1 = 2 * self.next_double() - 1
            v2 = 2 * self.next_double() - 1
            s = v1 * v1 + v2 * v2
            if 0 < s < 1:
                break
        multiplier = math.sqrt(-2 * math.log(s) / s)
        self._next_gaussian = v2 * multiplier
        return v1 * multiplier


def mixed_random(*keys: int) -> JavaRandom:
    """``DemoPopulation.random(keys...)``: trộn khóa qua splitmix64 rồi tạo Random."""
    value = 0x9E3779B97F4A7C15
    for key in keys:
        value = ((value ^ (key & _MASK64)) * 0xBF58476D1CE4E5B9) & _MASK64
        value = ((value ^ (value >> 31)) * 0x94D049BB133111EB) & _MASK64
        value ^= value >> 29
    return JavaRandom(_long(value))


def name_uuid(text: str) -> uuid.UUID:
    """``UUID.nameUUIDFromBytes``: MD5 của chuỗi, đặt version 3."""
    return uuid.UUID(bytes=hashlib.md5(text.encode("utf-8")).digest(), version=3)


def user_id(email: str) -> uuid.UUID:
    return name_uuid("goatsports:user:" + email)


@dataclass(frozen=True)
class City:
    name: str
    latitude: float
    longitude: float
    districts: tuple[str, ...]
    weight: int


CITIES = (
    City("Hồ Chí Minh", 10.7769, 106.7009,
         ("Quận 1", "Quận 3", "Bình Thạnh", "Phú Nhuận", "Tân Bình", "Gò Vấp", "Thủ Đức", "Quận 7"), 45),
    City("Hà Nội", 21.0278, 105.8342, ("Cầu Giấy", "Đống Đa", "Hai Bà Trưng", "Nam Từ Liêm", "Thanh Xuân", "Long Biên"), 25),
    City("Đà Nẵng", 16.0544, 108.2022, ("Hải Châu", "Sơn Trà", "Thanh Khê", "Ngũ Hành Sơn"), 15),
    City("Khánh Hòa", 12.2388, 109.1967, ("Nha Trang", "Cam Ranh", "Ninh Hòa"), 15),
)
SPORTS = ("FOOTBALL", "BADMINTON", "PICKLEBALL", "TENNIS", "BASKETBALL", "VOLLEYBALL")
_SPORT_WEIGHTS = (35, 30, 15, 10, 5, 5)
SKILLS = ("BEGINNER", "INTERMEDIATE", "ADVANCED", "PRO")
_SKILL_WEIGHTS = (30, 45, 20, 5)
_SKILL_ELO = (1000, 1250, 1500, 1750)
_SURNAMES = ("Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Phan", "Vũ", "Võ", "Đặng", "Bùi", "Đỗ", "Hồ", "Ngô",
             "Dương", "Lý")
_MALE_MIDDLE = ("Văn", "Minh", "Hoàng", "Thanh", "Quốc", "Gia", "Đức", "Bảo", "Anh", "Tuấn", "Hữu", "Công")
_FEMALE_MIDDLE = ("Thị", "Ngọc", "Thu", "Minh", "Thanh", "Khánh", "Bảo", "Phương", "Hồng", "Mai")
_MALE_GIVEN = ("Anh", "Bình", "Cường", "Dũng", "Duy", "Hải", "Hiếu", "Huy", "Hùng", "Khang", "Khoa", "Kiên", "Long",
               "Nam", "Nghĩa", "Phong", "Phúc", "Quang", "Sơn", "Tài", "Thắng", "Thành", "Tiến", "Toàn", "Trí", "Trung",
               "Tuấn", "Việt", "Vinh", "Vũ")
_FEMALE_GIVEN = ("An", "Châu", "Chi", "Giang", "Hà", "Hân", "Hạnh", "Hương", "Lan", "Linh", "Mai", "My", "Ngân",
                 "Nhi", "Như", "Phương", "Quyên", "Tâm", "Thảo", "Trâm", "Trang", "Uyên", "Vân", "Vy", "Yến")


@dataclass(frozen=True)
class Slot:
    day: int  # 1 = thứ Hai ... 7 = Chủ nhật (java.time.DayOfWeek)
    start_hour: int
    end_hour: int


@dataclass(frozen=True)
class Player:
    index: int
    email: str
    full_name: str
    male: bool
    city: City
    district: str
    latitude: float
    longitude: float
    sports: tuple[str, ...]
    skill: str
    elo: int
    bookings_per_week: float
    availability: tuple[Slot, ...]

    @property
    def user_id(self) -> uuid.UUID:
        return user_id(self.email)


def weighted(random: JavaRandom, weights: tuple[int, ...]) -> int:
    roll = random.next_int(sum(weights))
    for index, weight in enumerate(weights):
        roll -= weight
        if roll < 0:
            return index
    return len(weights) - 1


def _pick(random: JavaRandom, values: tuple[str, ...]) -> str:
    return values[random.next_int(len(values))]


def _round(value: float) -> float:
    return _java_round(value * 10_000) / 10_000.0


def _java_round(value: float) -> int:
    import math

    return math.floor(value + 0.5)


def player(index: int) -> Player:
    random = JavaRandom(0x60A7_5_2026 * 31 + index)
    male = random.next_int(100) < 68
    surname = _pick(random, _SURNAMES)
    middle = _pick(random, _MALE_MIDDLE) if male else _pick(random, _FEMALE_MIDDLE)
    given = _pick(random, _MALE_GIVEN) if male else _pick(random, _FEMALE_GIVEN)
    city = CITIES[weighted(random, tuple(item.weight for item in CITIES))]
    district = city.districts[random.next_int(len(city.districts))]
    latitude = _round(city.latitude + (random.next_double() - 0.5) * 0.08)
    longitude = _round(city.longitude + (random.next_double() - 0.5) * 0.08)
    sports = [SPORTS[weighted(random, _SPORT_WEIGHTS)]]
    if random.next_int(100) < 40:
        second = SPORTS[weighted(random, _SPORT_WEIGHTS)]
        if second not in sports:
            sports.append(second)
    skill = weighted(random, _SKILL_WEIGHTS)
    elo = _SKILL_ELO[skill] + random.next_int(200)
    if random.next_int(100) < 15:
        per_week = 2.5 + random.next_double() * 1.5
    else:
        per_week = 0.4 + random.next_double() * 1.4
    # Ngày sinh: giữ đúng số lần rút để phần sau (giờ rảnh) trùng với Java.
    random.next_int(21)
    random.next_int(12)
    random.next_int(28)
    return Player(index, f"sim{index:03d}@{DOMAIN}", f"{surname} {middle} {given}", male, city, district,
                  latitude, longitude, tuple(sports), SKILLS[skill], elo, _java_round(per_week * 100) / 100.0,
                  _availability(random))


def _availability(random: JavaRandom) -> tuple[Slot, ...]:
    slots = []
    evening = 17 + random.next_int(3)
    for day in range(1, 8):
        if day <= 5 and random.next_int(100) < 55:
            slots.append(Slot(day, evening, min(22, evening + 3)))
    weekend = 6 if random.next_boolean() else 7
    morning = 6 + random.next_int(3)
    slots.append(Slot(weekend, morning, morning + 4))
    return tuple(slots)


def players() -> list[Player]:
    return [player(index) for index in range(PLAYERS)]
