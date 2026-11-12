"""Bản Python của DemoPopulation phải ra đúng người như bản Java (cùng giá trị chốt với DemoPopulationTest.java)."""

from uuid import UUID

from app.infrastructure.demo.population import players, user_id


def test_generates_the_same_population_as_java():
    population = players()
    assert len(population) == 300
    first = population[0]
    assert first.full_name == "Lý Hoàng Tuấn"
    assert first.city.name == "Khánh Hòa"
    assert first.sports == ("BADMINTON",)
    assert first.user_id == UUID("427feff9-522c-333c-ab96-6add5aba4394")
    assert population[119].full_name == "Phạm Hoàng Quang"
    assert sum(player.elo for player in population[:120]) == 162878
    assert population[299].full_name == "Phan Bảo Huy"
    assert population[299].sports == ("BADMINTON", "BASKETBALL")
    assert sum(player.elo for player in population) == 401997
    assert user_id("simowner6@demo.goatsports.vn") == UUID("3a9f0569-9538-3bf2-b6a4-792ed8211aab")
    assert user_id("simowner15@demo.goatsports.vn") == UUID("e56e4e7a-68ae-3acf-acde-0236225b7ccc")


def test_every_player_has_a_weekend_slot():
    assert all(any(slot.day >= 6 for slot in player.availability) for player in players())
