"""Unit tests for every rule in data/rules.py, its thresholds, and the precedence order."""

from dataclasses import replace
from pathlib import Path

import pytest
import rules
from rules import Submission, decide

GUIDE = (Path(__file__).resolve().parent / "appetite_guide.md").read_text("utf-8")

CLEAN = Submission(
    business_class="office",
    tiv=2_000_000,
    losses_5y=0,
    flood_zone="X",
    construction="joisted_masonry",
    year_built=1995,
)


def d(**changes) -> rules.Decision:
    return decide(replace(CLEAN, **changes))


def test_clean_submission_is_accepted():
    r = decide(CLEAN)
    assert r.decision == "accept"
    assert r.decline_reasons == () and r.refer_reasons == ()
    assert r.in_appetite is True


@pytest.mark.parametrize("cls", rules.ELIGIBLE_CLASSES)
def test_every_eligible_class_can_be_accepted(cls):
    extra = {"cooking": True, "hood_suppression": True} if cls == "restaurant" else {}
    assert d(business_class=cls, **extra).decision == "accept"


# ---- decline rules -------------------------------------------------------------------------


@pytest.mark.parametrize("cls", rules.EXCLUDED_CLASSES)
def test_excluded_class_declines(cls):
    r = d(business_class=cls)
    assert r.decision == "decline"
    assert r.decline_reasons == ("excluded_class",)
    assert r.in_appetite is False


def test_vacant_declines():
    assert d(vacant=True).decline_reasons == ("vacant",)


@pytest.mark.parametrize(
    ("losses", "expected"),
    [(0, "accept"), (1, "accept"), (2, "refer"), (3, "refer"), (4, "decline"), (7, "decline")],
)
def test_loss_thresholds(losses, expected):
    assert d(losses_5y=losses).decision == expected


@pytest.mark.parametrize("zone", rules.HIGH_RISK_FLOOD_ZONES)
def test_high_risk_flood_without_mitigation_declines(zone):
    assert d(flood_zone=zone).decline_reasons == ("flood_unmitigated",)


@pytest.mark.parametrize("zone", rules.HIGH_RISK_FLOOD_ZONES)
def test_high_risk_flood_with_mitigation_is_fine(zone):
    assert d(flood_zone=zone, flood_mitigation=True).decision == "accept"


@pytest.mark.parametrize("zone", rules.LOW_RISK_FLOOD_ZONES)
def test_low_risk_flood_zone_is_fine(zone):
    assert d(flood_zone=zone).decision == "accept"


@pytest.mark.parametrize(
    ("tiv", "expected"),
    [
        (4_499_999, "accept"),
        (4_500_000, "refer"),  # "from $4,500,000" is inclusive
        (4_750_000, "refer"),
        (5_000_000, "refer"),  # "over $5,000,000" is exclusive
        (5_000_001, "decline"),
        (9_000_000, "decline"),
    ],
)
def test_tiv_thresholds(tiv, expected):
    assert d(tiv=tiv).decision == expected


# ---- refer rules ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", rules.UNLISTED_CLASSES + ("bowling_alley",))
def test_unlisted_class_refers(cls):
    assert d(business_class=cls).refer_reasons == ("unlisted_class",)


def test_restaurant_cooking_without_suppression_refers():
    r = d(business_class="restaurant", cooking=True, hood_suppression=False)
    assert r.refer_reasons == ("cooking_no_suppression",)


def test_restaurant_cooking_with_suppression_accepts():
    assert d(business_class="restaurant", cooking=True, hood_suppression=True).decision == "accept"


def test_restaurant_without_cooking_needs_no_suppression():
    assert d(business_class="restaurant", cooking=False).decision == "accept"


def test_cooking_rule_only_applies_to_restaurants():
    assert d(business_class="office", cooking=True, hood_suppression=False).decision == "accept"


@pytest.mark.parametrize(
    ("construction", "year", "expected"),
    [
        ("frame", 1979, "refer"),
        ("frame", 1980, "accept"),  # "before 1980" is exclusive
        ("frame", 1925, "refer"),
        ("joisted_masonry", 1950, "accept"),
        ("non_combustible", 1960, "accept"),
        ("fire_resistive", 1930, "accept"),
    ],
)
def test_frame_before_1980(construction, year, expected):
    assert d(construction=construction, year_built=year).decision == expected


def test_missing_year_refers():
    assert d(year_built=None).refer_reasons == ("missing_year_or_construction",)


def test_missing_construction_refers():
    assert d(construction=None).refer_reasons == ("missing_year_or_construction",)


def test_frame_with_missing_year_refers_as_missing():
    assert d(construction="frame", year_built=None).refer_reasons == (
        "missing_year_or_construction",
    )


def test_multiple_refer_reasons_are_all_reported():
    r = d(tiv=4_600_000, losses_5y=2, year_built=None)
    assert r.decision == "refer"
    assert r.refer_reasons == ("tiv_near_max", "losses_2_or_3", "missing_year_or_construction")


# ---- precedence ----------------------------------------------------------------------------


def test_decline_beats_refer():
    r = d(vacant=True, losses_5y=2, year_built=None)
    assert r.decision == "decline"
    assert r.decline_reasons == ("vacant",)
    assert r.refer_reasons == ("losses_2_or_3", "missing_year_or_construction")
    assert r.reasons == ("vacant",)


def test_excluded_class_is_not_also_unlisted():
    r = d(business_class="cannabis")
    assert r.refer_reasons == ()


def test_multiple_decline_reasons():
    r = d(business_class="nightclub", tiv=6_000_000, losses_5y=5, flood_zone="VE")
    assert r.decision == "decline"
    assert r.decline_reasons == (
        "excluded_class",
        "losses_over_3",
        "flood_unmitigated",
        "tiv_over_max",
    )


def test_refer_beats_accept():
    assert d(losses_5y=3).decision == "refer"


@pytest.mark.parametrize(
    ("decision", "in_appetite"), [("accept", True), ("refer", True), ("decline", False)]
)
def test_in_appetite_mapping(decision, in_appetite):
    r = rules.Decision(decision=decision, decline_reasons=(), refer_reasons=())
    assert r.in_appetite is in_appetite


# ---- validation and guide consistency -------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [{"tiv": 0}, {"losses_5y": -1}, {"flood_zone": "Q"}, {"construction": "straw"}],
)
def test_invalid_facts_raise(bad):
    with pytest.raises(ValueError):
        replace(CLEAN, **bad)


def test_guide_states_the_same_thresholds_as_the_code():
    assert f"${rules.TIV_MAX:,}" in GUIDE
    assert f"${rules.TIV_REFER_FLOOR:,}" in GUIDE
    assert f"before {rules.FRAME_CUTOFF_YEAR}" in GUIDE
    assert f"More than {rules.MAX_LOSSES} losses" in GUIDE
    assert f"{rules.REFER_LOSSES_MIN} or {rules.MAX_LOSSES} losses" in GUIDE
    for zone in rules.HIGH_RISK_FLOOD_ZONES:
        assert zone in GUIDE


def test_guide_is_short():
    # Small-context contestants need guide + email to fit; ~4 chars per token.
    assert len(GUIDE) / 4 < 350
