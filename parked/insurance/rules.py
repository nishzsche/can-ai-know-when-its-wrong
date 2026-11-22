"""Rule engine for the fictional Quillfield Mutual appetite guide (data/appetite_guide.md).

Every synthetic label comes from `decide()`, never from an LLM. Pure functions, no I/O.

Precedence: any decline rule -> "decline"; otherwise any refer rule -> "refer"; otherwise "accept".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

ELIGIBLE_CLASSES = ("office", "retail", "restaurant", "warehouse", "light_manufacturing")
EXCLUDED_CLASSES = ("nightclub", "fireworks", "cannabis", "scrap_yard")
# Real-world classes the guide does not list. Any class outside ELIGIBLE/EXCLUDED is "unlisted";
# these are the ones the generator uses.
UNLISTED_CLASSES = ("daycare", "fitness_gym", "motel", "church", "auto_body")

CONSTRUCTION_TYPES = ("frame", "joisted_masonry", "non_combustible", "fire_resistive")
HIGH_RISK_FLOOD_ZONES = ("A", "AE", "V", "VE")
LOW_RISK_FLOOD_ZONES = ("X", "B", "C")
FLOOD_ZONES = HIGH_RISK_FLOOD_ZONES + LOW_RISK_FLOOD_ZONES

TIV_MAX = 5_000_000  # decline above this
TIV_REFER_FLOOR = 4_500_000  # refer from here up to TIV_MAX inclusive (within 10% below the max)
MAX_LOSSES = 3  # decline above this many losses in 5 years
REFER_LOSSES_MIN = 2  # refer at 2..3 losses
FRAME_CUTOFF_YEAR = 1980  # frame construction built *before* this year -> refer

DECISIONS = ("accept", "refer", "decline")

# Reason codes, in the order the guide lists them.
DECLINE_REASONS = (
    "excluded_class",
    "vacant",
    "losses_over_3",
    "flood_unmitigated",
    "tiv_over_max",
)
REFER_REASONS = (
    "unlisted_class",
    "tiv_near_max",
    "losses_2_or_3",
    "cooking_no_suppression",
    "frame_pre_1980",
    "missing_year_or_construction",
)


@dataclass(frozen=True)
class Submission:
    """The facts an underwriter needs. `None` means the email does not state it."""

    business_class: str
    tiv: int
    losses_5y: int
    flood_zone: str
    flood_mitigation: bool = False
    vacant: bool = False
    construction: str | None = None
    year_built: int | None = None
    cooking: bool = False  # restaurant cooks on site (only read for restaurants)
    hood_suppression: bool = False  # hood fire-suppression system documented

    def __post_init__(self) -> None:
        if self.tiv <= 0:
            raise ValueError(f"tiv must be positive, got {self.tiv}")
        if self.losses_5y < 0:
            raise ValueError(f"losses_5y must be >= 0, got {self.losses_5y}")
        if self.flood_zone not in FLOOD_ZONES:
            raise ValueError(f"unknown flood zone {self.flood_zone!r}")
        if self.construction is not None and self.construction not in CONSTRUCTION_TYPES:
            raise ValueError(f"unknown construction {self.construction!r}")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Decision:
    decision: str
    decline_reasons: tuple[str, ...]
    refer_reasons: tuple[str, ...]

    @property
    def in_appetite(self) -> bool:
        return self.decision != "decline"

    @property
    def reasons(self) -> tuple[str, ...]:
        """The reasons that actually determine the decision."""
        if self.decision == "decline":
            return self.decline_reasons
        if self.decision == "refer":
            return self.refer_reasons
        return ()


def decline_reasons(s: Submission) -> list[str]:
    reasons = []
    if s.business_class in EXCLUDED_CLASSES:
        reasons.append("excluded_class")
    if s.vacant:
        reasons.append("vacant")
    if s.losses_5y > MAX_LOSSES:
        reasons.append("losses_over_3")
    if s.flood_zone in HIGH_RISK_FLOOD_ZONES and not s.flood_mitigation:
        reasons.append("flood_unmitigated")
    if s.tiv > TIV_MAX:
        reasons.append("tiv_over_max")
    return reasons


def refer_reasons(s: Submission) -> list[str]:
    reasons = []
    if s.business_class not in ELIGIBLE_CLASSES and s.business_class not in EXCLUDED_CLASSES:
        reasons.append("unlisted_class")
    if TIV_REFER_FLOOR <= s.tiv <= TIV_MAX:
        reasons.append("tiv_near_max")
    if REFER_LOSSES_MIN <= s.losses_5y <= MAX_LOSSES:
        reasons.append("losses_2_or_3")
    if s.business_class == "restaurant" and s.cooking and not s.hood_suppression:
        reasons.append("cooking_no_suppression")
    if s.construction == "frame" and s.year_built is not None and s.year_built < FRAME_CUTOFF_YEAR:
        reasons.append("frame_pre_1980")
    if s.year_built is None or s.construction is None:
        reasons.append("missing_year_or_construction")
    return reasons


def decide(s: Submission) -> Decision:
    dec = tuple(decline_reasons(s))
    ref = tuple(refer_reasons(s))
    if dec:
        decision = "decline"
    elif ref:
        decision = "refer"
    else:
        decision = "accept"
    return Decision(decision=decision, decline_reasons=dec, refer_reasons=ref)
