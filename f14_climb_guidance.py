# f14_climb_guidance.py
# Climb guidance utilities for the DCS F-14B performance app.
# Safe for Python 3.8+.

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

# Engine thrust (approximate, MIL scale) used for gradient proxies
ENGINE_THRUST_LBF = {"MIL": 16333.0, "AB": 26950.0}   # per engine
# Simple drag/weight proxies by flap
DRAG_OVER_W = {0: 0.06, 20: 0.08, 40: 0.10}

def _flap_label_to_deg(label: str) -> int:
    s = (label or "").strip().upper()
    if s.startswith("UP"):
        return 0
    if s.startswith("FULL"):
        return 40
    return 20  # MAN

def _required_n1_for_gradient(gw_lbs: float, flap_deg: int, gradient_req: float, engines: int) -> float:
    """
    Solve T/W - D/W >= gradient_req for N1 (% of MIL scale).
    T/W = engines * T_MIL(N1) / W; T_MIL approx linear with N1% on our MIL scale.
    """
    drag_over_w = DRAG_OVER_W.get(flap_deg, 0.08)
    rhs = gradient_req + drag_over_w
    if rhs <= 0.0:
        return 90.0
    n1 = (rhs * max(gw_lbs, 1.0)) / (engines * ENGINE_THRUST_LBF["MIL"]) * 100.0
    return float(max(0.0, n1))

@dataclass
class ClimbSchedule:
    climb_n1: int               # N1 % to set at acceleration altitude
    v2_plus: str                # initial segment (e.g., "V2+15")
    accel_alt_ft_agl: int       # acceleration altitude AGL
    limit_below_10k_kias: int   # regulatory IAS limit below 10k
    clean_climb_kias: int       # IAS target once clean
    mach_transition: float      # Mach target to transition to in climb
    notes: Tuple[str, ...]      # caveats/notes

def recommend_climb_schedule(gw_lbs: float,
                             flap_after_cleanup: str = "UP",
                             accel_alt_ft_agl: float = 1000.0,
                             policy: str = "conservative",
                             below_10k_limit_kias: int = 250,
                             clean_climb_kias: int = 300,
                             mach_transition: float = 0.73) -> ClimbSchedule:
    """
    Return a climb schedule that respects regulatory speed limits and provides an efficiency-aware climb N1.
    policy:
      - "conservative": +2% margin above minimum AEO gradient (200 ft/nm ~ 3.3%)
      - "practical":    +1% margin
    """
    flap_deg = _flap_label_to_deg(flap_after_cleanup)
    # Minimum N1 for AEO 200 ft/nm with both engines at selected N1
    min_n1_aeo = _required_n1_for_gradient(gw_lbs, flap_deg, gradient_req=0.033, engines=2)

    margin = 2.0 if (policy or "conservative").lower().startswith("conserv") else 1.0
    floor = 100.0 if flap_deg == 40 else 90.0
    n1_cmd = max(floor, min(100.0, min_n1_aeo + margin))
    n1_cmd_int = int(round(n1_cmd))

    v2_plus = "V2+15"

    return ClimbSchedule(
        climb_n1=n1_cmd_int,
        v2_plus=v2_plus,
        accel_alt_ft_agl=int(round(accel_alt_ft_agl)),
        limit_below_10k_kias=int(below_10k_limit_kias),
        clean_climb_kias=int(clean_climb_kias),
        mach_transition=float(mach_transition),
        notes=(
            "Respect 250 KIAS below 10,000 MSL (unless ATC/LOA differs).",
            "At accel altitude: set climb N1, accelerate to 250 KIAS (if below 10k), then 300 KIAS and Mach hold.",
            "Increase N1 if hot/high or if climb rate is poor after cleanup."
        )
    )

def format_climb_card(schedule: ClimbSchedule) -> str:
    """
    Return an HTML snippet suitable for Streamlit markdown.
    """
    # Build notes list items
    notes_html = "".join("<li>{}</li>".format(n) for n in schedule.notes)
    html = (
        "<div class='f14-card'>"
        "<b>Climb Recommendation</b>"
        "<div>Fly <b>{v2}</b> to <b>{aal} ft AGL</b>.</div>"
        "<div>At {aal} AGL (flaps UP), set <b>{n1}% N1</b>.</div>"
        "<div>Below 10,000 ft: <b>{lim} KIAS</b> max. Then <b>{clean} KIAS</b>, transition to <b>M {mach:.2f}</b>.</div>"
        "<ul>{notes}</ul>"
        "</div>"
    ).format(
        v2=schedule.v2_plus,
        aal=schedule.accel_alt_ft_agl,
        n1=schedule.climb_n1,
        lim=schedule.limit_below_10k_kias,
        clean=schedule.clean_climb_kias,
        mach=schedule.mach_transition,
        notes=notes_html
    )
    return html
