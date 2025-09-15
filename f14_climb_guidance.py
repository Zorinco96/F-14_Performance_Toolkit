# f14_climb_guidance.py
# Drop-in climb guidance for the DCS F-14B app.
# Place this file next to f14_takeoff_app.py and import its helpers in the app.

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict

# Local constants (kept here so this file is self-contained)
ENGINE_THRUST_LBF = {"MIL": 16333.0, "AB": 26950.0}   # per engine (approx, MIL scale)
# Drag/W estimates used in the core for gradient proxies
DRAG_OVER_W = {0: 0.06, 20: 0.08, 40: 0.10}

# Utility
def _flap_label_to_deg(label: str) -> int:
    s = (label or "").strip().upper()
    if s.startswith("UP"): return 0
    if s.startswith("FULL"): return 40
    return 20  # default MAN

def _required_n1_for_gradient(gw_lbs: float, flap_deg: int, gradient_req: float, engines: int) -> float:
    """Solve T/W - D/W >= gradient_req for N1 (% of MIL scale).
       T/W = thrust_total / W = engines * T_MIL(N1) / W, with T_MIL linearized by N1% here.
       This matches the proxy used in the core for OEI/AEO checks.
    """
    drag_over_w = DRAG_OVER_W.get(flap_deg, 0.08)
    # Need: engines * T_MIL(N1) / W - drag_over_w >= gradient_req
    # => engines * T_MIL(N1) / W >= gradient_req + drag_over_w
    rhs = gradient_req + drag_over_w
    if rhs <= 0.0:  # degenerate case
        return 90.0
    # T_MIL(N1) scales linearly with N1% on our simplified model
    n1 = (rhs * gw_lbs) / (engines * ENGINE_THRUST_LBF["MIL"]) * 100.0
    return float(max(0.0, n1))

@dataclass
class ClimbSchedule:
    climb_n1: int               # N1 % to set at acceleration altitude
    v2_plus: str                # recommended initial speed (string like "V2+15")
    accel_alt_ft_agl: int       # acceleration altitude AGL
    limit_below_10k_kias: int   # regulatory IAS limit below 10k
    clean_climb_kias: int       # IAS target once clean
    mach_transition: float      # Mach target to transition to in climb
    notes: tuple[str, ...]      # any caveats/notes

def recommend_climb_schedule(gw_lbs: float,
                             flap_after_cleanup: str = "UP",
                             accel_alt_ft_agl: float = 1000.0,
                             policy: str = "conservative",
                             below_10k_limit_kias: int = 250,
                             clean_climb_kias: int = 300,
                             mach_transition: float = 0.73) -> ClimbSchedule:
    """Return a climb schedule that respects regulatory speed limits and gives an efficiency-aware climb N1.

    policy:
      - "conservative": +2% margin above minimum AEO gradient (200 ft/nm) requirement
      - "practical":    +1% margin
    """
    flap_deg = _flap_label_to_deg(flap_after_cleanup)
    # Compute minimum N1 for AEO 200 ft/nm (3.3%) at the post-clean flap state.
    min_n1_aeo = _required_n1_for_gradient(gw_lbs, flap_deg, gradient_req=0.033, engines=2)

    # Add policy margin; apply floors/caps (UP/MAN >= 90%, FULL = 100%)
    margin = 2.0 if (policy or "conservative").lower().startswith("conserv") else 1.0
    floor = 100.0 if flap_deg == 40 else 90.0
    n1_cmd = max(floor, min(100.0, min_n1_aeo + margin))
    n1_cmd = int(round(n1_cmd))

    # Initial segment speed suggestion (pre-clean; informational in card)
    v2_plus = "V2+15"

    return ClimbSchedule(
        climb_n1 = n1_cmd,
        v2_plus = v2_plus,
        accel_alt_ft_agl = int(round(accel_alt_ft_agl)),
        limit_below_10k_kias = int(below_10k_limit_kias),
        clean_climb_kias = int(clean_climb_kias),
        mach_transition = float(mach_transition),
        notes = tuple([
            "Respect 250 KIAS below 10,000 MSL (unless ATC/LOA differs).",
            "At accel altitude: set climb N1, accelerate to 250 KIAS (if below 10k), then 300 KIAS and Mach hold.",
            "Increase N1 if hot/high or if climb rate is poor after cleanup."
        ])
    )

def format_climb_card(schedule: ClimbSchedule) -> str:
    """Return an HTML snippet suitable for a Streamlit markdown card."""
    html = f\"\"\"\n<div class='f14-card'>\n  <b>Climb Recommendation</b>\n  <div>Fly <b>{schedule.v2_plus}</b> to <b>{schedule.accel_alt_ft_agl} ft AGL</b>.</div>\n  <div>At {schedule.accel_alt_ft_agl} AGL (flaps UP), set <b>{schedule.climb_n1}% N1</b>.</div>\n  <div>Below 10,000 ft: <b>{schedule.limit_below_10k_kias} KIAS</b> max. Then <b>{schedule.clean_climb_kias} KIAS</b> and transition to <b>M {schedule.mach_transition:.2f}</b>.</div>\n  <ul>\n    {''.join(f'<li>{n}</li>' for n in schedule.notes)}\n  </ul>\n</div>\n\"\"\"\n    return html\n
