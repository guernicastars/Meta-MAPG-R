"""Five learning-rule arms from the engineering plan §2."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArmConfig:
    name: str
    label: str
    lam_peer: float
    lam_own: float
    is_handoff: bool = False


# Defaults match plan §4 ("conservative clipping"):
ARMS: dict[str, ArmConfig] = {
    "ippo":      ArmConfig("ippo",      "IPPO / PG",                lam_peer=0.0, lam_own=0.0),
    "own_only":  ArmConfig("own_only",  "Own-only",                 lam_peer=0.0, lam_own=0.5),
    "peer_only": ArmConfig("peer_only", "Peer-only",                lam_peer=1.0, lam_own=0.0),
    "meta_mapg": ArmConfig("meta_mapg", "Meta-MAPG",                lam_peer=1.0, lam_own=0.5),
    "handoff":   ArmConfig("handoff",   "Meta-MAPG warm-up → IPPO", lam_peer=1.0, lam_own=0.5,
                            is_handoff=True),
}


def get_arm_coefficients(arm: str, *, current_step: int, T_warm: int) -> tuple[float, float]:
    """Return (lam_peer, lam_own) at the current training step.

    The handoff arm (§2 Arm E) flips coefficients to zero once the warm-up
    budget elapses — the most important neural experiment per plan §6.5.
    """
    cfg = ARMS[arm]
    if cfg.is_handoff and current_step >= T_warm:
        return 0.0, 0.0
    return cfg.lam_peer, cfg.lam_own
