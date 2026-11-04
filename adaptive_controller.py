import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple


class SignalState(Enum):
    STATE_ALL_RED_STARTUP = "ALL_RED_STARTUP"
    STATE_NS_GREEN = "NS_GREEN"
    STATE_NS_YELLOW = "NS_YELLOW"
    STATE_ALL_RED_TO_EW = "ALL_RED_TO_EW"
    STATE_EW_GREEN = "EW_GREEN"
    STATE_EW_YELLOW = "EW_YELLOW"
    STATE_ALL_RED_TO_NS = "ALL_RED_TO_NS"
    STATE_FAILSAFE_FLASH = "FAILSAFE_FLASH"


LEGAL_TRANSITIONS = {
    SignalState.STATE_ALL_RED_STARTUP: {SignalState.STATE_NS_GREEN, SignalState.STATE_FAILSAFE_FLASH},
    SignalState.STATE_NS_GREEN: {SignalState.STATE_NS_YELLOW, SignalState.STATE_FAILSAFE_FLASH},
    SignalState.STATE_NS_YELLOW: {SignalState.STATE_ALL_RED_TO_EW, SignalState.STATE_FAILSAFE_FLASH},
    SignalState.STATE_ALL_RED_TO_EW: {SignalState.STATE_EW_GREEN, SignalState.STATE_FAILSAFE_FLASH},
    SignalState.STATE_EW_GREEN: {SignalState.STATE_EW_YELLOW, SignalState.STATE_FAILSAFE_FLASH},
    SignalState.STATE_EW_YELLOW: {SignalState.STATE_ALL_RED_TO_NS, SignalState.STATE_FAILSAFE_FLASH},
    SignalState.STATE_ALL_RED_TO_NS: {SignalState.STATE_NS_GREEN, SignalState.STATE_FAILSAFE_FLASH},
    SignalState.STATE_FAILSAFE_FLASH: set(),
}


@dataclass
class ApproachDemand:
    leg: str
    q_count: int = 0
    d_tail_m: float = 0.0
    p_pcu: float = 0.0
    last_seen_s: float = 0.0
    is_valid: bool = False


class AdaptiveTrafficController:
    """Adaptive signal controller implementing NEMA TS2 clearance sequencing and dynamic green allocation."""

    def __init__(
        self,
        min_green_s: float = 10.0,
        max_green_s: float = 45.0,
        yellow_s: float = 3.0,
        all_red_s: float = 1.5,
        startup_all_red_s: float = 2.0,
        startup_lost_time_s: float = 3.5,
        discharge_rate_pcu_per_s: float = 0.5,
        max_wait_starvation_s: float = 50.0,
        sensor_timeout_s: float = 5.0,
        nominal_fallback_pcu: float = 2.0,
    ):
        self.min_green_s = min_green_s
        self.max_green_s = max_green_s
        self.yellow_s = yellow_s
        self.all_red_s = all_red_s
        self.startup_all_red_s = startup_all_red_s
        self.startup_lost_time_s = startup_lost_time_s
        self.discharge_rate_pcu_per_s = discharge_rate_pcu_per_s
        self.max_wait_starvation_s = max_wait_starvation_s
        self.sensor_timeout_s = sensor_timeout_s
        self.nominal_fallback_pcu = nominal_fallback_pcu

        self.state = SignalState.STATE_ALL_RED_STARTUP
        self.sim_time_s = 0.0
        self.state_start_time_s = 0.0
        self.allocated_green_s = self.min_green_s
        self.ns_wait_time_s = 0.0
        self.ew_wait_time_s = 0.0
        self.approaches: Dict[str, ApproachDemand] = {leg: ApproachDemand(leg=leg) for leg in ("N", "S", "E", "W")}
        self.transition_log = []

    def update_approach(
        self,
        leg: str,
        q_count: int,
        d_tail_m: float,
        p_pcu: float,
        is_valid: bool = True,
        timestamp_s: Optional[float] = None,
    ):
        if leg not in self.approaches:
            raise ValueError(f"Unknown approach leg: {leg}")

        for val, name in ((q_count, "q_count"), (d_tail_m, "d_tail_m"), (p_pcu, "p_pcu")):
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"Non-finite or invalid numeric value for {name}: {val}")

        if timestamp_s is not None and (
            isinstance(timestamp_s, bool) or not isinstance(timestamp_s, (int, float)) or not math.isfinite(timestamp_s)
        ):
            raise ValueError(f"Non-finite or invalid numeric value for timestamp_s: {timestamp_s}")

        app = self.approaches[leg]
        app.q_count = max(0, int(q_count))
        app.d_tail_m = max(0.0, float(d_tail_m))
        app.p_pcu = max(0.0, float(p_pcu))
        app.last_seen_s = self.sim_time_s if timestamp_s is None else float(timestamp_s)
        app.is_valid = is_valid

    def get_approach_demand(self, leg: str) -> Tuple[float, bool]:
        app = self.approaches[leg]
        fresh = (self.sim_time_s - app.last_seen_s) <= self.sensor_timeout_s
        return (app.p_pcu, True) if (app.is_valid and fresh) else (self.nominal_fallback_pcu, False)

    def get_phase_demand(self, leg1: str, leg2: str) -> Tuple[float, bool]:
        pcu1, m1 = self.get_approach_demand(leg1)
        pcu2, m2 = self.get_approach_demand(leg2)
        return pcu1 + pcu2, m1 and m2

    def calculate_green_time(self, phase_pcu: float, num_concurrent_legs: int = 2) -> float:
        rate = max(0.1, num_concurrent_legs * self.discharge_rate_pcu_per_s)
        return min(max(self.startup_lost_time_s + (phase_pcu / rate), self.min_green_s), self.max_green_s)

    def verify_safety_interlock(self) -> bool:
        ns = self.state in {SignalState.STATE_NS_GREEN, SignalState.STATE_NS_YELLOW}
        ew = self.state in {SignalState.STATE_EW_GREEN, SignalState.STATE_EW_YELLOW}
        return not (ns and ew)

    def _switch_state(self, new_state: SignalState, reason: str = ""):
        if new_state not in LEGAL_TRANSITIONS.get(self.state, set()):
            err = (
                f"CRITICAL SAFETY FAULT: Illegal transition attempted from {self.state.value} to {new_state.value}. "
                "Yellow or clearance was skipped! Enforcing immediate FAILSAFE_FLASH."
            )
            self.state = SignalState.STATE_FAILSAFE_FLASH
            self.state_start_time_s = self.sim_time_s
            self.transition_log.append({
                "time_s": round(self.sim_time_s, 2),
                "state": SignalState.STATE_FAILSAFE_FLASH.value,
                "reason": err,
            })
            return

        self.state = new_state
        self.state_start_time_s = self.sim_time_s
        self.transition_log.append({
            "time_s": round(self.sim_time_s, 2),
            "state": new_state.value,
            "reason": reason,
        })

    def tick(self, dt: float = 0.1):
        self.sim_time_s += dt
        elapsed_s = self.sim_time_s - self.state_start_time_s

        if not self.verify_safety_interlock():
            self._switch_state(SignalState.STATE_FAILSAFE_FLASH, "SAFETY INTERLOCK CONFLICT")
            return

        pcu_ns, ns_measured = self.get_phase_demand("N", "S")
        pcu_ew, ew_measured = self.get_phase_demand("E", "W")

        self.ew_wait_time_s = (self.ew_wait_time_s + dt) if (self.state != SignalState.STATE_EW_GREEN and pcu_ew >= 1.0) else 0.0
        self.ns_wait_time_s = (self.ns_wait_time_s + dt) if (self.state != SignalState.STATE_NS_GREEN and pcu_ns >= 1.0) else 0.0

        if self.state == SignalState.STATE_ALL_RED_STARTUP:
            if elapsed_s >= self.startup_all_red_s:
                self.allocated_green_s = self.calculate_green_time(pcu_ns)
                self.ns_wait_time_s = 0.0
                self._switch_state(
                    SignalState.STATE_NS_GREEN,
                    f"STARTUP_COMPLETE (alloc={self.allocated_green_s:.1f}s, NS_PCU={pcu_ns:.1f})"
                )

        elif self.state in (SignalState.STATE_NS_GREEN, SignalState.STATE_EW_GREEN):
            is_ns = self.state == SignalState.STATE_NS_GREEN
            active_pcu, opp_pcu = (pcu_ns, pcu_ew) if is_ns else (pcu_ew, pcu_ns)
            opp_measured = ns_measured and ew_measured
            opp_wait = self.ew_wait_time_s if is_ns else self.ns_wait_time_s
            next_yellow = SignalState.STATE_NS_YELLOW if is_ns else SignalState.STATE_EW_YELLOW

            gap_out = (elapsed_s >= self.min_green_s) and (active_pcu < 0.5) and (opp_pcu >= 1.0) if opp_measured else False
            starvation = (opp_wait >= self.max_wait_starvation_s) and (opp_pcu >= 1.0)
            max_green_expired = elapsed_s >= self.allocated_green_s

            if gap_out:
                self._switch_state(next_yellow, f"GAP_OUT (elapsed={elapsed_s:.1f}s)")
            elif starvation:
                tag = "ew_wait" if is_ns else "ns_wait"
                self._switch_state(next_yellow, f"ANTI_STARVATION ({tag}={opp_wait:.1f}s)")
            elif max_green_expired:
                self._switch_state(next_yellow, f"MAX_GREEN_EXPIRED (alloc={self.allocated_green_s:.1f}s)")

        elif self.state == SignalState.STATE_NS_YELLOW:
            if elapsed_s >= self.yellow_s:
                self._switch_state(SignalState.STATE_ALL_RED_TO_EW, "YELLOW_CLEARED")

        elif self.state == SignalState.STATE_EW_YELLOW:
            if elapsed_s >= self.yellow_s:
                self._switch_state(SignalState.STATE_ALL_RED_TO_NS, "YELLOW_CLEARED")

        elif self.state in (SignalState.STATE_ALL_RED_TO_EW, SignalState.STATE_ALL_RED_TO_NS):
            if elapsed_s >= self.all_red_s:
                to_ew = self.state == SignalState.STATE_ALL_RED_TO_EW
                next_green = SignalState.STATE_EW_GREEN if to_ew else SignalState.STATE_NS_GREEN
                pcu = pcu_ew if to_ew else pcu_ns
                tag = "EW" if to_ew else "NS"
                self.allocated_green_s = self.calculate_green_time(pcu)
                self._switch_state(next_green, f"ALLOCATED_GREEN={self.allocated_green_s:.1f}s ({tag}_PCU={pcu:.1f})")
