from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List


class SignalState(Enum):
    STATE_ALL_RED_STARTUP = "ALL_RED_STARTUP"
    STATE_NS_GREEN = "NS_GREEN"
    STATE_NS_YELLOW = "NS_YELLOW"
    STATE_ALL_RED_TO_EW = "ALL_RED_TO_EW"
    STATE_EW_GREEN = "EW_GREEN"
    STATE_EW_YELLOW = "EW_YELLOW"
    STATE_ALL_RED_TO_NS = "ALL_RED_TO_NS"
    STATE_FAILSAFE_FLASH = "FAILSAFE_FLASH"


# Legal Transition Graph: Enforces strict NEMA TS2 phase clearance sequencing
LEGAL_TRANSITIONS = {
    SignalState.STATE_ALL_RED_STARTUP: {SignalState.STATE_NS_GREEN, SignalState.STATE_FAILSAFE_FLASH},
    SignalState.STATE_NS_GREEN: {SignalState.STATE_NS_YELLOW, SignalState.STATE_FAILSAFE_FLASH},
    SignalState.STATE_NS_YELLOW: {SignalState.STATE_ALL_RED_TO_EW, SignalState.STATE_FAILSAFE_FLASH},
    SignalState.STATE_ALL_RED_TO_EW: {SignalState.STATE_EW_GREEN, SignalState.STATE_FAILSAFE_FLASH},
    SignalState.STATE_EW_GREEN: {SignalState.STATE_EW_YELLOW, SignalState.STATE_FAILSAFE_FLASH},
    SignalState.STATE_EW_YELLOW: {SignalState.STATE_ALL_RED_TO_NS, SignalState.STATE_FAILSAFE_FLASH},
    SignalState.STATE_ALL_RED_TO_NS: {SignalState.STATE_NS_GREEN, SignalState.STATE_FAILSAFE_FLASH},
    SignalState.STATE_FAILSAFE_FLASH: set(),  # Terminal failsafe requiring manual reset
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
    """
    Refined Adaptive Traffic Signal Controller.
    Features:
      - Total concurrent phase demand aggregation (N + S and E + W)
      - Explicit separation of measured vs. fallback demand
      - Missing-data policy (disables gap-out on assumed traffic)
      - Strict transition sequence enforcement
      - Dedicated startup all-red clearance buffer
    """

    def __init__(
        self,
        min_green_s: float = 10.0,
        max_green_s: float = 45.0,
        yellow_s: float = 3.0,
        all_red_s: float = 1.5,
        startup_all_red_s: float = 2.0,
        startup_lost_time_s: float = 3.5,
        discharge_rate_pcu_per_s: float = 0.5,  # ~1800 PCU/hr per lane
        max_wait_starvation_s: float = 65.0,
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

        # Initial state: Startup hold in All-Red
        self.state = SignalState.STATE_ALL_RED_STARTUP
        self.sim_time_s = 0.0
        self.state_start_time_s = 0.0
        self.allocated_green_s = self.min_green_s

        self.ns_red_start_time_s = 0.0
        self.ew_red_start_time_s = 0.0

        self.approaches: Dict[str, ApproachDemand] = {
            "N": ApproachDemand(leg="N"),
            "S": ApproachDemand(leg="S"),
            "E": ApproachDemand(leg="E"),
            "W": ApproachDemand(leg="W"),
        }

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

        t = timestamp_s if timestamp_s is not None else self.sim_time_s
        app = self.approaches[leg]
        app.q_count = max(0, q_count)
        app.d_tail_m = max(0.0, d_tail_m)
        app.p_pcu = max(0.0, p_pcu)
        app.last_seen_s = t
        app.is_valid = is_valid

    def get_approach_demand(self, leg: str) -> Tuple[float, bool]:
        """
        Returns (demand_pcu, is_measured).
        Distinguishes active camera measurements from sensor timeout fallbacks.
        """
        app = self.approaches[leg]
        fresh = (self.sim_time_s - app.last_seen_s) <= self.sensor_timeout_s
        if app.is_valid and fresh:
            return app.p_pcu, True
        # Explicit fallback demand
        return self.nominal_fallback_pcu, False

    def get_phase_demand(self, leg1: str, leg2: str) -> Tuple[float, bool]:
        """
        Computes combined demand served by a two-stage concurrent phase:
        Total Demand = Demand(leg1) + Demand(leg2).
        Flags whether both legs are actively measured.
        """
        pcu1, measured1 = self.get_approach_demand(leg1)
        pcu2, measured2 = self.get_approach_demand(leg2)
        total_pcu = pcu1 + pcu2
        all_measured = measured1 and measured2
        return total_pcu, all_measured

    def calculate_green_time(self, phase_pcu: float, num_concurrent_legs: int = 2) -> float:
        """
        Computes dynamic green allocation for a concurrent phase.
        phase_discharge_rate = num_concurrent_legs * discharge_rate_pcu_per_s
        g = startup_lost_time + (phase_pcu / phase_discharge_rate)
        bounded within [min_green, max_green].
        """
        effective_discharge_rate = max(0.1, num_concurrent_legs * self.discharge_rate_pcu_per_s)
        raw_green = self.startup_lost_time_s + (phase_pcu / effective_discharge_rate)
        return min(max(raw_green, self.min_green_s), self.max_green_s)

    def verify_safety_interlock(self) -> bool:
        """
        Strict hardware safety invariant: Conflicting corridors can NEVER both be active.
        """
        ns_active = self.state in {SignalState.STATE_NS_GREEN, SignalState.STATE_NS_YELLOW}
        ew_active = self.state in {SignalState.STATE_EW_GREEN, SignalState.STATE_EW_YELLOW}
        return not (ns_active and ew_active)

    def _switch_state(self, new_state: SignalState, reason: str = ""):
        # Sequence validator: Enforce NEMA TS2 legal transition graph
        valid_next_states = LEGAL_TRANSITIONS.get(self.state, set())
        if new_state not in valid_next_states:
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
        """
        Advances the controller clock by dt seconds and evaluates state transitions.
        """
        self.sim_time_s += dt
        elapsed_s = self.sim_time_s - self.state_start_time_s

        # Hardware safety invariant verification
        if not self.verify_safety_interlock():
            self._switch_state(SignalState.STATE_FAILSAFE_FLASH, "SAFETY INTERLOCK CONFLICT")
            return

        # Phase demands (Combined N+S and E+W)
        pcu_ns, ns_measured = self.get_phase_demand("N", "S")
        pcu_ew, ew_measured = self.get_phase_demand("E", "W")

        # 1. Startup All-Red Hold
        if self.state == SignalState.STATE_ALL_RED_STARTUP:
            if elapsed_s >= self.startup_all_red_s:
                self.allocated_green_s = self.calculate_green_time(pcu_ns)
                self.ns_red_start_time_s = self.sim_time_s
                self._switch_state(
                    SignalState.STATE_NS_GREEN,
                    f"STARTUP_COMPLETE (alloc={self.allocated_green_s:.1f}s, NS_PCU={pcu_ns:.1f})"
                )

        # 2. North-South Green Phase
        elif self.state == SignalState.STATE_NS_GREEN:
            # Missing-Data Policy: Only execute gap-out if opposing demand is genuinely measured
            # Assumed/fallback traffic must never trigger premature termination of an active corridor
            gap_out = False
            if ns_measured and ew_measured:
                gap_out = (elapsed_s >= self.min_green_s) and (pcu_ns < 0.5) and (pcu_ew >= 1.0)

            # Anti-Starvation Protection (Cross-street wait ceiling)
            ew_wait_s = self.sim_time_s - self.ew_red_start_time_s
            starvation = (ew_wait_s >= self.max_wait_starvation_s) and (pcu_ew >= 1.0)

            # Maximum Green Expiration
            max_green_expired = elapsed_s >= self.allocated_green_s

            if gap_out:
                self._switch_state(SignalState.STATE_NS_YELLOW, f"GAP_OUT (elapsed={elapsed_s:.1f}s)")
            elif starvation:
                self._switch_state(SignalState.STATE_NS_YELLOW, f"ANTI_STARVATION (ew_wait={ew_wait_s:.1f}s)")
            elif max_green_expired:
                self._switch_state(SignalState.STATE_NS_YELLOW, f"MAX_GREEN_EXPIRED (alloc={self.allocated_green_s:.1f}s)")

        # 3. North-South Yellow Clearance Phase
        elif self.state == SignalState.STATE_NS_YELLOW:
            if elapsed_s >= self.yellow_s:
                self._switch_state(SignalState.STATE_ALL_RED_TO_EW, "YELLOW_CLEARED")

        # 4. All-Red Clearance to East-West
        elif self.state == SignalState.STATE_ALL_RED_TO_EW:
            if elapsed_s >= self.all_red_s:
                self.allocated_green_s = self.calculate_green_time(pcu_ew)
                self.ew_red_start_time_s = self.sim_time_s
                self._switch_state(
                    SignalState.STATE_EW_GREEN,
                    f"ALLOCATED_GREEN={self.allocated_green_s:.1f}s (EW_PCU={pcu_ew:.1f})"
                )

        # 5. East-West Green Phase
        elif self.state == SignalState.STATE_EW_GREEN:
            gap_out = False
            if ew_measured and ns_measured:
                gap_out = (elapsed_s >= self.min_green_s) and (pcu_ew < 0.5) and (pcu_ns >= 1.0)

            ns_wait_s = self.sim_time_s - self.ns_red_start_time_s
            starvation = (ns_wait_s >= self.max_wait_starvation_s) and (pcu_ns >= 1.0)
            max_green_expired = elapsed_s >= self.allocated_green_s

            if gap_out:
                self._switch_state(SignalState.STATE_EW_YELLOW, f"GAP_OUT (elapsed={elapsed_s:.1f}s)")
            elif starvation:
                self._switch_state(SignalState.STATE_EW_YELLOW, f"ANTI_STARVATION (ns_wait={ns_wait_s:.1f}s)")
            elif max_green_expired:
                self._switch_state(SignalState.STATE_EW_YELLOW, f"MAX_GREEN_EXPIRED (alloc={self.allocated_green_s:.1f}s)")

        # 6. East-West Yellow Clearance Phase
        elif self.state == SignalState.STATE_EW_YELLOW:
            if elapsed_s >= self.yellow_s:
                self._switch_state(SignalState.STATE_ALL_RED_TO_NS, "YELLOW_CLEARED")

        # 7. All-Red Clearance to North-South
        elif self.state == SignalState.STATE_ALL_RED_TO_NS:
            if elapsed_s >= self.all_red_s:
                self.allocated_green_s = self.calculate_green_time(pcu_ns)
                self.ns_red_start_time_s = self.sim_time_s
                self._switch_state(
                    SignalState.STATE_NS_GREEN,
                    f"ALLOCATED_GREEN={self.allocated_green_s:.1f}s (NS_PCU={pcu_ns:.1f})"
                )
