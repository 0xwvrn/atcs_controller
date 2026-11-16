import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Sequence, Tuple, Union


class SignalState(Enum):
    STATE_ALL_RED_STARTUP = "ALL_RED_STARTUP"
    STATE_NS_GREEN = "NS_GREEN"
    STATE_NS_YELLOW = "NS_YELLOW"
    STATE_ALL_RED_TO_EW = "ALL_RED_TO_EW"
    STATE_EW_GREEN = "EW_GREEN"
    STATE_EW_YELLOW = "EW_YELLOW"
    STATE_ALL_RED_TO_NS = "ALL_RED_TO_NS"
    STATE_FAILSAFE_FLASH = "FAILSAFE_FLASH"


class ModeratoRegime(Enum):
    """MODERATO congestion control regimes based on intersection Load Ratio (Hayashi & Sugimoto, IEEE 1999)."""
    DELAY_MINIMUM = "DELAY_MINIMUM"          # Under-saturated (L_total < 0.50): queue-only short cycle
    CONGESTION_CONTROL = "CONGESTION_CONTROL"  # Near-saturated (0.50 <= L_total <= 1.00): queue + inflow + step-down cap
    BALANCED_CONTROL = "BALANCED_CONTROL"    # Over-saturated (L_total > 1.00): balanced degree-of-saturation & wait split


class IntersectionTopology:
    """
    Intersection stage ring topologies (2-phase, 3-stage, and A->B->A->C double-cycling).

    Firmware-supported topologies (must match firmware/include/TrafficController.hpp):
      - STANDARD_4WAY
      - T_JUNCTION_3WAY_NS_E
      - T_JUNCTION_3WAY_EW_N
      - DOUBLE_CYCLE_NS_E_NS_W

    Simulation-only experimental topologies (not supported in ESP32-S3 firmware):
      - SIX_WAY_2PHASE
      - THREE_STAGE_NS_E_W
    """
    STANDARD_4WAY: Tuple[Tuple[str, ...], ...] = (("N", "S"), ("E", "W"))
    T_JUNCTION_3WAY_NS_E: Tuple[Tuple[str, ...], ...] = (("N", "S"), ("E",))
    T_JUNCTION_3WAY_EW_N: Tuple[Tuple[str, ...], ...] = (("E", "W"), ("N",))
    SIX_WAY_2PHASE: Tuple[Tuple[str, ...], ...] = (("N", "S", "NE"), ("E", "W", "SW"))
    THREE_STAGE_NS_E_W: Tuple[Tuple[str, ...], ...] = (("N", "S"), ("E",), ("W",))
    DOUBLE_CYCLE_NS_E_NS_W: Tuple[Tuple[str, ...], ...] = (("N", "S"), ("E",), ("N", "S"), ("W",))


@dataclass(frozen=True)
class TimingProfileSpec:
    """Signal timing quantum profile (in seconds) with coupled max_green and anti-starvation ceiling."""
    name: str
    min_green_s: float
    max_green_s: float
    yellow_s: float
    all_red_s: float
    max_wait_starvation_s: float
    fixed_fallback_green_s: float


class IRC93TimingProfile:
    """Selectable IRC:93-1985 / Indo-HCM (2017) signal timing profiles."""
    BASELINE_45S = TimingProfileSpec("BASELINE_45S", 10.0, 45.0, 3.0, 1.5, 50.0, 25.0)
    URBAN_ARTERIAL = TimingProfileSpec("URBAN_ARTERIAL", 10.0, 60.0, 3.0, 1.5, 65.0, 30.0)
    HEAVY_METRO = TimingProfileSpec("HEAVY_METRO", 15.0, 75.0, 4.0, 2.0, 82.0, 35.0)

    _PROFILES = {
        "BASELINE": BASELINE_45S,
        "BASELINE_45S": BASELINE_45S,
        "URBAN_ARTERIAL": URBAN_ARTERIAL,
        "HEAVY_METRO": HEAVY_METRO,
    }

    @classmethod
    def resolve(cls, profile: Union[str, TimingProfileSpec]) -> TimingProfileSpec:
        if isinstance(profile, TimingProfileSpec):
            return profile
        key = str(profile).strip().upper()
        if key not in cls._PROFILES:
            raise ValueError(f"Unknown timing profile '{profile}'. Valid: {sorted(cls._PROFILES)}")
        return cls._PROFILES[key]


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
    inflow_pcu: float = 0.0
    last_seen_s: float = 0.0
    is_valid: bool = False


class AdaptiveTrafficController:
    """Adaptive signal controller implementing NEMA TS2 clearance sequencing, dynamic topologies, and MODERATO regimes."""

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
        phase_approaches: Sequence[Sequence[str]] = IntersectionTopology.STANDARD_4WAY,
        enable_moderato: bool = False,
        moderato_max_step_down_s: float = 10.0,
        timing_profile: Optional[Union[str, TimingProfileSpec]] = None,
        max_stage_skips: int = 1,
    ):
        self.min_green_s = min_green_s
        self.max_green_s = max_green_s
        self.yellow_s = yellow_s
        self.all_red_s = all_red_s
        self.startup_all_red_s = startup_all_red_s
        self.startup_lost_time_s = startup_lost_time_s
        self.discharge_rate_pcu_per_s = discharge_rate_pcu_per_s
        self.max_wait_starvation_s = max_wait_starvation_s
        self.fixed_fallback_green_s = 25.0
        self.sensor_timeout_s = sensor_timeout_s
        self.nominal_fallback_pcu = nominal_fallback_pcu
        self.active_timing_profile_name = IRC93TimingProfile.BASELINE_45S.name

        if timing_profile is not None:
            self.apply_timing_profile(timing_profile)

        if len(phase_approaches) < 2 or any(not st for st in phase_approaches):
            raise ValueError("phase_approaches must contain at least 2 non-empty stage sequences")
        stages = tuple(tuple(str(leg).upper() for leg in st) for st in phase_approaches)
        n_stages = len(stages)
        for i in range(n_stages):
            nxt = stages[(i + 1) % n_stages]
            if set(stages[i]) & set(nxt):
                raise ValueError(f"Conflicting adjacent stage configuration: {set(stages[i]) & set(nxt)}")

        self.phase_approaches: Tuple[Tuple[str, ...], ...] = stages
        self.max_stage_skips = max(0, int(max_stage_skips))
        self.stage_idx: int = 0
        self.next_stage_idx: int = 0
        self.stage_skip_counts: Dict[Tuple[str, ...], int] = {st: 0 for st in set(stages)}
        self.stage_history: list = []

        self.enable_moderato = bool(enable_moderato)
        self.moderato_max_step_down_s = float(moderato_max_step_down_s)
        self.active_regime: ModeratoRegime = ModeratoRegime.DELAY_MINIMUM
        self.prev_phase_green_s: Dict[int, Optional[float]] = {i: None for i in range(n_stages)}

        self.state = SignalState.STATE_ALL_RED_STARTUP
        self.sim_time_s = 0.0
        self.state_start_time_s = 0.0
        self.allocated_green_s = self.min_green_s
        self.ns_wait_time_s = 0.0
        self.ew_wait_time_s = 0.0

        flat_legs = [leg for st in stages for leg in st]
        all_legs = list(dict.fromkeys((*flat_legs, "N", "S", "E", "W")))
        self.approaches: Dict[str, ApproachDemand] = {leg: ApproachDemand(leg=leg) for leg in all_legs}
        self.transition_log = []

    def apply_timing_profile(self, profile: Union[str, TimingProfileSpec]) -> TimingProfileSpec:
        """
        Switches the controller's timing quantums and coupled anti-starvation ceiling
        between IRC:93-1985 URBAN_ARTERIAL (60s), HEAVY_METRO (75s), or BASELINE_45S (45s).
        """
        spec = IRC93TimingProfile.resolve(profile)
        self.min_green_s = spec.min_green_s
        self.max_green_s = spec.max_green_s
        self.yellow_s = spec.yellow_s
        self.all_red_s = spec.all_red_s
        self.max_wait_starvation_s = spec.max_wait_starvation_s
        self.fixed_fallback_green_s = spec.fixed_fallback_green_s
        self.active_timing_profile_name = spec.name
        return spec

    def update_approach(
        self,
        leg: str,
        q_count: int,
        d_tail_m: float,
        p_pcu: float,
        is_valid: bool = True,
        timestamp_s: Optional[float] = None,
        inflow_pcu: float = 0.0,
    ):
        if leg not in self.approaches:
            raise ValueError(f"Unknown approach leg: {leg}")

        for val, name in ((q_count, "q_count"), (d_tail_m, "d_tail_m"), (p_pcu, "p_pcu"), (inflow_pcu, "inflow_pcu")):
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0:
                raise ValueError(f"Negative, non-finite, or invalid numeric value for {name}: {val}")

        if timestamp_s is not None and (
            isinstance(timestamp_s, bool) or not isinstance(timestamp_s, (int, float)) or not math.isfinite(timestamp_s)
        ):
            raise ValueError(f"Non-finite or invalid numeric value for timestamp_s: {timestamp_s}")

        app = self.approaches[leg]
        if not is_valid:
            app.is_valid = False
            return

        app.q_count = int(q_count)
        app.d_tail_m = float(d_tail_m)
        app.p_pcu = float(p_pcu)
        app.inflow_pcu = float(inflow_pcu)
        app.last_seen_s = self.sim_time_s if timestamp_s is None else float(timestamp_s)
        app.is_valid = True

    def get_approach_demand(self, leg: str) -> Tuple[float, bool]:
        app = self.approaches[leg]
        fresh = (self.sim_time_s - app.last_seen_s) <= self.sensor_timeout_s
        return (app.p_pcu, True) if (app.is_valid and fresh) else (self.nominal_fallback_pcu, False)

    def get_approach_inflow(self, leg: str) -> float:
        app = self.approaches[leg]
        fresh = (self.sim_time_s - app.last_seen_s) <= self.sensor_timeout_s
        return app.inflow_pcu if (app.is_valid and fresh) else 0.0

    def get_stage_demand(self, stage_idx: int) -> Tuple[float, bool]:
        total_pcu = 0.0
        all_measured = True
        for leg in self.phase_approaches[stage_idx]:
            pcu, m = self.get_approach_demand(leg)
            total_pcu += pcu
            all_measured = all_measured and m
        return total_pcu, all_measured

    def get_phase_demand(self, leg1: Optional[str] = None, leg2: Optional[str] = None) -> Tuple[float, bool]:
        if leg1 is not None and leg2 is not None and self.phase_approaches == IntersectionTopology.STANDARD_4WAY:
            pcu1, m1 = self.get_approach_demand(leg1)
            pcu2, m2 = self.get_approach_demand(leg2)
            return pcu1 + pcu2, m1 and m2
        idx = 0 if leg1 in (None, "N") else (self.stage_idx if not self._is_ns_stage(self.stage_idx) else 1)
        return self.get_stage_demand(idx)

    def _is_ns_stage(self, stage_idx: int) -> bool:
        return any(leg in ("N", "S") for leg in self.phase_approaches[stage_idx])

    def _select_next_stage_idx(self) -> int:
        n_stages = len(self.phase_approaches)
        if n_stages == 2:
            return (self.stage_idx + 1) % 2

        curr_legs = self.phase_approaches[self.stage_idx]
        cand = (self.stage_idx + 1) % n_stages
        for _ in range(n_stages - 1):
            cand_legs = self.phase_approaches[cand]
            cand_pcu, cand_measured = self.get_stage_demand(cand)
            if (
                self.max_stage_skips > 0
                and cand_measured
                and cand_pcu < 0.5
                and self.stage_skip_counts[cand_legs] < self.max_stage_skips
            ):
                self.stage_skip_counts[cand_legs] += 1
                cand = (cand + 1) % n_stages
                if self.phase_approaches[cand] == curr_legs:
                    cand = (cand + 1) % n_stages
            else:
                break

        self.stage_skip_counts[self.phase_approaches[cand]] = 0
        return cand

    def get_phase_inflow(self, phase_idx: int) -> float:
        return sum(self.get_approach_inflow(leg) for leg in self.phase_approaches[phase_idx])

    def get_phase_discharge_rate(self, num_concurrent_legs: int = 2) -> float:
        return max(0.1, num_concurrent_legs * self.discharge_rate_pcu_per_s)

    def get_phase_max_capacity_pcu(self, num_concurrent_legs: int = 2) -> float:
        rate = self.get_phase_discharge_rate(num_concurrent_legs=num_concurrent_legs)
        return max(1.0, (self.max_green_s - self.startup_lost_time_s) * rate)

    def evaluate_moderato_regime(self) -> Tuple[ModeratoRegime, float, float, float]:
        """
        Computes MODERATO Load Ratios (L_0, L_1, L_total) and classifies the congestion regime:
          - L_total < 0.50: DELAY_MINIMUM (Under-saturated light traffic)
          - 0.50 <= L_total <= 1.00: CONGESTION_CONTROL (Near-saturated rush hour)
          - L_total > 1.00: BALANCED_CONTROL (Over-saturated gridlock)
        """
        q0, _ = self.get_phase_demand("N", "S")
        q1, _ = self.get_phase_demand("E", "W")
        in0 = self.get_phase_inflow(0)
        in1 = self.get_phase_inflow(1)

        cap = self.get_phase_max_capacity_pcu(num_concurrent_legs=2)
        l0 = (q0 + in0) / cap
        l1 = (q1 + in1) / cap
        l_total = l0 + l1

        if l_total < 0.50:
            regime = ModeratoRegime.DELAY_MINIMUM
        elif l_total <= 1.00:
            regime = ModeratoRegime.CONGESTION_CONTROL
        else:
            regime = ModeratoRegime.BALANCED_CONTROL

        self.active_regime = regime
        return regime, l0, l1, l_total

    def calculate_green_time(self, phase_pcu: float, num_concurrent_legs: int = 2) -> float:
        rate = self.get_phase_discharge_rate(num_concurrent_legs=num_concurrent_legs)
        return min(max(self.startup_lost_time_s + (phase_pcu / rate), self.min_green_s), self.max_green_s)

    def is_complete_sensor_blackout(self) -> bool:
        """True when every configured approach leg is unobserved, matching C++ MODE_FIXED_FALLBACK."""
        configured_legs = {leg for stage in self.phase_approaches for leg in stage}
        return all(not self.get_approach_demand(leg)[1] for leg in configured_legs)

    def _allocate_phase_green(self, phase_idx: int) -> Tuple[float, str]:
        """Calculates green duration for stage index phase_idx using fallback, standard, or MODERATO rules."""
        if self.is_complete_sensor_blackout():
            return self.fixed_fallback_green_s, f"FIXED_FALLBACK ({self.active_timing_profile_name})"

        phase_pcu, _ = self.get_stage_demand(phase_idx)
        tag = "+".join(self.phase_approaches[phase_idx]) if len(self.phase_approaches) > 2 else ("NS" if phase_idx == 0 else "EW")

        if not self.enable_moderato:
            alloc = self.calculate_green_time(phase_pcu)
            self.prev_phase_green_s[phase_idx] = alloc
            return alloc, f"{tag}_PCU={phase_pcu:.1f}"

        regime, l0, l1, l_total = self.evaluate_moderato_regime()
        phase_inflow = self.get_phase_inflow(phase_idx)

        if regime == ModeratoRegime.DELAY_MINIMUM:
            alloc = self.calculate_green_time(phase_pcu)
            detail = f"MODERATO:DELAY_MIN, {tag}_PCU={phase_pcu:.1f}, L={l_total:.2f}"

        elif regime == ModeratoRegime.CONGESTION_CONTROL:
            effective_pcu = phase_pcu + phase_inflow
            raw_alloc = self.calculate_green_time(effective_pcu)
            prev_alloc = self.prev_phase_green_s.get(phase_idx)
            if prev_alloc is not None and raw_alloc < prev_alloc - self.moderato_max_step_down_s:
                alloc = max(self.min_green_s, prev_alloc - self.moderato_max_step_down_s)
            else:
                alloc = raw_alloc
            detail = f"MODERATO:CONGESTION, {tag}_PCU={phase_pcu:.1f}+{phase_inflow:.1f}in, L={l_total:.2f}"

        else:
            pcu_0, _ = self.get_stage_demand(0)
            pcu_1, _ = self.get_stage_demand(1)
            inflow_0 = self.get_phase_inflow(0)
            inflow_1 = self.get_phase_inflow(1)
            wait_0 = self.ns_wait_time_s
            wait_1 = self.ew_wait_time_s
            score_0 = max(0.05, l0 + (wait_0 / max(1.0, self.max_wait_starvation_s)))
            score_1 = max(0.05, l1 + (wait_1 / max(1.0, self.max_wait_starvation_s)))

            raw_g0 = self.startup_lost_time_s + (pcu_0 + inflow_0) / self.get_phase_discharge_rate(2)
            raw_g1 = self.startup_lost_time_s + (pcu_1 + inflow_1) / self.get_phase_discharge_rate(2)
            cycle_green_budget = min(2.0 * self.max_green_s, max(2.0 * self.min_green_s, raw_g0 + raw_g1))

            share = (score_0 if self._is_ns_stage(phase_idx) else score_1) / (score_0 + score_1)
            alloc = min(max(cycle_green_budget * share, self.min_green_s), self.max_green_s)
            detail = f"MODERATO:BALANCED, {tag}_PCU={phase_pcu:.1f}, L={l_total:.2f}, share={share:.2f}"

        self.prev_phase_green_s[phase_idx] = alloc
        return alloc, detail

    def verify_safety_interlock(self) -> bool:
        ns = self.state in {SignalState.STATE_NS_GREEN, SignalState.STATE_NS_YELLOW}
        ew = self.state in {SignalState.STATE_EW_GREEN, SignalState.STATE_EW_YELLOW}
        return not (ns and ew)

    def _switch_state(self, new_state: SignalState, reason: str = ""):
        legal = set(LEGAL_TRANSITIONS.get(self.state, set()))
        if len(self.phase_approaches) > 2:
            if self.state == SignalState.STATE_EW_YELLOW:
                legal.add(SignalState.STATE_ALL_RED_TO_EW)
            elif self.state == SignalState.STATE_NS_YELLOW:
                legal.add(SignalState.STATE_ALL_RED_TO_NS)

        if new_state not in legal:
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
                self.stage_idx = 0
                self.stage_history.append(self.phase_approaches[0])
                self.allocated_green_s, detail = self._allocate_phase_green(0)
                self.ns_wait_time_s = 0.0
                self._switch_state(
                    SignalState.STATE_NS_GREEN,
                    f"STARTUP_COMPLETE (alloc={self.allocated_green_s:.1f}s, {detail})"
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

            if gap_out or starvation or max_green_expired:
                self.next_stage_idx = self._select_next_stage_idx()
                if gap_out:
                    self._switch_state(next_yellow, f"GAP_OUT (elapsed={elapsed_s:.1f}s)")
                elif starvation:
                    tag = "ew_wait" if is_ns else "ns_wait"
                    self._switch_state(next_yellow, f"ANTI_STARVATION ({tag}={opp_wait:.1f}s)")
                else:
                    self._switch_state(next_yellow, f"MAX_GREEN_EXPIRED (alloc={self.allocated_green_s:.1f}s)")

        elif self.state in (SignalState.STATE_NS_YELLOW, SignalState.STATE_EW_YELLOW):
            if elapsed_s >= self.yellow_s:
                if len(self.phase_approaches) == 2:
                    target_ar = SignalState.STATE_ALL_RED_TO_EW if self.state == SignalState.STATE_NS_YELLOW else SignalState.STATE_ALL_RED_TO_NS
                else:
                    target_ar = SignalState.STATE_ALL_RED_TO_NS if self._is_ns_stage(self.next_stage_idx) else SignalState.STATE_ALL_RED_TO_EW
                self._switch_state(target_ar, "YELLOW_CLEARED")

        elif self.state in (SignalState.STATE_ALL_RED_TO_EW, SignalState.STATE_ALL_RED_TO_NS):
            if elapsed_s >= self.all_red_s:
                to_ew = self.state == SignalState.STATE_ALL_RED_TO_EW
                next_green = SignalState.STATE_EW_GREEN if to_ew else SignalState.STATE_NS_GREEN
                self.stage_idx = (1 if to_ew else 0) if len(self.phase_approaches) == 2 else self.next_stage_idx
                self.stage_history.append(self.phase_approaches[self.stage_idx])
                self.allocated_green_s, detail = self._allocate_phase_green(self.stage_idx)
                self._switch_state(next_green, f"ALLOCATED_GREEN={self.allocated_green_s:.1f}s ({detail})")
