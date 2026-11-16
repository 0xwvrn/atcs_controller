"""Verification suite for MODERATO 3-regime control, flexible 2-phase topologies, and IRC:93 timing profiles."""

import sys
from pathlib import Path

_CONTROLLER_DIR = Path(__file__).resolve().parent.parent
if str(_CONTROLLER_DIR) not in sys.path:
    sys.path.insert(0, str(_CONTROLLER_DIR))

from adaptive_controller import (
    AdaptiveTrafficController,
    IRC93TimingProfile,
    IntersectionTopology,
    ModeratoRegime,
    SignalState,
)
from telemetry_schema import build_telemetry_packet, parse_and_validate_telemetry


def advance_until_state(ctrl: AdaptiveTrafficController, target: SignalState, refresh_fn=None):
    for _ in range(2000):
        if ctrl.state == target:
            return
        ctrl.tick(dt=0.1)
        if refresh_fn is not None:
            refresh_fn(ctrl)
    assert ctrl.state == target, f"Failed to reach {target}, stuck in {ctrl.state}"


def set_4way_demand(ctrl: AdaptiveTrafficController, n: float, s: float, e: float, w: float, inflow_ns: float = 0.0):
    ctrl.update_approach("N", int(n), n * 3.0, n, True, inflow_pcu=inflow_ns)
    ctrl.update_approach("S", int(s), s * 3.0, s, True, inflow_pcu=inflow_ns)
    ctrl.update_approach("E", int(e), e * 3.0, e, True)
    ctrl.update_approach("W", int(w), w * 3.0, w, True)


def test_1_moderato_light_traffic_delay_minimum():
    ctrl = AdaptiveTrafficController(startup_all_red_s=1.0, enable_moderato=True)
    set_4way_demand(ctrl, 4.0, 4.0, 2.0, 2.0, inflow_ns=1.25)
    advance_until_state(ctrl, SignalState.STATE_NS_GREEN)

    assert ctrl.active_regime == ModeratoRegime.DELAY_MINIMUM
    assert abs(ctrl.allocated_green_s - 11.5) < 0.05
    print(f"[PASS] Test 1 (DELAY_MINIMUM): alloc={ctrl.allocated_green_s:.1f}s")


def test_2_moderato_near_saturated_congestion_control_inflow():
    std_ctrl = AdaptiveTrafficController(startup_all_red_s=1.0, enable_moderato=False)
    mod_ctrl = AdaptiveTrafficController(startup_all_red_s=1.0, enable_moderato=True)

    for c in (std_ctrl, mod_ctrl):
        set_4way_demand(c, 9.0, 9.0, 5.0, 5.0, inflow_ns=3.0)
        advance_until_state(c, SignalState.STATE_NS_GREEN)

    assert mod_ctrl.active_regime == ModeratoRegime.CONGESTION_CONTROL
    assert abs(std_ctrl.allocated_green_s - 21.5) < 0.05
    assert abs(mod_ctrl.allocated_green_s - 27.5) < 0.05
    print(f"[PASS] Test 2 (CONGESTION_CONTROL): std={std_ctrl.allocated_green_s:.1f}s vs mod={mod_ctrl.allocated_green_s:.1f}s")


def test_3_moderato_asymmetric_step_down():
    mod_ctrl = AdaptiveTrafficController(startup_all_red_s=1.0, enable_moderato=True, moderato_max_step_down_s=10.0)
    set_4way_demand(mod_ctrl, 16.0, 16.0, 4.0, 4.0)
    advance_until_state(mod_ctrl, SignalState.STATE_NS_GREEN)
    assert abs(mod_ctrl.allocated_green_s - 35.5) < 0.05

    refresh_c2 = lambda c: set_4way_demand(c, 6.0, 6.0, 7.5, 7.5)
    advance_until_state(mod_ctrl, SignalState.STATE_EW_GREEN, refresh_fn=refresh_c2)
    advance_until_state(mod_ctrl, SignalState.STATE_NS_GREEN, refresh_fn=refresh_c2)

    assert abs(mod_ctrl.allocated_green_s - 25.5) < 0.05
    print(f"[PASS] Test 3 (Asymmetric Step-Down): 35.5s -> {mod_ctrl.allocated_green_s:.1f}s (raw 15.5s)")


def test_4_moderato_oversaturated_balanced_control():
    std_ctrl = AdaptiveTrafficController(startup_all_red_s=1.0, enable_moderato=False)
    mod_ctrl = AdaptiveTrafficController(startup_all_red_s=1.0, enable_moderato=True)
    refresh_gridlock = lambda c: set_4way_demand(c, 23.0, 23.0, 11.0, 11.0)

    for c in (std_ctrl, mod_ctrl):
        refresh_gridlock(c)
        advance_until_state(c, SignalState.STATE_NS_GREEN, refresh_fn=refresh_gridlock)
        advance_until_state(c, SignalState.STATE_EW_GREEN, refresh_fn=refresh_gridlock)

    assert mod_ctrl.active_regime == ModeratoRegime.BALANCED_CONTROL
    assert abs(std_ctrl.allocated_green_s - 25.5) < 0.05
    assert mod_ctrl.allocated_green_s > 36.0
    print(f"[PASS] Test 4 (BALANCED_CONTROL): std={std_ctrl.allocated_green_s:.1f}s vs mod={mod_ctrl.allocated_green_s:.1f}s")


def test_5_three_way_t_junction_topology():
    fixed_4way = AdaptiveTrafficController(startup_all_red_s=1.0, phase_approaches=IntersectionTopology.STANDARD_4WAY)
    flex_3way = AdaptiveTrafficController(startup_all_red_s=1.0, phase_approaches=IntersectionTopology.T_JUNCTION_3WAY_NS_E)

    def feed_3way(c: AdaptiveTrafficController, ns_pcu: float = 7.5):
        c.update_approach("N", int(ns_pcu), 25.0, ns_pcu, True)
        c.update_approach("S", int(ns_pcu), 25.0, ns_pcu, True)
        c.update_approach("E", 4, 16.0, 4.0, True)

    for c in (fixed_4way, flex_3way):
        feed_3way(c)
        advance_until_state(c, SignalState.STATE_NS_GREEN)

    for _ in range(105):
        for c in (fixed_4way, flex_3way):
            c.tick(dt=0.1)
            feed_3way(c)

    for c in (fixed_4way, flex_3way):
        feed_3way(c, ns_pcu=0.0)
        c.tick(dt=0.1)

    assert fixed_4way.state == SignalState.STATE_NS_GREEN
    assert flex_3way.state == SignalState.STATE_NS_YELLOW
    print("[PASS] Test 5 (3-Way T-Junction): early gap-out triggered without false 'W' dropout")


def test_6_six_way_two_phase_topology_and_conflict_rejection():
    ctrl_6way = AdaptiveTrafficController(
        startup_all_red_s=1.0,
        phase_approaches=IntersectionTopology.SIX_WAY_2PHASE,
        enable_moderato=True,
    )
    for leg, pcu in (("N", 6.0), ("S", 5.0), ("NE", 4.0), ("E", 3.0), ("W", 3.0), ("SW", 2.0)):
        pkt = build_telemetry_packet(leg, int(pcu), pcu, 20.0, True, inflow_pcu=1.0, allow_extended_approaches=True)
        ok, parsed, err = parse_and_validate_telemetry(pkt, allow_extended_approaches=True)
        assert ok, err
        ctrl_6way.update_approach(
            parsed["approach"], parsed["Q"], parsed["D_tail_m"], parsed["PCU"],
            is_valid=parsed["valid"], inflow_pcu=parsed.get("inflow_pcu", 0.0),
        )

    advance_until_state(ctrl_6way, SignalState.STATE_NS_GREEN)
    assert abs(ctrl_6way.allocated_green_s - 21.5) < 0.05

    try:
        AdaptiveTrafficController(phase_approaches=(("N", "S", "E"), ("E", "W")))
        assert False, "Expected ValueError on overlapping approaches"
    except ValueError:
        pass
    print(f"[PASS] Test 6 (6-Way 2-Phase): alloc={ctrl_6way.allocated_green_s:.1f}s & conflict rejected")


def test_7_irc93_timing_profiles_urban_arterial_and_heavy_metro():
    ctrl = AdaptiveTrafficController(startup_all_red_s=1.0, enable_moderato=True, timing_profile="URBAN_ARTERIAL")
    assert ctrl.active_timing_profile_name == IRC93TimingProfile.URBAN_ARTERIAL.name

    feed_urban = lambda c: set_4way_demand(c, 35.0, 30.0, 1.0, 1.0)
    feed_urban(ctrl)
    advance_until_state(ctrl, SignalState.STATE_NS_GREEN, refresh_fn=feed_urban)
    assert abs(ctrl.allocated_green_s - 60.0) < 0.05

    ns_start = ctrl.sim_time_s
    advance_until_state(ctrl, SignalState.STATE_NS_YELLOW, refresh_fn=feed_urban)
    assert abs((ctrl.sim_time_s - ns_start) - 60.0) < 0.25

    ctrl.apply_timing_profile(IRC93TimingProfile.HEAVY_METRO)
    assert ctrl.active_timing_profile_name == IRC93TimingProfile.HEAVY_METRO.name
    advance_until_state(ctrl, SignalState.STATE_EW_GREEN, refresh_fn=feed_urban)
    assert abs(ctrl.allocated_green_s - 15.0) < 0.05

    feed_metro = lambda c: set_4way_demand(c, 40.0, 40.0, 1.0, 1.0)
    advance_until_state(ctrl, SignalState.STATE_NS_GREEN, refresh_fn=feed_metro)
    assert abs(ctrl.allocated_green_s - 75.0) < 0.05

    ns_start_metro = ctrl.sim_time_s
    advance_until_state(ctrl, SignalState.STATE_NS_YELLOW, refresh_fn=feed_metro)
    assert abs((ctrl.sim_time_s - ns_start_metro) - 75.0) < 0.25
    print("[PASS] Test 7 (IRC:93 Profiles): URBAN_ARTERIAL (10-60s) & HEAVY_METRO (15-75s)")


def test_8_multi_stage_omit_and_double_cycling():
    # A -> B -> A -> C double-cycling: (("N", "S"), ("E",), ("N", "S"), ("W",)) with max_stage_skips=1
    ctrl = AdaptiveTrafficController(
        startup_all_red_s=1.0,
        phase_approaches=IntersectionTopology.DOUBLE_CYCLE_NS_E_NS_W,
        max_stage_skips=1,
    )
    # E has 0.0 PCU (empty), W has 8.0 PCU -> Cycle 1 omits E once and jumps NS(0) -> W(3); Cycle 2 must serve E(1)!
    feed = lambda c: set_4way_demand(c, 5.0, 5.0, 0.0, 8.0)
    feed(ctrl)

    advance_until_state(ctrl, SignalState.STATE_NS_GREEN, refresh_fn=feed)
    assert ctrl.stage_idx == 0  # Stage A: ("N", "S")

    advance_until_state(ctrl, SignalState.STATE_EW_GREEN, refresh_fn=feed)
    assert ctrl.stage_idx == 3  # Stage B ("E",) was omitted once -> jumped to Stage C ("W",)!

    advance_until_state(ctrl, SignalState.STATE_NS_GREEN, refresh_fn=feed)
    assert ctrl.stage_idx == 0  # Back to Stage A ("N", "S")

    advance_until_state(ctrl, SignalState.STATE_EW_GREEN, refresh_fn=feed)
    assert ctrl.stage_idx == 1  # Bounded skip limit reached (max_stage_skips=1) -> Stage B ("E",) guaranteed service!
    print(f"[PASS] Test 8 (Double-Cycling & Bounded Omit): history={ctrl.stage_history}")


def main():
    print("# ATCS Prototype Suite: MODERATO, Topologies & IRC:93 Timing Profiles")
    test_1_moderato_light_traffic_delay_minimum()
    test_2_moderato_near_saturated_congestion_control_inflow()
    test_3_moderato_asymmetric_step_down()
    test_4_moderato_oversaturated_balanced_control()
    test_5_three_way_t_junction_topology()
    test_6_six_way_two_phase_topology_and_conflict_rejection()
    test_7_irc93_timing_profiles_urban_arterial_and_heavy_metro()
    test_8_multi_stage_omit_and_double_cycling()
    print("All 8 prototype tests passed.")


if __name__ == "__main__":
    main()
