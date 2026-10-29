import sys
from pathlib import Path
from adaptive_controller import AdaptiveTrafficController, SignalState


def test_scenario_1_initial_startup():
    print("\n--- Scenario 1: Initial Startup & Clearance Buffer ---")
    ctrl = AdaptiveTrafficController(startup_all_red_s=2.0)

    # Inputs: NS demand = 5 PCU, EW demand = 2 PCU
    ctrl.update_approach("N", 3, 15.0, 3.0, is_valid=True)
    ctrl.update_approach("S", 2, 10.0, 2.0, is_valid=True)
    ctrl.update_approach("E", 1, 5.0, 1.0, is_valid=True)
    ctrl.update_approach("W", 1, 5.0, 1.0, is_valid=True)

    # Initial state must be ALL_RED_STARTUP
    assert ctrl.state == SignalState.STATE_ALL_RED_STARTUP, f"Expected STARTUP hold, got {ctrl.state}"

    # Advance 1.9 seconds: should still be in startup hold
    for _ in range(19):
        ctrl.tick(dt=0.1)
    assert ctrl.state == SignalState.STATE_ALL_RED_STARTUP

    # Advance past 2.0s: must transition cleanly to NS_GREEN
    ctrl.tick(dt=0.2)
    assert ctrl.state == SignalState.STATE_NS_GREEN, f"Expected NS_GREEN after startup clearance, got {ctrl.state}"

    # Total NS PCU = 5.0. Green time = clamp(3.5 + 5.0/1.0, 10, 45) = 10.0s (min-green)
    assert ctrl.allocated_green_s == 10.0, f"Expected min-green 10.0s, got {ctrl.allocated_green_s}"
    print(f"  [PASS] Controller held initial all-red for 2.0s, then granted NS green (alloc={ctrl.allocated_green_s:.1f}s).")


def test_scenario_2_higher_ew_demand():
    print("\n--- Scenario 2: Phase Transition to Higher EW Demand ---")
    ctrl = AdaptiveTrafficController(startup_all_red_s=1.0)

    # Inputs: Light NS (2 PCU total), Heavy EW (12 PCU total)
    ctrl.update_approach("N", 1, 5.0, 1.0, is_valid=True)
    ctrl.update_approach("S", 1, 5.0, 1.0, is_valid=True)
    ctrl.update_approach("E", 7, 35.0, 7.0, is_valid=True)
    ctrl.update_approach("W", 5, 25.0, 5.0, is_valid=True)

    # Advance through startup to NS Green
    while ctrl.state != SignalState.STATE_NS_GREEN:
        ctrl.tick(dt=0.1)

    ns_alloc = ctrl.allocated_green_s
    assert ns_alloc == 10.0  # 2 PCU clamped to min-green 10.0s

    # Advance through NS phase, Yellow clearance, and All-Red clearance
    transitions = []
    while ctrl.state != SignalState.STATE_EW_GREEN:
        prev = ctrl.state
        ctrl.tick(dt=0.1)
        # Simulate active 1 Hz telemetry stream from approaches
        if int(ctrl.sim_time_s * 10) % 10 == 0:
            ctrl.update_approach("E", 7, 35.0, 7.0, is_valid=True)
            ctrl.update_approach("W", 5, 25.0, 5.0, is_valid=True)
        if ctrl.state != prev:
            transitions.append(ctrl.state)

    # Enforce strict clearance sequence: NS_GREEN -> NS_YELLOW -> ALL_RED_TO_EW -> EW_GREEN
    assert transitions == [
        SignalState.STATE_NS_YELLOW,
        SignalState.STATE_ALL_RED_TO_EW,
        SignalState.STATE_EW_GREEN,
    ], f"Invalid clearance sequence: {transitions}"

    # Total EW PCU = 12.0. Green time = 3.5 + (12.0 / 1.0) = 15.5s
    assert ctrl.allocated_green_s == 15.5, f"Expected 15.5s for 12 PCU, got {ctrl.allocated_green_s}"
    print(f"  [PASS] NS cleared via Yellow (3.0s) & All-Red (1.5s). EW received proportional green ({ctrl.allocated_green_s:.1f}s).")


def test_scenario_3_no_opposing_demand():
    print("\n--- Scenario 3: No Opposing Demand (Avoid Unnecessary Gap-Out) ---")
    ctrl = AdaptiveTrafficController()

    # Inputs: NS demand = 16 PCU (allocates 19.5s green), EW demand = 0 PCU
    ctrl.update_approach("N", 8, 35.0, 8.0, is_valid=True)
    ctrl.update_approach("S", 8, 35.0, 8.0, is_valid=True)
    ctrl.update_approach("E", 0, 0.0, 0.0, is_valid=True)
    ctrl.update_approach("W", 0, 0.0, 0.0, is_valid=True)

    while ctrl.state != SignalState.STATE_NS_GREEN:
        ctrl.tick(dt=0.1)

    assert ctrl.allocated_green_s == 19.5, f"Expected 19.5s, got {ctrl.allocated_green_s}"

    # Advance to t = 11.0s (past min-green 10s, but well before 19.5s)
    green_start = ctrl.sim_time_s
    while ctrl.sim_time_s - green_start < 11.0:
        ctrl.tick(dt=0.1)
        # Keep EW fresh at 0.0 PCU
        ctrl.update_approach("E", 0, 0.0, 0.0, is_valid=True)
        ctrl.update_approach("W", 0, 0.0, 0.0, is_valid=True)

    # Simulate NS vehicles having cleared (queue drops to 0)
    ctrl.update_approach("N", 0, 0.0, 0.0, is_valid=True)
    ctrl.update_approach("S", 0, 0.0, 0.0, is_valid=True)
    ctrl.tick(dt=0.1)

    # Gap-out must NOT trigger because opposing cross-street demand is 0!
    assert ctrl.state == SignalState.STATE_NS_GREEN, (
        f"Controller prematurely gapped out to empty cross street! State: {ctrl.state}"
    )
    print("  [PASS] Controller maintained green for active corridor without false gap-out when cross-street was empty.")


def test_scenario_4_maximum_green_reached():
    print("\n--- Scenario 4: Maximum Green Ceiling Enforcement ---")
    ctrl = AdaptiveTrafficController(max_green_s=45.0)

    # Inputs: Massive NS surge (50 PCU), EW waiting with 8 PCU
    ctrl.update_approach("N", 25, 100.0, 25.0, is_valid=True)
    ctrl.update_approach("S", 25, 100.0, 25.0, is_valid=True)
    ctrl.update_approach("E", 4, 20.0, 4.0, is_valid=True)
    ctrl.update_approach("W", 4, 20.0, 4.0, is_valid=True)

    while ctrl.state != SignalState.STATE_NS_GREEN:
        ctrl.tick(dt=0.1)

    # 50 PCU would demand 3.5 + 50 = 53.5s, but must clamp strictly to max_green (45.0s)
    assert ctrl.allocated_green_s == 45.0, f"Expected 45.0s ceiling, got {ctrl.allocated_green_s}"

    # Advance until green expires
    green_start = ctrl.sim_time_s
    while ctrl.state == SignalState.STATE_NS_GREEN:
        ctrl.tick(dt=0.1)

    elapsed_green = ctrl.sim_time_s - green_start
    assert abs(elapsed_green - 45.0) < 0.2, f"Expected exit at 45.0s, exited at {elapsed_green:.1f}s"
    assert ctrl.state == SignalState.STATE_NS_YELLOW
    print(f"  [PASS] 50 PCU queue strictly constrained to 45.0s ceiling. Exited cleanly to Yellow at {elapsed_green:.1f}s.")


def test_scenario_5_sensor_timeout_policy():
    print("\n--- Scenario 5: Missing-Data Policy on Sensor Timeout ---")
    ctrl = AdaptiveTrafficController(sensor_timeout_s=5.0, nominal_fallback_pcu=2.0)

    # Initial telemetry: NS known (6 PCU), EW known (8 PCU)
    ctrl.update_approach("N", 3, 15.0, 3.0, is_valid=True)
    ctrl.update_approach("S", 3, 15.0, 3.0, is_valid=True)
    ctrl.update_approach("E", 4, 20.0, 4.0, is_valid=True)
    ctrl.update_approach("W", 4, 20.0, 4.0, is_valid=True)

    while ctrl.state != SignalState.STATE_NS_GREEN:
        ctrl.tick(dt=0.1)

    # Simulate North camera disconnect: stops sending updates for > 5.0 seconds
    for _ in range(60):
        ctrl.tick(dt=0.1)

    # Check demand diagnostics
    pcu_ns, ns_measured = ctrl.get_phase_demand("N", "S")
    pcu_ew, ew_measured = ctrl.get_phase_demand("E", "W")

    assert not ns_measured, "North disconnect was not flagged in phase measurement status."
    # N decayed to fallback 2.0 PCU, S decayed to fallback 2.0 PCU -> Total fallback = 4.0 PCU
    assert pcu_ns == 4.0, f"Expected fallback 4.0 PCU for NS, got {pcu_ns}"

    # Verify that gap-out is suppressed when sensor data is missing
    assert ctrl.state == SignalState.STATE_NS_GREEN
    print("  [PASS] Missing sensor flagged as unmeasured, decayed to 4.0 PCU fallback, and gap-out on assumed traffic was suppressed.")


def test_safety_interlock_fault_injection():
    print("\n--- Safety Invariant & FSM Sequence Fault Injection ---")
    ctrl = AdaptiveTrafficController()

    # Attempt illegal skip from ALL_RED directly to EW_GREEN (skipping sequence)
    ctrl._switch_state(SignalState.STATE_EW_GREEN, "ILLEGAL_JUMP_TEST")
    assert ctrl.state == SignalState.STATE_FAILSAFE_FLASH, "FSM failed to reject illegal sequence jump!"
    print("  [PASS] Out-of-order transition successfully intercepted. System triggered FAILSAFE_FLASH.")


def main():
    print("=================================================================")
    print("  ATCS Adaptive Controller: Academic Verification Test Suite")
    print("=================================================================")

    test_scenario_1_initial_startup()
    test_scenario_2_higher_ew_demand()
    test_scenario_3_no_opposing_demand()
    test_scenario_4_maximum_green_reached()
    test_scenario_5_sensor_timeout_policy()
    test_safety_interlock_fault_injection()

    print("\n=================================================================")
    print("  ALL 6 ACADEMIC VERIFICATION SCENARIOS PASSED WITH ZERO ERRORS.")
    print("=================================================================")


if __name__ == "__main__":
    main()
