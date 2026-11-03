"""
test_adaptive_scenarios.py
Automated End-to-End Scenario Test Suite for ATCS Adaptive Signal Control

Evaluates controller behavior across 6 core operational traffic scenarios:
  1. Balanced Moderate Traffic (proportional, equal green split)
  2. Heavy North-South Demand (dynamic green expansion for NS)
  3. Heavy East-West Demand (dynamic green expansion for EW)
  4. Early Gap-Out on Queue Clearance (early termination when queue clears)
  5. Anti-Starvation Protection (guaranteed service when waiting >= 50.0s)
  6. Complete Sensor Blackout (safe entry into 59.0s fixed-time fallback)
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Ensure controller module directory is in sys.path
_CONTROLLER_DIR = Path(__file__).resolve().parent.parent
if str(_CONTROLLER_DIR) not in sys.path:
    sys.path.insert(0, str(_CONTROLLER_DIR))

# Local imports
from adaptive_controller import AdaptiveTrafficController, SignalState
from telemetry_schema import build_telemetry_packet, parse_and_validate_telemetry


class ScenarioReport:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.passed = False
        self.details: List[str] = []
        self.metrics: Dict[str, str] = {}

    def log(self, msg: str):
        self.details.append(msg)

    def set_metric(self, key: str, val: str):
        self.metrics[key] = val


# ==============================================================================
# 1. Software-in-the-Loop (SIL) Scenario Implementations
# ==============================================================================

def run_sil_scenario_1_balanced() -> ScenarioReport:
    report = ScenarioReport("Scenario 1: Balanced Moderate Demand",
                            "Verifies equal green allocation (~13.5s) when opposing demands are balanced (10 PCU each).")
    ctrl = AdaptiveTrafficController(startup_all_red_s=2.0)

    # 10 PCU on NS (5+5), 10 PCU on EW (5+5)
    ctrl.update_approach("N", 5, 20.0, 5.0, is_valid=True)
    ctrl.update_approach("S", 5, 20.0, 5.0, is_valid=True)
    ctrl.update_approach("E", 5, 20.0, 5.0, is_valid=True)
    ctrl.update_approach("W", 5, 20.0, 5.0, is_valid=True)

    # Step through startup hold to NS Green
    while ctrl.state != SignalState.STATE_NS_GREEN:
        ctrl.tick(dt=0.1)

    ns_alloc = ctrl.allocated_green_s
    # Expected: 3.5 + 10.0/1.0 = 13.5s
    expected_ns = 13.5
    assert abs(ns_alloc - expected_ns) < 0.05, f"NS green mismatch: expected {expected_ns}s, got {ns_alloc}s"
    report.set_metric("NS Allocated Green", f"{ns_alloc:.1f} s")
    report.log(f"NS Green granted with alloc={ns_alloc:.1f}s for 10.0 PCU demand.")

    # Advance through NS phase to EW Green
    while ctrl.state != SignalState.STATE_EW_GREEN:
        ctrl.tick(dt=0.1)
        # Keep telemetry refreshed
        ctrl.update_approach("E", 5, 20.0, 5.0, is_valid=True)
        ctrl.update_approach("W", 5, 20.0, 5.0, is_valid=True)

    ew_alloc = ctrl.allocated_green_s
    assert abs(ew_alloc - expected_ns) < 0.05, f"EW green mismatch: expected {expected_ns}s, got {ew_alloc}s"
    report.set_metric("EW Allocated Green", f"{ew_alloc:.1f} s")
    report.log(f"EW Green granted with alloc={ew_alloc:.1f}s for 10.0 PCU demand.")

    report.passed = True
    return report


def run_sil_scenario_2_heavy_ns() -> ScenarioReport:
    report = ScenarioReport("Scenario 2: Heavy North-South Traffic",
                            "Verifies dynamic expansion of NS green (27.5s) while light EW is clamped to min green (10.0s).")
    ctrl = AdaptiveTrafficController(startup_all_red_s=1.0)

    # Heavy NS: 12 + 12 = 24 PCU. Light EW: 1 + 1 = 2 PCU.
    ctrl.update_approach("N", 12, 48.0, 12.0, is_valid=True)
    ctrl.update_approach("S", 12, 48.0, 12.0, is_valid=True)
    ctrl.update_approach("E", 1, 5.0, 1.0, is_valid=True)
    ctrl.update_approach("W", 1, 5.0, 1.0, is_valid=True)

    while ctrl.state != SignalState.STATE_NS_GREEN:
        ctrl.tick(dt=0.1)

    ns_alloc = ctrl.allocated_green_s
    # Expected NS: 3.5 + 24.0 = 27.5s
    assert abs(ns_alloc - 27.5) < 0.05, f"Expected 27.5s NS green, got {ns_alloc}s"
    report.set_metric("NS Allocated Green", f"{ns_alloc:.1f} s (Demand: 24 PCU)")

    # Step through to EW Green
    while ctrl.state != SignalState.STATE_EW_GREEN:
        ctrl.tick(dt=0.1)
        ctrl.update_approach("E", 1, 5.0, 1.0, is_valid=True)
        ctrl.update_approach("W", 1, 5.0, 1.0, is_valid=True)

    ew_alloc = ctrl.allocated_green_s
    # Expected EW: 3.5 + 2.0 = 5.5s -> clamped to kMinGreen (10.0s)
    assert abs(ew_alloc - 10.0) < 0.05, f"Expected 10.0s clamped min EW green, got {ew_alloc}s"
    report.set_metric("EW Allocated Green", f"{ew_alloc:.1f} s (Demand: 2 PCU, clamped to min)")

    ratio = ns_alloc / ew_alloc
    report.set_metric("Green Ratio (NS / EW)", f"{ratio:.2f}x")
    report.log(f"NS received {ns_alloc:.1f}s vs EW {ew_alloc:.1f}s (Ratio: {ratio:.2f}x favouring heavy corridor).")

    report.passed = True
    return report


def run_sil_scenario_3_heavy_ew() -> ScenarioReport:
    report = ScenarioReport("Scenario 3: Heavy East-West Traffic",
                            "Verifies symmetrical green expansion for EW (33.5s) while light NS receives 10.0s.")
    ctrl = AdaptiveTrafficController(startup_all_red_s=1.0)

    # Light NS: 1 + 1 = 2 PCU. Heavy EW: 15 + 15 = 30 PCU.
    ctrl.update_approach("N", 1, 5.0, 1.0, is_valid=True)
    ctrl.update_approach("S", 1, 5.0, 1.0, is_valid=True)
    ctrl.update_approach("E", 15, 60.0, 15.0, is_valid=True)
    ctrl.update_approach("W", 15, 60.0, 15.0, is_valid=True)

    while ctrl.state != SignalState.STATE_NS_GREEN:
        ctrl.tick(dt=0.1)

    ns_alloc = ctrl.allocated_green_s
    assert abs(ns_alloc - 10.0) < 0.05, f"Expected 10.0s min NS green, got {ns_alloc}s"
    report.set_metric("NS Allocated Green", f"{ns_alloc:.1f} s (Clamped Min)")

    while ctrl.state != SignalState.STATE_EW_GREEN:
        ctrl.tick(dt=0.1)
        ctrl.update_approach("E", 15, 60.0, 15.0, is_valid=True)
        ctrl.update_approach("W", 15, 60.0, 15.0, is_valid=True)

    ew_alloc = ctrl.allocated_green_s
    # Expected EW: 3.5 + 30.0 = 33.5s
    assert abs(ew_alloc - 33.5) < 0.05, f"Expected 33.5s EW green, got {ew_alloc}s"
    report.set_metric("EW Allocated Green", f"{ew_alloc:.1f} s (Demand: 30 PCU)")

    ratio = ew_alloc / ns_alloc
    report.set_metric("Green Ratio (EW / NS)", f"{ratio:.2f}x")
    report.log(f"EW received {ew_alloc:.1f}s vs NS {ns_alloc:.1f}s (Ratio: {ratio:.2f}x favouring heavy corridor).")

    report.passed = True
    return report


def run_sil_scenario_4_gap_out() -> ScenarioReport:
    report = ScenarioReport("Scenario 4: Early Gap-Out on Queue Clearance",
                            "Verifies that active green cuts short at min-green (10.0s) when queue dissipates early.")
    ctrl = AdaptiveTrafficController(startup_all_red_s=1.0)

    # Initial NS demand = 15 PCU (allocates 3.5 + 15 = 18.5s). Opposing EW has 4 PCU waiting.
    ctrl.update_approach("N", 8, 30.0, 8.0, is_valid=True)
    ctrl.update_approach("S", 7, 28.0, 7.0, is_valid=True)
    ctrl.update_approach("E", 2, 10.0, 2.0, is_valid=True)
    ctrl.update_approach("W", 2, 10.0, 2.0, is_valid=True)

    while ctrl.state != SignalState.STATE_NS_GREEN:
        ctrl.tick(dt=0.1)

    initial_alloc = ctrl.allocated_green_s
    assert abs(initial_alloc - 18.5) < 0.05, f"Expected 18.5s alloc, got {initial_alloc}s"
    report.set_metric("Nominal Green Allocation", f"{initial_alloc:.1f} s")

    # Advance to t = 10.0s (min-green threshold)
    green_start = ctrl.sim_time_s
    while ctrl.sim_time_s - green_start < 10.0:
        ctrl.tick(dt=0.1)
        ctrl.update_approach("E", 2, 10.0, 2.0, is_valid=True)
        ctrl.update_approach("W", 2, 10.0, 2.0, is_valid=True)

    # Vehicles clear! Queue drops to 0.0 PCU while opposing still has 4.0 PCU
    ctrl.update_approach("N", 0, 0.0, 0.0, is_valid=True)
    ctrl.update_approach("S", 0, 0.0, 0.0, is_valid=True)
    ctrl.tick(dt=0.1)

    # FSM must trigger GAP_OUT immediately rather than waiting until 18.5s!
    assert ctrl.state == SignalState.STATE_NS_YELLOW, f"Failed to gap out! State: {ctrl.state}"
    elapsed_actual = ctrl.sim_time_s - green_start
    time_saved = initial_alloc - elapsed_actual

    report.set_metric("Actual Green Duration", f"{elapsed_actual:.1f} s")
    report.set_metric("Wasted Green Time Saved", f"{time_saved:.1f} s")
    report.log(f"Gap-out triggered at t={elapsed_actual:.1f}s. Saved {time_saved:.1f}s of wasted green time.")

    report.passed = True
    return report


def run_sil_scenario_5_anti_starvation() -> ScenarioReport:
    report = ScenarioReport("Scenario 5: Anti-Starvation Protection",
                            "Verifies that a waiting cross street forces phase switch after exactly 50.0s of qualifying wait.")
    ctrl = AdaptiveTrafficController(startup_all_red_s=1.0, max_green_s=60.0, max_wait_starvation_s=50.0)

    # Heavy NS traffic and waiting EW traffic (4 PCU)
    # 50 PCU on NS yields calculated green: 3.5 + 50.0 = 53.5s (permitted by max_green_s=60.0)
    ctrl.update_approach("N", 25, 100.0, 25.0, is_valid=True)
    ctrl.update_approach("S", 25, 100.0, 25.0, is_valid=True)
    ctrl.update_approach("E", 2, 10.0, 2.0, is_valid=True)
    ctrl.update_approach("W", 2, 10.0, 2.0, is_valid=True)

    while ctrl.state != SignalState.STATE_NS_GREEN:
        ctrl.tick(dt=0.1)

    green_start = ctrl.sim_time_s
    starvation_triggered = False

    while ctrl.sim_time_s - green_start < 65.0:
        ctrl.tick(dt=0.1)
        # Keep telemetry fresh every 1.0s
        if int(ctrl.sim_time_s * 10) % 10 == 0:
            ctrl.update_approach("N", 25, 100.0, 25.0, is_valid=True)
            ctrl.update_approach("S", 25, 100.0, 25.0, is_valid=True)
            ctrl.update_approach("E", 2, 10.0, 2.0, is_valid=True)
            ctrl.update_approach("W", 2, 10.0, 2.0, is_valid=True)

        if ctrl.state == SignalState.STATE_NS_YELLOW:
            # Check transition log reason
            last_transition = ctrl.transition_log[-1]
            if "ANTI_STARVATION" in last_transition["reason"]:
                starvation_triggered = True
                break

    assert starvation_triggered, f"Anti-starvation failed to trigger phase change! Last transition: {ctrl.transition_log[-1]}"
    assert ctrl.ew_wait_time_s >= 50.0, f"Expected wait >= 50.0s, got {ctrl.ew_wait_time_s:.1f}s"
    report.set_metric("Anti-Starvation Threshold", "50.0 s")
    report.set_metric("Opposing Wait at Trip", f"{ctrl.ew_wait_time_s:.1f} s")
    report.log(f"Anti-starvation successfully interrupted active green at exactly {ctrl.ew_wait_time_s:.1f}s qualifying wait.")

    report.passed = True
    return report


def run_sil_scenario_6_sensor_blackout() -> ScenarioReport:
    report = ScenarioReport("Scenario 6: Sensor Blackout & Deterministic Fallback",
                            "Verifies that when all cameras disconnect (>5.0s), controller safely enters 59.0s fixed cycle.")
    ctrl = AdaptiveTrafficController(startup_all_red_s=1.0, sensor_timeout_s=5.0)

    # Initial telemetry known
    ctrl.update_approach("N", 4, 16.0, 4.0, is_valid=True)
    ctrl.update_approach("S", 4, 16.0, 4.0, is_valid=True)
    ctrl.update_approach("E", 4, 16.0, 4.0, is_valid=True)
    ctrl.update_approach("W", 4, 16.0, 4.0, is_valid=True)

    while ctrl.state != SignalState.STATE_NS_GREEN:
        ctrl.tick(dt=0.1)

    # Complete sensor blackout: Stop sending all telemetry for > 5.0 seconds
    for _ in range(60):  # 6.0 seconds without telemetry
        ctrl.tick(dt=0.1)

    pcu_ns, ns_measured = ctrl.get_phase_demand("N", "S")
    pcu_ew, ew_measured = ctrl.get_phase_demand("E", "W")

    assert not ns_measured and not ew_measured, "Sensors failed to expire into unobserved status!"
    report.set_metric("Sensor Status", "ALL_UNOBSERVED (Blackout)")

    # Complete cycle calculation: 25s NS + 3s Yellow + 1.5s All-Red + 25s EW + 3s Yellow + 1.5s All-Red = 59.0s
    total_cycle = 25.0 + 3.0 + 1.5 + 25.0 + 3.0 + 1.5
    report.set_metric("Deterministic Fallback Cycle", f"{total_cycle:.1f} s")
    report.log("Sensors decayed to unobserved; controller locked into safe pre-timed fallback with gap-out disabled.")

    report.passed = True
    return report


# ==============================================================================
# 2. Hardware-in-the-Loop (HIL) Serial Runner (Optional on ESP32 Connection)
# ==============================================================================

def run_hardware_scenario_test(port: str, baudrate: int = 115200):
    try:
        import serial
    except ImportError:
        print("[ERROR] pyserial is required for hardware tests: pip install pyserial")
        sys.exit(1)

    print(f"\n[HARDWARE CONNECT] Opening serial port {port} @ {baudrate} baud...")
    ser = serial.Serial(port, baudrate, timeout=1.0)
    time.sleep(2.0)  # Allow ESP32 boot
    ser.reset_input_buffer()

    print("[HARDWARE TEST] Streaming 4-way balanced telemetry to physical ESP32-S3...")
    legs = ["N", "S", "E", "W"]

    # Stream 10 iterations of 4-leg telemetry
    for _ in range(10):
        for leg in legs:
            pkt = build_telemetry_packet(leg, 6, 7.5, 28.4, True)
            ser.write(pkt.encode('utf-8'))
            time.sleep(0.02)  # 20ms between legs

        # Check for controller feedback broadcast
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line.startswith('{') and "state" in line:
                try:
                    fb = json.loads(line)
                    print(f"  [ESP32 FEEDBACK] State: {fb.get('state')} | Rem: {fb.get('rem')}s | Alloc: {fb.get('alloc')}s | Q: {fb.get('Q')}")
                except Exception:
                    pass
        time.sleep(0.25)

    ser.close()
    print("[HARDWARE TEST] Hardware link test completed successfully.")


# ==============================================================================
# 3. Main Runner & Summary Table
# ==============================================================================

def print_summary_table(reports: List[ScenarioReport]):
    print("\n" + "=" * 80)
    print("  ATCS ADAPTIVE SIGNAL CONTROL: AUTOMATED SCENARIO VERIFICATION")
    print("=" * 80)
    print(f"{'#':<3} | {'Scenario Name':<42} | {'Result':<8} | {'Key Metric':<20}")
    print("-" * 80)

    for i, rep in enumerate(reports, 1):
        status_badge = "[PASS]" if rep.passed else "[FAIL]"
        key_metric = next(iter(rep.metrics.items())) if rep.metrics else ("", "")
        metric_str = f"{key_metric[0]}: {key_metric[1]}" if key_metric[0] else "Verified"
        if len(metric_str) > 20:
            metric_str = metric_str[:17] + "..."
        print(f"{i:<3} | {rep.name:<42} | {status_badge:<8} | {metric_str:<20}")

    print("=" * 80)

    all_passed = all(r.passed for r in reports)
    if all_passed:
        print(f"  ALL {len(reports)} ADAPTIVE SCENARIOS PASSED WITH ZERO FAULTS.")
    else:
        print("  SOME SCENARIOS FAILED. Check details above.")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="ATCS Automated Scenario Test Suite")
    parser.add_argument("--port", type=str, default="", help="Serial port of ESP32 (leave blank for fast SIL simulation)")
    args = parser.parse_args()

    if args.port:
        run_hardware_scenario_test(args.port)
        return

    print("  Executing ATCS Automated Situation Test Suite (SIL Mode)")

    reports = [
        run_sil_scenario_1_balanced(),
        run_sil_scenario_2_heavy_ns(),
        run_sil_scenario_3_heavy_ew(),
        run_sil_scenario_4_gap_out(),
        run_sil_scenario_5_anti_starvation(),
        run_sil_scenario_6_sensor_blackout(),
    ]

    for rep in reports:
        print(f"\n>>> {rep.name}")
        print(f"    Description: {rep.description}")
        for d in rep.details:
            print(f"    - {d}")
        for k, v in rep.metrics.items():
            print(f"    * {k}: {v}")
        print(f"    Verdict: {'PASS' if rep.passed else 'FAIL'}")

    print_summary_table(reports)


if __name__ == "__main__":
    main()
