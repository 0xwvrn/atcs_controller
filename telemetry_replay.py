import argparse
import json
import random
import sys
import time
from typing import Any, Dict, Generator, Optional

from adaptive_controller import AdaptiveTrafficController, IntersectionTopology, SignalState
from telemetry_schema import MAX_PAYLOAD_BYTES, build_telemetry_packet as schema_build

try:
    import serial
except ImportError:
    serial = None  # type: ignore

SCENARIOS = {
    "empty": {
        "description": "Empty intersection: zero vehicles across all approaches",
        "demands": {k: (0, 0.0, 0.0) for k in "NSEW"},
        "inflow": {k: 0.0 for k in "NSEW"},
        "arrival_rate": {k: 0.05 for k in "NSEW"},
        "valid": {k: True for k in "NSEW"},
    },
    "heavy_ns": {
        "description": "Heavy North-South traffic (20 PCU + 6 PCU inflow), light East-West (3 PCU)",
        "demands": {"N": (12, 45.0, 12.0), "S": (8, 30.0, 8.0), "E": (1, 5.0, 1.0), "W": (2, 10.0, 2.0)},
        "inflow": {"N": 3.0, "S": 3.0, "E": 0.0, "W": 0.0},
        "arrival_rate": {"N": 0.35, "S": 0.30, "E": 0.08, "W": 0.08},
        "valid": {k: True for k in "NSEW"},
    },
    "heavy_ew": {
        "description": "Heavy East-West traffic (24 PCU + 6 PCU inflow), light North-South (3 PCU)",
        "demands": {"N": (1, 5.0, 1.0), "S": (2, 8.0, 2.0), "E": (10, 40.0, 10.0), "W": (14, 55.0, 14.0)},
        "inflow": {"N": 0.0, "S": 0.0, "E": 3.0, "W": 3.0},
        "arrival_rate": {"N": 0.08, "S": 0.08, "E": 0.35, "W": 0.35},
        "valid": {k: True for k in "NSEW"},
    },
    "balanced": {
        "description": "Balanced moderate queues across all four approaches",
        "demands": {"N": (5, 20.0, 5.0), "S": (6, 22.0, 6.0), "E": (5, 19.0, 5.0), "W": (6, 24.0, 6.0)},
        "inflow": {"N": 1.0, "S": 1.0, "E": 1.0, "W": 1.0},
        "arrival_rate": {"N": 0.18, "S": 0.18, "E": 0.18, "W": 0.18},
        "valid": {k: True for k in "NSEW"},
    },
    "gridlock": {
        "description": "Oversaturated gridlock on both corridors triggering MODERATO BALANCED_CONTROL",
        "demands": {"N": (22, 70.0, 22.0), "S": (20, 65.0, 20.0), "E": (16, 52.0, 16.0), "W": (16, 52.0, 16.0)},
        "inflow": {"N": 4.0, "S": 4.0, "E": 3.0, "W": 3.0},
        "arrival_rate": {"N": 0.45, "S": 0.45, "E": 0.40, "W": 0.40},
        "valid": {k: True for k in "NSEW"},
    },
    "partial_failure": {
        "description": "Partial sensor failure: South camera offline, North/East/West online",
        "demands": {"N": (12, 45.0, 12.0), "S": None, "E": (2, 10.0, 2.0), "W": (3, 12.0, 3.0)},
        "inflow": {"N": 2.0, "S": 0.0, "E": 0.0, "W": 0.0},
        "arrival_rate": {"N": 0.25, "S": 0.0, "E": 0.10, "W": 0.10},
        "valid": {"N": True, "S": False, "E": True, "W": True},
    },
    "complete_blackout": {
        "description": "Complete sensor failure: All 4 camera feeds offline/unobserved",
        "demands": {k: None for k in "NSEW"},
        "inflow": {k: 0.0 for k in "NSEW"},
        "arrival_rate": {k: 0.0 for k in "NSEW"},
        "valid": {k: False for k in "NSEW"},
    },
}

TOPOLOGY_MAP = {
    "STANDARD_4WAY": IntersectionTopology.STANDARD_4WAY,
    "T_JUNCTION_3WAY_NS_E": IntersectionTopology.T_JUNCTION_3WAY_NS_E,
    "T_JUNCTION_3WAY_EW_N": IntersectionTopology.T_JUNCTION_3WAY_EW_N,
    "DOUBLE_CYCLE_NS_E_NS_W": IntersectionTopology.DOUBLE_CYCLE_NS_E_NS_W,
}


def build_telemetry_packet(
    leg: str, q_count: int, d_tail_m: float, pcu: float, is_valid: bool = True, inflow_pcu: float = 0.0
) -> str:
    return schema_build(approach=leg, q=q_count, pcu=pcu, d_tail_m=d_tail_m, valid=is_valid, inflow_pcu=inflow_pcu)


def build_closed_loop_tx_packet(
    leg: str,
    q_count: int,
    pcu: float,
    d_tail_m: float,
    inflow_pcu: float,
    arrival_rate_pcu_s: float,
    seq: int,
    profile: Optional[str] = None,
    topology: Optional[str] = None,
    enable_moderato: Optional[bool] = None,
) -> str:
    """
    Constructs a Canonical Schema v1 observation packet via build_telemetry_packet().
    When runtime config commands are provided for the initial HIL handshake, attaches
    profile/topology/moderato while enforcing the 511-byte payload limit.
    """
    raw_line = schema_build(
        approach=leg,
        q=q_count,
        pcu=pcu,
        d_tail_m=d_tail_m,
        valid=True,
        inflow_pcu=inflow_pcu,
        pcu_upstream=inflow_pcu,
        arrival_rate_pcu_s=arrival_rate_pcu_s,
        calibrated=True,
        fresh=True,
        seq=seq,
    )
    if profile is None and topology is None and enable_moderato is None:
        return raw_line

    payload = json.loads(raw_line)
    if profile is not None:
        payload["profile"] = profile
    if topology is not None:
        payload["topology"] = topology
    if enable_moderato is not None:
        payload["moderato"] = bool(enable_moderato)
    raw_cfg = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    if len(raw_cfg.encode("utf-8")) > MAX_PAYLOAD_BYTES:
        raise ValueError(f"Config-bearing payload exceeds {MAX_PAYLOAD_BYTES} bytes limit: {len(raw_cfg)} bytes")
    return raw_cfg + "\n"


def generate_scenario_stream(
    scenario_name: str,
    duration_s: float = 10.0,
    rate_hz: float = 10.0,
    fuzz_rate: float = 0.0,
) -> Generator[Dict[str, Any], None, None]:
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}. Available: {list(SCENARIOS.keys())}")

    scenario = SCENARIOS[scenario_name]
    demands, validity = scenario["demands"], scenario["valid"]
    inflows = scenario.get("inflow", {k: 0.0 for k in "NSEW"})
    interval = 1.0 / max(0.1, rate_hz)
    end_time = time.time() + duration_s

    while time.time() < end_time:
        for leg in ("N", "S", "E", "W"):
            dem = demands[leg]
            if dem is None:
                continue

            q, d_tail, pcu = dem
            is_fuzz = fuzz_rate > 0 and random.random() < fuzz_rate
            fuzz_type = random.choice(["negative_q", "nan_pcu", "corrupted_syntax", "oversized"]) if is_fuzz else None

            if is_fuzz:
                fuzz_map = {
                    "negative_q": f'{{"approach": "{leg}", "Q": -5, "D_tail_m": 20.0, "PCU": 5.0, "valid": true}}\n',
                    "nan_pcu": f'{{"approach": "{leg}", "Q": 5, "D_tail_m": 20.0, "PCU": nan, "valid": true}}\n',
                    "corrupted_syntax": f'{{"approach": "{leg}", "Q": 5, "D_tail_m": 20.0, "PCU": 5.0\n',
                    "oversized": f'{{"approach": "{leg}", "Q": 5, "D_tail_m": 20.0, "PCU": 5.0, "padding": "{"X" * 600}"}}\n',
                }
                raw_str = fuzz_map[fuzz_type]
            else:
                raw_str = build_telemetry_packet(leg, q, d_tail, pcu, validity[leg], inflow_pcu=inflows.get(leg, 0.0))

            yield {"time": time.time(), "leg": leg, "packet_str": raw_str, "is_fuzz": is_fuzz, "fuzz_type": fuzz_type}
        time.sleep(interval)


def run_closed_loop_demo(
    scenario_name: str,
    duration_s: float,
    profile: str = "URBAN_ARTERIAL",
    topology: str = "STANDARD_4WAY",
    enable_moderato: bool = True,
    serial_port: str = "",
    baudrate: int = 115200,
    sim_speed: float = 1.0,
):
    """Runs a closed-loop traffic simulation draining queues on Green and accumulating arrivals on Red."""
    scenario = SCENARIOS[scenario_name]
    ctrl = AdaptiveTrafficController(
        phase_approaches=TOPOLOGY_MAP.get(topology, IntersectionTopology.STANDARD_4WAY),
        enable_moderato=enable_moderato,
        timing_profile=profile,
    )

    ser_conn: Optional[Any] = None
    if serial_port and serial is not None:
        try:
            ser_conn = serial.Serial(serial_port, baudrate, timeout=0.0, write_timeout=0.05)
            print(f"[HIL] Connected to ESP32-S3 on {serial_port} @ {baudrate} baud.")
        except Exception as e:
            print(f"[SIL] Serial port {serial_port} unavailable ({e}); running closed-loop in standalone SIL mode.")

    pcu_state = {leg: float(scenario["demands"][leg][2]) if scenario["demands"][leg] else 0.0 for leg in "NSEW"}
    inflow_state = dict(scenario.get("inflow", {k: 0.0 for k in "NSEW"}))
    arr_rates = dict(scenario.get("arrival_rate", {k: 0.1 for k in "NSEW"}))
    valid_flags = dict(scenario["valid"])

    dt = 0.1
    steps = max(1, int(duration_s / dt))
    config_sent = False
    tx_seq = 0

    print(
        f"# Closed-Loop Demo | Scenario={scenario_name} | Profile={profile} "
        f"| Topology={topology} | MODERATO={enable_moderato}"
    )

    try:
        for step in range(steps):
            state_elapsed = ctrl.sim_time_s - ctrl.state_start_time_s
            ns_discharging = ctrl.state == SignalState.STATE_NS_GREEN and state_elapsed >= ctrl.startup_lost_time_s
            ew_discharging = ctrl.state == SignalState.STATE_EW_GREEN and state_elapsed >= ctrl.startup_lost_time_s

            for leg in ("N", "S", "E", "W"):
                if not valid_flags[leg]:
                    continue
                is_green = ns_discharging if leg in ("N", "S") else ew_discharging
                if is_green:
                    pcu_state[leg] = max(0.0, pcu_state[leg] - (ctrl.discharge_rate_pcu_per_s * dt))
                    inflow_state[leg] = max(0.0, inflow_state[leg] - (0.25 * dt))
                else:
                    pcu_state[leg] = min(60.0, pcu_state[leg] + (arr_rates[leg] * dt))

                q_cnt = int(round(pcu_state[leg]))
                d_tail = round(pcu_state[leg] * 3.5, 1)
                ctrl.update_approach(leg, q_cnt, d_tail, pcu_state[leg], is_valid=True, inflow_pcu=inflow_state[leg])

            prev_state = ctrl.state
            ctrl.tick(dt=dt)

            if step % 5 == 0:
                for leg in ("N", "S", "E", "W"):
                    if not valid_flags[leg]:
                        continue
                    tx_seq += 1
                    send_cfg = (not config_sent and leg == "N")
                    raw_pkt = build_closed_loop_tx_packet(
                        leg=leg,
                        q_count=int(round(pcu_state[leg])),
                        pcu=round(pcu_state[leg], 2),
                        d_tail_m=round(pcu_state[leg] * 3.5, 2),
                        inflow_pcu=round(inflow_state[leg], 2),
                        arrival_rate_pcu_s=round(arr_rates[leg], 3),
                        seq=tx_seq,
                        profile=profile if send_cfg else None,
                        topology=topology if send_cfg else None,
                        enable_moderato=enable_moderato if send_cfg else None,
                    )
                    if send_cfg:
                        config_sent = True
                    if ser_conn is not None:
                        try:
                            ser_conn.write(raw_pkt.encode("utf-8"))
                        except Exception:
                            pass

            if ctrl.state != prev_state or step % 20 == 0:
                ns_pcu, _ = ctrl.get_phase_demand("N", "S")
                ew_pcu, _ = ctrl.get_phase_demand("E", "W")
                rem_s = max(0.0, ctrl.allocated_green_s - (ctrl.sim_time_s - ctrl.state_start_time_s))
                print(
                    f"  t={ctrl.sim_time_s:5.1f}s | State={ctrl.state.value:14s} | Regime={ctrl.active_regime.value:18s} "
                    f"| Alloc={ctrl.allocated_green_s:4.1f}s (Rem={rem_s:4.1f}s) | NS={ns_pcu:4.1f} PCU | EW={ew_pcu:4.1f} PCU"
                )

            if sim_speed > 0:
                time.sleep(dt / sim_speed)
    finally:
        if ser_conn is not None:
            ser_conn.close()


def main():
    parser = argparse.ArgumentParser(description="ATCS Telemetry Replay & Closed-Loop Demo Engine")
    parser.add_argument("--scenario", default="heavy_ns", choices=list(SCENARIOS.keys()) + ["all_scenarios"])
    parser.add_argument("--duration", type=float, default=15.0, help="Duration in seconds (default: 15.0s).")
    parser.add_argument("--rate", type=float, default=2.0, help="Packet frequency in Hz per leg (default: 2.0 Hz).")
    parser.add_argument("--fuzz-rate", type=float, default=0.0, help="Probability of injecting fuzzed packets.")
    parser.add_argument("--output-file", type=str, default="", help="Optional path to record telemetry stream.")
    parser.add_argument("--closed-loop", action="store_true", help="Run closed-loop queue drain/arrival simulation.")
    parser.add_argument("--profile", default="URBAN_ARTERIAL", choices=["BASELINE_45S", "URBAN_ARTERIAL", "HEAVY_METRO"])
    parser.add_argument("--topology", default="STANDARD_4WAY", choices=list(TOPOLOGY_MAP.keys()))
    parser.add_argument("--no-moderato", action="store_true", help="Disable MODERATO 3-regime control in closed-loop.")
    parser.add_argument("--serial-port", default="", help="Optional ESP32-S3 serial port for HIL closed-loop demo.")
    parser.add_argument("--sim-speed", type=float, default=10.0, help="Simulation speed multiplier (1.0=real-time, 0=max).")
    args = parser.parse_args()

    scenarios = list(SCENARIOS.keys()) if args.scenario == "all_scenarios" else [args.scenario]

    if args.closed_loop:
        for sc in scenarios:
            run_closed_loop_demo(
                scenario_name=sc,
                duration_s=args.duration,
                profile=args.profile,
                topology=args.topology,
                enable_moderato=not args.no_moderato,
                serial_port=args.serial_port,
                sim_speed=args.sim_speed,
            )
        return

    out_fp = open(args.output_file, "w") if args.output_file else None
    print(f"# ATCS Telemetry Replay Engine | Rate: {args.rate} Hz | Fuzz Rate: {args.fuzz_rate * 100:.1f}%")
    try:
        for sc in scenarios:
            print(f"\n# Scenario: {sc} ({SCENARIOS[sc]['description']})")
            count, start_t = 0, time.time()
            for item in generate_scenario_stream(sc, duration_s=args.duration, rate_hz=args.rate, fuzz_rate=args.fuzz_rate):
                sys.stdout.write(f"[{item['leg']}] {item['packet_str']}")
                sys.stdout.flush()
                if out_fp:
                    out_fp.write(item["packet_str"])
                count += 1
            print(f"Completed: {count} packets streamed in {time.time() - start_t:.2f}s.")
    finally:
        if out_fp:
            out_fp.close()
            print(f"Telemetry saved to: {args.output_file}")


if __name__ == "__main__":
    main()
