import sys
import time
import json
import random
import argparse
from typing import Generator, Dict, Any, Optional, List


SCENARIOS = {
    "empty": {
        "description": "Empty intersection: zero vehicles across all approaches",
        "demands": {"N": (0, 0.0, 0.0), "S": (0, 0.0, 0.0), "E": (0, 0.0, 0.0), "W": (0, 0.0, 0.0)},
        "valid": {"N": True, "S": True, "E": True, "W": True},
    },
    "heavy_ns": {
        "description": "Heavy North-South traffic (20 PCU total), light East-West (3 PCU total)",
        "demands": {"N": (12, 45.0, 12.0), "S": (8, 30.0, 8.0), "E": (1, 5.0, 1.0), "W": (2, 10.0, 2.0)},
        "valid": {"N": True, "S": True, "E": True, "W": True},
    },
    "heavy_ew": {
        "description": "Heavy East-West traffic (24 PCU total), light North-South (3 PCU total)",
        "demands": {"N": (1, 5.0, 1.0), "S": (2, 8.0, 2.0), "E": (10, 40.0, 10.0), "W": (14, 55.0, 14.0)},
        "valid": {"N": True, "S": True, "E": True, "W": True},
    },
    "balanced": {
        "description": "Balanced moderate queues across all four approaches",
        "demands": {"N": (5, 20.0, 5.0), "S": (6, 22.0, 6.0), "E": (5, 19.0, 5.0), "W": (6, 24.0, 6.0)},
        "valid": {"N": True, "S": True, "E": True, "W": True},
    },
    "partial_failure": {
        "description": "Partial sensor failure: South camera offline, North/East/West online",
        "demands": {"N": (12, 45.0, 12.0), "S": None, "E": (2, 10.0, 2.0), "W": (3, 12.0, 3.0)},
        "valid": {"N": True, "S": False, "E": True, "W": True},
    },
    "complete_blackout": {
        "description": "Complete sensor failure: All 4 camera feeds offline/unobserved",
        "demands": {"N": None, "S": None, "E": None, "W": None},
        "valid": {"N": False, "S": False, "E": False, "W": False},
    },
}


def build_telemetry_packet(leg: str, q_count: int, d_tail_m: float, pcu: float, is_valid: bool = True) -> str:
    """
    Constructs a single ATCS newline-terminated JSON telemetry packet
    strictly conforming to the ESP32-S3 firmware parser schema.
    Canonical format: {"approach":"N","Q":6,"PCU":7.5,"D_tail_m":28.4,"valid":true}\n
    """
    packet = {
        "approach": leg,
        "Q": int(q_count),
        "PCU": round(float(pcu), 2),
        "D_tail_m": round(float(d_tail_m), 2),
        "valid": bool(is_valid),
    }
    return json.dumps(packet) + "\n"


def generate_scenario_stream(
    scenario_name: str,
    duration_s: float = 10.0,
    rate_hz: float = 10.0,
    fuzz_rate: float = 0.0,
) -> Generator[Dict[str, Any], None, None]:
    """
    Yields telemetry events at specified rate for a predefined or dynamic scenario.
    """
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}. Available: {list(SCENARIOS.keys())}")

    scenario = SCENARIOS[scenario_name]
    demands = scenario["demands"]
    validity = scenario["valid"]

    interval = 1.0 / max(0.1, rate_hz)
    end_time = time.time() + duration_s
    legs = ["N", "S", "E", "W"]

    while time.time() < end_time:
        for leg in legs:
            dem = demands[leg]
            if dem is None:
                # Sensor offline/stale: skip sending to simulate packet drop or send invalid packet
                continue

            q, d_tail, pcu = dem

            # Optional fuzzing injection
            if fuzz_rate > 0 and random.random() < fuzz_rate:
                # Send corrupted or malformed packet
                fuzz_choice = random.choice(["negative_q", "nan_pcu", "corrupted_syntax", "oversized"])
                if fuzz_choice == "negative_q":
                    raw_str = f'{{"approach": "{leg}", "Q": -5, "D_tail_m": 20.0, "PCU": 5.0, "valid": true}}\n'
                elif fuzz_choice == "nan_pcu":
                    raw_str = f'{{"approach": "{leg}", "Q": 5, "D_tail_m": 20.0, "PCU": nan, "valid": true}}\n'
                elif fuzz_choice == "corrupted_syntax":
                    raw_str = f'{{"approach": "{leg}", "Q": 5, "D_tail_m": 20.0, "PCU": 5.0\n'  # Missing brace
                else:
                    raw_str = f'{{"approach": "{leg}", "Q": 5, "D_tail_m": 20.0, "PCU": 5.0, "padding": "{"X" * 600}"}}\n'

                yield {
                    "time": time.time(),
                    "leg": leg,
                    "packet_str": raw_str,
                    "is_fuzz": True,
                    "fuzz_type": fuzz_choice,
                }
            else:
                raw_str = build_telemetry_packet(leg, q, d_tail, pcu, validity[leg])
                yield {
                    "time": time.time(),
                    "leg": leg,
                    "packet_str": raw_str,
                    "is_fuzz": False,
                    "fuzz_type": None,
                }

        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="ATCS Telemetry Replay & Traffic Generator")
    parser.add_argument(
        "--scenario",
        type=str,
        default="heavy_ns",
        choices=list(SCENARIOS.keys()) + ["all_scenarios"],
        help="Traffic scenario to simulate.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Duration in seconds for the simulation scenario (default: 5.0s).",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=2.0,
        help="Packet broadcast frequency in Hz per leg (default: 2.0 Hz).",
    )
    parser.add_argument(
        "--fuzz-rate",
        type=float,
        default=0.0,
        help="Probability of injecting malformed/fuzzed packets (0.0 to 1.0).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="",
        help="Optional path to record telemetry stream to disk.",
    )

    args = parser.parse_args()

    scenarios_to_run = list(SCENARIOS.keys()) if args.scenario == "all_scenarios" else [args.scenario]
    out_fp = open(args.output_file, "w") if args.output_file else None

    print(f"  ATCS Telemetry Replay Engine")
    print(f"  Rate: {args.rate} Hz | Fuzz Rate: {args.fuzz_rate * 100:.1f}%")

    try:
        for sc in scenarios_to_run:
            print(f"\n--- Running Scenario: {sc} ---")
            print(f"  Description: {SCENARIOS[sc]['description']}")
            packet_count = 0
            start_t = time.time()

            for item in generate_scenario_stream(sc, duration_s=args.duration, rate_hz=args.rate, fuzz_rate=args.fuzz_rate):
                sys.stdout.write(f"[{item['leg']}] {item['packet_str']}")
                sys.stdout.flush()
                if out_fp:
                    out_fp.write(item["packet_str"])
                packet_count += 1

            elapsed = time.time() - start_t
            print(f"  Scenario completed: {packet_count} packets streamed in {elapsed:.2f}s.")
    finally:
        if out_fp:
            out_fp.close()
            print(f"\nTelemetry saved to: {args.output_file}")


if __name__ == "__main__":
    main()
