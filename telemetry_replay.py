import argparse
import json
import random
import sys
import time
from typing import Any, Dict, Generator
from telemetry_schema import build_telemetry_packet as schema_build

SCENARIOS = {
    "empty": {
        "description": "Empty intersection: zero vehicles across all approaches",
        "demands": {k: (0, 0.0, 0.0) for k in "NSEW"},
        "valid": {k: True for k in "NSEW"},
    },
    "heavy_ns": {
        "description": "Heavy North-South traffic (20 PCU total), light East-West (3 PCU total)",
        "demands": {"N": (12, 45.0, 12.0), "S": (8, 30.0, 8.0), "E": (1, 5.0, 1.0), "W": (2, 10.0, 2.0)},
        "valid": {k: True for k in "NSEW"},
    },
    "heavy_ew": {
        "description": "Heavy East-West traffic (24 PCU total), light North-South (3 PCU total)",
        "demands": {"N": (1, 5.0, 1.0), "S": (2, 8.0, 2.0), "E": (10, 40.0, 10.0), "W": (14, 55.0, 14.0)},
        "valid": {k: True for k in "NSEW"},
    },
    "balanced": {
        "description": "Balanced moderate queues across all four approaches",
        "demands": {"N": (5, 20.0, 5.0), "S": (6, 22.0, 6.0), "E": (5, 19.0, 5.0), "W": (6, 24.0, 6.0)},
        "valid": {k: True for k in "NSEW"},
    },
    "partial_failure": {
        "description": "Partial sensor failure: South camera offline, North/East/West online",
        "demands": {"N": (12, 45.0, 12.0), "S": None, "E": (2, 10.0, 2.0), "W": (3, 12.0, 3.0)},
        "valid": {"N": True, "S": False, "E": True, "W": True},
    },
    "complete_blackout": {
        "description": "Complete sensor failure: All 4 camera feeds offline/unobserved",
        "demands": {k: None for k in "NSEW"},
        "valid": {k: False for k in "NSEW"},
    },
}


def build_telemetry_packet(leg: str, q_count: int, d_tail_m: float, pcu: float, is_valid: bool = True) -> str:
    return schema_build(approach=leg, q=q_count, pcu=pcu, d_tail_m=d_tail_m, valid=is_valid)


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
                raw_str = build_telemetry_packet(leg, q, d_tail, pcu, validity[leg])

            yield {"time": time.time(), "leg": leg, "packet_str": raw_str, "is_fuzz": is_fuzz, "fuzz_type": fuzz_type}
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="ATCS Telemetry Replay & Traffic Generator")
    parser.add_argument("--scenario", default="heavy_ns", choices=list(SCENARIOS.keys()) + ["all_scenarios"])
    parser.add_argument("--duration", type=float, default=5.0, help="Duration in seconds (default: 5.0s).")
    parser.add_argument("--rate", type=float, default=2.0, help="Packet frequency in Hz per leg (default: 2.0 Hz).")
    parser.add_argument("--fuzz-rate", type=float, default=0.0, help="Probability of injecting fuzzed packets.")
    parser.add_argument("--output-file", type=str, default="", help="Optional path to record telemetry stream.")
    args = parser.parse_args()

    scenarios = list(SCENARIOS.keys()) if args.scenario == "all_scenarios" else [args.scenario]
    out_fp = open(args.output_file, "w") if args.output_file else None

    print(f"  ATCS Telemetry Replay Engine | Rate: {args.rate} Hz | Fuzz Rate: {args.fuzz_rate * 100:.1f}%")
    try:
        for sc in scenarios:
            print(f"\n--- Running Scenario: {sc} ---\n  Description: {SCENARIOS[sc]['description']}")
            count, start_t = 0, time.time()
            for item in generate_scenario_stream(sc, duration_s=args.duration, rate_hz=args.rate, fuzz_rate=args.fuzz_rate):
                sys.stdout.write(f"[{item['leg']}] {item['packet_str']}")
                sys.stdout.flush()
                if out_fp:
                    out_fp.write(item["packet_str"])
                count += 1
            print(f"  Scenario completed: {count} packets streamed in {time.time() - start_t:.2f}s.")
    finally:
        if out_fp:
            out_fp.close()
            print(f"\nTelemetry saved to: {args.output_file}")


if __name__ == "__main__":
    main()
