"""
ATCS Canonical Telemetry Protocol Schema (v1).

Cross-language contract shared between Python perception (vision/perception/telemetry.py),
Python simulation/controller (controller/), and C++ ESP32-S3 firmware (TelemetryParser.hpp).

Buffer & Framing Alignment:
  - Hardware Receiver Buffer (BoardConfig::SERIAL_BUF_SIZE): 512 bytes total
  - Maximum JSON Payload (excluding newline): 511 bytes (MAX_PAYLOAD_BYTES)
  - Maximum Wire Frame (including '\\n'): 512 bytes (MAX_PACKET_LENGTH_BYTES)
  - Approaches supported by ESP32-S3 firmware: {"N", "S", "E", "W"} (FIRMWARE_APPROACHES)
"""

import json
import math
from typing import Any, Dict, List, Optional, Tuple

PROTOCOL_VERSION = 1
FIRMWARE_APPROACHES = {"N", "S", "E", "W"}
EXTENDED_SIM_APPROACHES = {"N", "S", "E", "W", "NE", "NW", "SE", "SW"}
CANONICAL_APPROACHES = FIRMWARE_APPROACHES

MAX_PAYLOAD_BYTES = 511
MAX_PACKET_LENGTH_BYTES = 512
MAX_QUEUE_COUNT = 500
MAX_METRIC_FLOAT = 2000.0
MAX_ARRIVAL_RATE_PCU_S = 50.0
MAX_SEQ_VALUE = 2147483647


def _is_finite_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _is_strict_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def build_telemetry_packet(
    approach: str,
    q: int,
    pcu: float,
    d_tail_m: float = 0.0,
    valid: bool = True,
    inflow_pcu: Optional[float] = None,
    pcu_upstream: Optional[float] = None,
    arrival_rate_pcu_s: Optional[float] = None,
    calibrated: Optional[bool] = None,
    fresh: Optional[bool] = None,
    seq: Optional[int] = None,
    version: int = PROTOCOL_VERSION,
    allow_extended_approaches: bool = False,
) -> str:
    if not _is_strict_int(version) or version != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported protocol version '{version}'. Must be {PROTOCOL_VERSION}")

    app = str(approach).strip().upper()
    allowed_apps = EXTENDED_SIM_APPROACHES if allow_extended_approaches else FIRMWARE_APPROACHES
    if app not in allowed_apps:
        raise ValueError(f"Invalid approach '{approach}'. Must be one of {sorted(allowed_apps)}")

    if not isinstance(valid, bool):
        raise ValueError(f"Invalid 'valid': must be bool, got {valid}")
    if calibrated is not None and not isinstance(calibrated, bool):
        raise ValueError(f"Invalid 'calibrated': must be bool, got {calibrated}")
    if fresh is not None and not isinstance(fresh, bool):
        raise ValueError(f"Invalid 'fresh': must be bool, got {fresh}")

    if not _is_finite_number(q) or q < 0 or q > MAX_QUEUE_COUNT or int(q) != q:
        raise ValueError(f"Invalid 'Q': must be an integer in [0, {MAX_QUEUE_COUNT}], got {q}")
    if not _is_finite_number(pcu) or pcu < 0.0 or pcu > MAX_METRIC_FLOAT:
        raise ValueError(f"Invalid 'PCU': must be a finite float in [0.0, {MAX_METRIC_FLOAT}], got {pcu}")
    if not _is_finite_number(d_tail_m) or d_tail_m < 0.0 or d_tail_m > MAX_METRIC_FLOAT:
        raise ValueError(f"Invalid 'D_tail_m': must be a finite float in [0.0, {MAX_METRIC_FLOAT}], got {d_tail_m}")

    for val, name, limit in (
        (inflow_pcu, "inflow_pcu", MAX_METRIC_FLOAT),
        (pcu_upstream, "pcu_upstream", MAX_METRIC_FLOAT),
        (arrival_rate_pcu_s, "arrival_rate_pcu_s", MAX_ARRIVAL_RATE_PCU_S),
    ):
        if val is not None and (not _is_finite_number(val) or val < 0.0 or val > limit):
            raise ValueError(f"Invalid '{name}': must be a finite float in [0.0, {limit}], got {val}")

    if seq is not None and (not _is_strict_int(seq) or seq < 0 or seq > MAX_SEQ_VALUE):
        raise ValueError(f"Invalid 'seq': must be an int in [0, {MAX_SEQ_VALUE}], got {seq}")

    is_fresh = bool(valid) if fresh is None else bool(fresh)
    is_cal = bool(valid) if calibrated is None else bool(calibrated)
    effective_valid = bool(valid and is_fresh)

    payload: Dict[str, Any] = {
        "v": PROTOCOL_VERSION,
        "approach": app,
        "Q": int(q) if effective_valid else 0,
        "PCU": round(float(pcu), 2) if effective_valid else 0.0,
        "D_tail_m": round(float(d_tail_m), 2) if (effective_valid and is_cal) else 0.0,
        "valid": effective_valid,
    }
    if pcu_upstream is not None:
        payload["pcu_upstream"] = round(float(pcu_upstream), 2) if (effective_valid and is_cal) else 0.0
    if inflow_pcu is not None:
        payload["inflow_pcu"] = round(float(inflow_pcu), 2) if (effective_valid and is_cal) else 0.0
    if arrival_rate_pcu_s is not None:
        payload["arrival_rate_pcu_s"] = round(float(arrival_rate_pcu_s), 3) if (effective_valid and is_cal) else 0.0
    if calibrated is not None:
        payload["calibrated"] = is_cal
    if fresh is not None:
        payload["fresh"] = is_fresh
    if seq is not None:
        payload["seq"] = int(seq)

    raw_json = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    if len(raw_json.encode("utf-8")) > MAX_PAYLOAD_BYTES:
        raise ValueError(f"Payload exceeds {MAX_PAYLOAD_BYTES} bytes limit: {len(raw_json)} bytes")
    return raw_json + "\n"


def _reject_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"Duplicate JSON key '{k}'")
        out[k] = v
    return out


def parse_and_validate_telemetry(
    raw_line: str,
    allow_extended_approaches: bool = False,
) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    if not raw_line or not isinstance(raw_line, str):
        return False, None, "Empty packet"
    if "\x00" in raw_line:
        return False, None, "Embedded NUL byte in packet"

    line = raw_line.strip()
    if len(line) < 7:
        return False, None, "Packet too short (< 7 bytes)"
    if len(line.encode("utf-8")) > MAX_PAYLOAD_BYTES:
        return False, None, f"Packet payload too long (> {MAX_PAYLOAD_BYTES} bytes)"
    if not (line.startswith("{") and line.endswith("}")):
        return False, None, "Missing root JSON object boundaries '{' and '}'"

    def _reject_non_finite_constant(token: str):
        raise ValueError(f"Non-finite constant '{token}' not permitted")

    try:
        data = json.loads(
            line,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite_constant,
        )
    except Exception as e:
        return False, None, f"JSON parse error: {e}"

    if not isinstance(data, dict):
        return False, None, "Root must be a JSON object"

    try:
        ver = data.get("v", PROTOCOL_VERSION)
        if not _is_strict_int(ver) or ver != PROTOCOL_VERSION:
            return False, None, f"Unsupported protocol version '{ver}'"

        if "approach" in data and "leg" in data:
            return False, None, "Ambiguous duplicate: both 'approach' and 'leg' present"
        app = data.get("approach", data.get("leg"))
        if not isinstance(app, str) or not app:
            return False, None, "Missing or non-string required key 'approach' or 'leg'"
        allowed_apps = EXTENDED_SIM_APPROACHES if allow_extended_approaches else FIRMWARE_APPROACHES
        if app not in allowed_apps:
            return False, None, f"Invalid approach leg '{app}'"

        if "valid" not in data or not isinstance(data["valid"], bool):
            return False, None, "Missing or non-boolean required key 'valid'"

        q = data.get("Q")
        if not _is_strict_int(q) or q < 0 or q > MAX_QUEUE_COUNT:
            return False, None, "Missing or invalid integer key 'Q'"

        pcu = data.get("PCU")
        if not _is_finite_number(pcu) or pcu < 0.0 or pcu > MAX_METRIC_FLOAT:
            return False, None, "Missing or invalid numeric key 'PCU'"

        if "D_tail_m" in data and "d_tail" in data:
            return False, None, "Ambiguous duplicate: both 'D_tail_m' and 'd_tail' present"
        d_tail = data.get("D_tail_m", data.get("d_tail", 0.0))
        if not _is_finite_number(d_tail) or d_tail < 0.0 or d_tail > MAX_METRIC_FLOAT:
            return False, None, "Optional 'D_tail_m' must be a float in [0.0, 2000.0]"

        pcu_up = data.get("pcu_upstream", 0.0)
        if not _is_finite_number(pcu_up) or pcu_up < 0.0 or pcu_up > MAX_METRIC_FLOAT:
            return False, None, "Optional 'pcu_upstream' must be a float in [0.0, 2000.0]"

        inflow = data.get("inflow_pcu", 0.0)
        if not _is_finite_number(inflow) or inflow < 0.0 or inflow > MAX_METRIC_FLOAT:
            return False, None, "Optional 'inflow_pcu' must be a float in [0.0, 2000.0]"

        arr_rate = data.get("arrival_rate_pcu_s", 0.0)
        if not _is_finite_number(arr_rate) or arr_rate < 0.0 or arr_rate > MAX_ARRIVAL_RATE_PCU_S:
            return False, None, "Optional 'arrival_rate_pcu_s' must be a float in [0.0, 50.0]"

        calibrated = data.get("calibrated", bool(data["valid"]))
        if not isinstance(calibrated, bool):
            return False, None, "Optional 'calibrated' must be a boolean"

        fresh = data.get("fresh", bool(data["valid"]))
        if not isinstance(fresh, bool):
            return False, None, "Optional 'fresh' must be a boolean"

        seq = data.get("seq", 0)
        if not _is_strict_int(seq) or seq < 0 or seq > MAX_SEQ_VALUE:
            return False, None, "Optional 'seq' must be a non-negative integer"

        effective_valid = bool(data["valid"] and fresh)
        parsed_out: Dict[str, Any] = {
            "v": int(ver),
            "approach": app,
            "Q": int(q) if effective_valid else 0,
            "PCU": float(pcu) if effective_valid else 0.0,
            "D_tail_m": float(d_tail) if (effective_valid and calibrated) else 0.0,
            "pcu_upstream": float(pcu_up) if (effective_valid and calibrated) else 0.0,
            "inflow_pcu": float(inflow) if (effective_valid and calibrated) else 0.0,
            "arrival_rate_pcu_s": float(arr_rate) if (effective_valid and calibrated) else 0.0,
            "calibrated": bool(calibrated),
            "fresh": bool(fresh),
            "valid": effective_valid,
            "seq": int(seq),
        }
        return True, parsed_out, ""
    except Exception as e:
        return False, None, f"Malformed packet: {e}"


if __name__ == "__main__":
    sample = build_telemetry_packet("N", 6, 7.5, 28.4, True)
    ok, parsed, err = parse_and_validate_telemetry(sample)
    assert ok and parsed["approach"] == "N" and parsed["Q"] == 6 and parsed["PCU"] == 7.5 and parsed["D_tail_m"] == 28.4

    for non_finite in [float("nan"), float("inf"), float("-inf")]:
        ok, _, err = parse_and_validate_telemetry(f'{{"approach":"N","Q":{non_finite},"PCU":1.0,"valid":true}}')
        assert not ok, f"Expected rejection of Q={non_finite}"
        ok, _, err = parse_and_validate_telemetry(f'{{"approach":"N","Q":1,"PCU":{non_finite},"valid":true}}')
        assert not ok, f"Expected rejection of PCU={non_finite}"
        ok, _, err = parse_and_validate_telemetry(f'{{"approach":"N","Q":1,"PCU":1.0,"D_tail_m":{non_finite},"valid":true}}')
        assert not ok, f"Expected rejection of D_tail_m={non_finite}"

        for args in ((non_finite, 1.0, 0.0), (1, non_finite, 0.0), (1, 1.0, non_finite)):
            try:
                build_telemetry_packet("N", *args)
                assert False, f"build_telemetry_packet should raise on non-finite {args}"
            except ValueError:
                pass

    print("Telemetry contract self-test and non-finite validation passed.")
