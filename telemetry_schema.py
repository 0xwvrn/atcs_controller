"""
telemetry_schema.py
ATCS Standardized Telemetry Contract Definition & Validator

Establishes the canonical telemetry interface between host vision pipelines
and the real-time ESP32-S3 microcontroller firmware.

Canonical JSON packet format (newline-delimited, strictly < 100 bytes):
  {"approach":"N","Q":6,"PCU":7.5,"D_tail_m":28.4,"valid":true}\n
"""

import json
from typing import Dict, Any, Tuple, Optional


CANONICAL_APPROACHES = {"N", "S", "E", "W"}
MAX_PACKET_LENGTH_BYTES = 128


def build_telemetry_packet(
    approach: str,
    q: int,
    pcu: float,
    d_tail_m: float = 0.0,
    valid: bool = True
) -> str:
    """
    Constructs a compact newline-terminated JSON telemetry packet
    guaranteed to conform to the ESP32-S3 TelemetryParser schema.
    """
    approach_clean = str(approach).strip().upper()
    if approach_clean not in CANONICAL_APPROACHES:
        raise ValueError(f"Invalid approach '{approach}'. Must be one of {CANONICAL_APPROACHES}")

    q_clean = max(0, int(q))
    pcu_clean = max(0.0, round(float(pcu), 2))
    d_tail_clean = max(0.0, round(float(d_tail_m), 2))
    valid_clean = bool(valid)

    packet = {
        "approach": approach_clean,
        "Q": q_clean,
        "PCU": pcu_clean,
        "D_tail_m": d_tail_clean,
        "valid": valid_clean
    }

    raw = json.dumps(packet, separators=(',', ':')) + "\n"
    if len(raw.encode('utf-8')) > MAX_PACKET_LENGTH_BYTES:
        raise ValueError(f"Packet exceeds {MAX_PACKET_LENGTH_BYTES} bytes limit: {len(raw)} bytes")

    return raw


def parse_and_validate_telemetry(raw_line: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Strict validation mirror of the ESP32 TelemetryParser::parse() logic.
    Returns: (is_valid, parsed_dict_or_None, error_reason)
    """
    if not raw_line:
        return False, None, "Empty packet"

    line = raw_line.strip()
    if len(line) < 7:
        return False, None, "Packet too short (< 7 bytes)"
    if len(line.encode('utf-8')) > MAX_PACKET_LENGTH_BYTES:
        return False, None, f"Packet too long (> {MAX_PACKET_LENGTH_BYTES} bytes)"

    if not (line.startswith('{') and line.endswith('}')):
        return False, None, "Missing root JSON object boundaries '{' and '}'"

    try:
        data = json.loads(line)
    except Exception as e:
        return False, None, f"JSON parse error: {e}"

    if not isinstance(data, dict):
        return False, None, "Root must be a JSON object"

    # 1. Approach identifier ('approach' or 'leg')
    app = data.get("approach", data.get("leg"))
    if app is None:
        return False, None, "Missing required key 'approach' or 'leg'"
    if "approach" in data and "leg" in data:
        return False, None, "Ambiguous duplicate: both 'approach' and 'leg' present"
    if str(app).upper() not in CANONICAL_APPROACHES:
        return False, None, f"Invalid approach leg '{app}'"

    # 2. Validity boolean
    if "valid" not in data or not isinstance(data["valid"], bool):
        return False, None, "Missing or non-boolean required key 'valid'"

    # 3. Queue count integer
    if "Q" not in data or isinstance(data["Q"], bool) or not isinstance(data["Q"], (int, float)):
        return False, None, "Missing or invalid numeric key 'Q'"
    if data["Q"] < 0 or int(data["Q"]) != data["Q"]:
        return False, None, "Key 'Q' must be a non-negative integer"

    # 4. PCU float
    if "PCU" not in data or isinstance(data["PCU"], bool) or not isinstance(data["PCU"], (int, float)):
        return False, None, "Missing or invalid numeric key 'PCU'"
    if data["PCU"] < 0.0:
        return False, None, "Key 'PCU' must be >= 0.0"

    # 5. D_tail_m optional float
    d_tail = data.get("D_tail_m", data.get("d_tail", 0.0))
    if "D_tail_m" in data and "d_tail" in data:
        return False, None, "Ambiguous duplicate: both 'D_tail_m' and 'd_tail' present"
    if isinstance(d_tail, bool) or not isinstance(d_tail, (int, float)) or d_tail < 0.0:
        return False, None, "Optional 'D_tail_m' must be a non-negative float"

    normalized = {
        "approach": str(app).upper(),
        "Q": int(data["Q"]),
        "PCU": float(data["PCU"]),
        "D_tail_m": float(d_tail),
        "valid": bool(data["valid"])
    }
    return True, normalized, ""


if __name__ == "__main__":
    print("Testing ATCS Telemetry Contract...")
    sample = build_telemetry_packet("N", 6, 7.5, 28.4, True)
    print(f"Generated sample ({len(sample)} bytes): {sample.strip()}")
    ok, parsed, err = parse_and_validate_telemetry(sample)
    assert ok, f"Failed: {err}"
    assert parsed["approach"] == "N"
    assert parsed["Q"] == 6
    assert parsed["PCU"] == 7.5
    assert parsed["D_tail_m"] == 28.4
    assert parsed["valid"] is True
    print("Self-test passed successfully!")
