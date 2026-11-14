import json
import math
from typing import Dict, Any, Tuple, Optional

CANONICAL_APPROACHES = {"N", "S", "E", "W", "NE", "NW", "SE", "SW"}
MAX_PACKET_LENGTH_BYTES = 128


def _is_finite_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def build_telemetry_packet(
    approach: str,
    q: int,
    pcu: float,
    d_tail_m: float = 0.0,
    valid: bool = True,
    inflow_pcu: Optional[float] = None,
) -> str:
    app = str(approach).strip().upper()
    if app not in CANONICAL_APPROACHES:
        raise ValueError(f"Invalid approach '{approach}'. Must be one of {CANONICAL_APPROACHES}")

    if not _is_finite_number(q) or q < 0 or int(q) != q:
        raise ValueError(f"Invalid 'Q': must be a finite non-negative integer, got {q}")
    if not _is_finite_number(pcu) or pcu < 0.0:
        raise ValueError(f"Invalid 'PCU': must be a finite non-negative float, got {pcu}")
    if not _is_finite_number(d_tail_m) or d_tail_m < 0.0:
        raise ValueError(f"Invalid 'D_tail_m': must be a finite non-negative float, got {d_tail_m}")
    if inflow_pcu is not None and (not _is_finite_number(inflow_pcu) or inflow_pcu < 0.0):
        raise ValueError(f"Invalid 'inflow_pcu': must be a finite non-negative float, got {inflow_pcu}")

    payload: Dict[str, Any] = {
        "approach": app,
        "Q": max(0, int(q)),
        "PCU": max(0.0, round(float(pcu), 2)),
        "D_tail_m": max(0.0, round(float(d_tail_m), 2)),
        "valid": bool(valid),
    }
    if inflow_pcu is not None:
        payload["inflow_pcu"] = max(0.0, round(float(inflow_pcu), 2))

    raw = json.dumps(payload, separators=(',', ':')) + "\n"

    if len(raw.encode('utf-8')) > MAX_PACKET_LENGTH_BYTES:
        raise ValueError(f"Packet exceeds {MAX_PACKET_LENGTH_BYTES} bytes limit: {len(raw)} bytes")
    return raw


def parse_and_validate_telemetry(raw_line: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
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

    try:
        if "approach" in data and "leg" in data:
            return False, None, "Ambiguous duplicate: both 'approach' and 'leg' present"
        app = data.get("approach", data.get("leg"))
        if app is None:
            return False, None, "Missing required key 'approach' or 'leg'"
        if str(app).upper() not in CANONICAL_APPROACHES:
            return False, None, f"Invalid approach leg '{app}'"

        if "valid" not in data or not isinstance(data["valid"], bool):
            return False, None, "Missing or non-boolean required key 'valid'"

        q = data.get("Q")
        if not _is_finite_number(q) or q < 0 or int(q) != q:
            return False, None, "Missing or invalid numeric key 'Q'"

        pcu = data.get("PCU")
        if not _is_finite_number(pcu) or pcu < 0.0:
            return False, None, "Missing or invalid numeric key 'PCU'"

        if "D_tail_m" in data and "d_tail" in data:
            return False, None, "Ambiguous duplicate: both 'D_tail_m' and 'd_tail' present"
        d_tail = data.get("D_tail_m", data.get("d_tail", 0.0))
        if not _is_finite_number(d_tail) or d_tail < 0.0:
            return False, None, "Optional 'D_tail_m' must be a non-negative float"

        inflow = data.get("inflow_pcu", 0.0)
        if not _is_finite_number(inflow) or inflow < 0.0:
            return False, None, "Optional 'inflow_pcu' must be a non-negative float"

        parsed_out: Dict[str, Any] = {
            "approach": str(app).upper(),
            "Q": int(q),
            "PCU": float(pcu),
            "D_tail_m": float(d_tail),
            "valid": bool(data["valid"]),
        }
        if "inflow_pcu" in data:
            parsed_out["inflow_pcu"] = float(inflow)

        return True, parsed_out, ""
    except Exception as e:
        return False, None, f"Malformed packet: {e}"


if __name__ == "__main__":
    sample = build_telemetry_packet("N", 6, 7.5, 28.4, True)
    ok, parsed, err = parse_and_validate_telemetry(sample)
    assert ok and parsed["approach"] == "N" and parsed["Q"] == 6 and parsed["PCU"] == 7.5 and parsed["D_tail_m"] == 28.4

    # Non-finite and malformed rejection tests
    for non_finite in [float('nan'), float('inf'), float('-inf')]:
        ok, _, err = parse_and_validate_telemetry(f'{{"approach":"N","Q":{non_finite},"PCU":1.0,"valid":true}}')
        assert not ok, f"Expected rejection of Q={non_finite}"
        ok, _, err = parse_and_validate_telemetry(f'{{"approach":"N","Q":1,"PCU":{non_finite},"valid":true}}')
        assert not ok, f"Expected rejection of PCU={non_finite}"
        ok, _, err = parse_and_validate_telemetry(f'{{"approach":"N","Q":1,"PCU":1.0,"D_tail_m":{non_finite},"valid":true}}')
        assert not ok, f"Expected rejection of D_tail_m={non_finite}"

        try:
            build_telemetry_packet("N", non_finite, 1.0)
            assert False, "build_telemetry_packet should raise on non-finite Q"
        except ValueError:
            pass

        try:
            build_telemetry_packet("N", 1, non_finite)
            assert False, "build_telemetry_packet should raise on non-finite PCU"
        except ValueError:
            pass

        try:
            build_telemetry_packet("N", 1, 1.0, non_finite)
            assert False, "build_telemetry_packet should raise on non-finite D_tail_m"
        except ValueError:
            pass

    print("Telemetry contract self-test and non-finite validation passed.")
