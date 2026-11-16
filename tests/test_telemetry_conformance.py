import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vision"))

from telemetry_schema import (
    MAX_PAYLOAD_BYTES,
    build_telemetry_packet,
    parse_and_validate_telemetry,
)
from telemetry_replay import build_closed_loop_tx_packet
from perception.telemetry import (
    NonBlockingTelemetryBridge,
    serialize_observation_packet,
)


class TestPhase1DTelemetryConformance(unittest.TestCase):
    def setUp(self):
        fixtures_path = Path(__file__).resolve().parent / "telemetry_conformance_fixtures.json"
        with open(fixtures_path, "r") as f:
            self.fixtures = json.load(f)

    def test_1_shared_conformance_fixtures_python_parser(self):
        for fx in self.fixtures:
            with self.subTest(fixture=fx["name"]):
                ok, parsed, err = parse_and_validate_telemetry(fx["raw"])
                self.assertEqual(ok, fx["expect_parse_ok"], msg=f"Fixture {fx['name']} failed: {err}")
                if fx["expect_parse_ok"]:
                    self.assertIsNotNone(parsed)
                    self.assertEqual(parsed["valid"], fx["expect_valid"])
                    self.assertEqual(parsed["calibrated"], fx["expect_calibrated"])
                    self.assertEqual(parsed["fresh"], fx["expect_fresh"])
                    self.assertEqual(parsed["approach"], fx["expect_approach"])
                    self.assertEqual(parsed["Q"], fx["expect_q"])
                    self.assertAlmostEqual(parsed["PCU"], fx["expect_pcu"], places=2)
                    self.assertAlmostEqual(parsed["D_tail_m"], fx["expect_d_tail_m"], places=2)
                    self.assertAlmostEqual(parsed["pcu_upstream"], fx["expect_pcu_upstream"], places=2)
                    self.assertAlmostEqual(parsed["inflow_pcu"], fx["expect_inflow_pcu"], places=2)
                    self.assertAlmostEqual(parsed["arrival_rate_pcu_s"], fx["expect_arrival_rate_pcu_s"], places=3)
                    self.assertEqual(parsed["seq"], fx["expect_seq"])

    def test_2_oversized_packet_boundary_511_vs_512_bytes(self):
        # Exactly 511 payload bytes -> accepted
        prefix = '{"v":1,"approach":"N","Q":1,"PCU":1.0,"valid":true,'
        suffix = '"D_tail_m":5.0}'
        pad_len = MAX_PAYLOAD_BYTES - len(prefix) - len(suffix)
        exact_511 = prefix + (" " * pad_len) + suffix
        self.assertEqual(len(exact_511.encode("utf-8")), 511)
        ok, parsed, err = parse_and_validate_telemetry(exact_511)
        self.assertTrue(ok, msg=err)

        # 512 payload bytes -> rejected
        over_512 = prefix + (" " * (pad_len + 1)) + suffix
        self.assertEqual(len(over_512.encode("utf-8")), 512)
        ok2, _, _ = parse_and_validate_telemetry(over_512)
        self.assertFalse(ok2)

    def test_3_vision_serializer_and_schema_agreement(self):
        smoothed_q = {
            "Q_stopline": 6,
            "PCU_stopline": 7.5,
            "D_tail_m": 28.4,
            "Q_upstream": 2,
            "PCU_upstream": 2.5,
            "inflow_pcu": 2.5,
            "arrival_rate_pcu_s": 0.35,
        }
        raw = serialize_observation_packet(
            approach_leg="N",
            smoothed_q=smoothed_q,
            is_calibrated=True,
            frame_valid=True,
            fresh=True,
            seq=42,
        )
        ok, parsed, err = parse_and_validate_telemetry(raw)
        self.assertTrue(ok, msg=err)
        self.assertEqual(parsed["v"], 1)
        self.assertEqual(parsed["approach"], "N")
        self.assertEqual(parsed["Q"], 6)
        self.assertAlmostEqual(parsed["PCU"], 7.5)
        self.assertAlmostEqual(parsed["D_tail_m"], 28.4)
        self.assertAlmostEqual(parsed["pcu_upstream"], 2.5)
        self.assertAlmostEqual(parsed["inflow_pcu"], 2.5)
        self.assertAlmostEqual(parsed["arrival_rate_pcu_s"], 0.35)
        self.assertTrue(parsed["calibrated"])
        self.assertTrue(parsed["fresh"])
        self.assertTrue(parsed["valid"])

    def test_4_unsupported_firmware_approach_rejected_by_producer(self):
        for bad_leg in ("NE", "NW", "SE", "SW", "X"):
            with self.assertRaises(ValueError):
                serialize_observation_packet(bad_leg, {"Q_stopline": 1, "PCU_stopline": 1.0, "D_tail_m": 5.0}, True)
            with self.assertRaises(ValueError):
                build_telemetry_packet(bad_leg, 1, 1.0, 5.0, True)

    def test_5_closed_loop_replay_packet_builder_and_config_handshake(self):
        # Ordinary observation packet delegates to build_telemetry_packet()
        obs_pkt = build_closed_loop_tx_packet(
            leg="N", q_count=8, pcu=8.0, d_tail_m=28.0, inflow_pcu=2.0, arrival_rate_pcu_s=0.35, seq=5
        )
        self.assertLessEqual(len(obs_pkt.rstrip("\n").encode("utf-8")), MAX_PAYLOAD_BYTES)
        ok, parsed, err = parse_and_validate_telemetry(obs_pkt)
        self.assertTrue(ok, msg=err)
        self.assertEqual(parsed["v"], 1)
        self.assertEqual(parsed["seq"], 5)
        self.assertTrue(parsed["calibrated"])
        self.assertTrue(parsed["fresh"])

        # Initial HIL config handshake preserves profile/topology/moderato within 511 bytes
        cfg_pkt = build_closed_loop_tx_packet(
            leg="N", q_count=8, pcu=8.0, d_tail_m=28.0, inflow_pcu=2.0, arrival_rate_pcu_s=0.35, seq=1,
            profile="URBAN_ARTERIAL", topology="DOUBLE_CYCLE_NS_E_NS_W", enable_moderato=True,
        )
        self.assertLessEqual(len(cfg_pkt.rstrip("\n").encode("utf-8")), MAX_PAYLOAD_BYTES)
        raw_obj = json.loads(cfg_pkt)
        self.assertEqual(raw_obj["profile"], "URBAN_ARTERIAL")
        self.assertEqual(raw_obj["topology"], "DOUBLE_CYCLE_NS_E_NS_W")
        self.assertTrue(raw_obj["moderato"])
        ok_cfg, parsed_cfg, err_cfg = parse_and_validate_telemetry(cfg_pkt)
        self.assertTrue(ok_cfg, msg=err_cfg)
        self.assertEqual(parsed_cfg["v"], 1)


if __name__ == "__main__":
    unittest.main()
