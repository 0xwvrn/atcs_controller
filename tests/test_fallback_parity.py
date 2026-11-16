import sys
import unittest
from pathlib import Path

_CONTROLLER_DIR = Path(__file__).resolve().parents[1]
if str(_CONTROLLER_DIR) not in sys.path:
    sys.path.insert(0, str(_CONTROLLER_DIR))

from adaptive_controller import AdaptiveTrafficController, SignalState


class TestControllerFallbackParity(unittest.TestCase):
    """Stage 4A regression tests for Python/C++ complete sensor-blackout fallback parity."""

    def test_1_complete_blackout_uses_fixed_fallback_green_both_corridors(self):
        """Complete sensor blackout allocates fixed_fallback_green_s (25.0s) for NS and EW."""
        ctrl = AdaptiveTrafficController(startup_all_red_s=2.0, sensor_timeout_s=5.0)
        self.assertTrue(ctrl.is_complete_sensor_blackout())

        # Step past 2.0s startup all-red into NS_GREEN
        for _ in range(21):
            ctrl.tick(dt=0.1)
        self.assertEqual(ctrl.state, SignalState.STATE_NS_GREEN)
        self.assertAlmostEqual(ctrl.allocated_green_s, 25.0, places=5)

        # Advance through full NS_GREEN (25.0s), NS_YELLOW (3.0s), ALL_RED_TO_EW (1.5s) -> EW_GREEN
        while ctrl.state != SignalState.STATE_EW_GREEN:
            self.assertTrue(ctrl.verify_safety_interlock())
            ctrl.tick(dt=0.1)

        self.assertAlmostEqual(ctrl.allocated_green_s, 25.0, places=5)

        # Complete EW_GREEN + clearance -> back to NS_GREEN; total cycle = 59.0s
        t_ew_start = ctrl.state_start_time_s
        while ctrl.state != SignalState.STATE_NS_GREEN:
            self.assertTrue(ctrl.verify_safety_interlock())
            ctrl.tick(dt=0.1)
        half_cycle = ctrl.state_start_time_s - t_ew_start
        self.assertAlmostEqual(half_cycle, 25.0 + 3.0 + 1.5, places=1)

    def test_2_irc93_timing_profiles_fallback_durations(self):
        """BASELINE_45S (25s), URBAN_ARTERIAL (30s), and HEAVY_METRO (35s) match C++ fallback durations."""
        expected = [
            ("BASELINE_45S", 25.0, 3.0, 1.5),
            ("URBAN_ARTERIAL", 30.0, 3.0, 1.5),
            ("HEAVY_METRO", 35.0, 4.0, 2.0),
        ]
        for profile_name, exp_fallback, exp_yellow, exp_all_red in expected:
            for enable_mod in (False, True):
                ctrl = AdaptiveTrafficController(
                    timing_profile=profile_name,
                    enable_moderato=enable_mod,
                    startup_all_red_s=2.0,
                )
                for _ in range(21):
                    ctrl.tick(dt=0.1)
                self.assertEqual(ctrl.state, SignalState.STATE_NS_GREEN)
                self.assertAlmostEqual(ctrl.allocated_green_s, exp_fallback, places=5)
                self.assertAlmostEqual(ctrl.yellow_s, exp_yellow, places=5)
                self.assertAlmostEqual(ctrl.all_red_s, exp_all_red, places=5)

    def test_3_partial_degraded_mode_does_not_trigger_fixed_fallback(self):
        """Partial sensor loss (e.g. N and E fresh, S and W expired) uses adaptive allocation, not fixed fallback."""
        ctrl = AdaptiveTrafficController(startup_all_red_s=2.0, sensor_timeout_s=5.0)
        ctrl.update_approach("N", 1, 5.0, 1.0, is_valid=True, timestamp_s=0.0)
        ctrl.update_approach("E", 1, 5.0, 1.0, is_valid=True, timestamp_s=0.0)

        self.assertFalse(ctrl.is_complete_sensor_blackout())
        for _ in range(21):
            ctrl.tick(dt=0.1)
        self.assertEqual(ctrl.state, SignalState.STATE_NS_GREEN)
        # N=1.0 + S(fallback)=2.0 => 3.0 PCU => 3.5 + 3.0/1.0 = 6.5s clamped to min_green=10.0s (not 25.0s)
        self.assertAlmostEqual(ctrl.allocated_green_s, 10.0, places=5)

    def test_4_recovery_from_blackout_restores_adaptive_control(self):
        """Recovering all 4 sensors during blackout does not truncate active green, then restores adaptive green."""
        ctrl = AdaptiveTrafficController(startup_all_red_s=2.0, sensor_timeout_s=5.0)
        for _ in range(21):
            ctrl.tick(dt=0.1)
        self.assertEqual(ctrl.state, SignalState.STATE_NS_GREEN)
        self.assertAlmostEqual(ctrl.allocated_green_s, 25.0, places=5)

        # Recover all 4 sensors mid-NS_GREEN with heavy EW demand (20 PCU)
        for leg, pcu in (("N", 2.0), ("S", 2.0), ("E", 10.0), ("W", 10.0)):
            ctrl.update_approach(leg, int(pcu), 20.0, pcu, is_valid=True)

        self.assertFalse(ctrl.is_complete_sensor_blackout())
        # Active NS_GREEN remains 25.0s without truncation
        self.assertAlmostEqual(ctrl.allocated_green_s, 25.0, places=5)

        # Keep sensors fresh until EW_GREEN
        step = 0
        while ctrl.state != SignalState.STATE_EW_GREEN:
            if step % 10 == 0:
                for leg, pcu in (("N", 2.0), ("S", 2.0), ("E", 10.0), ("W", 10.0)):
                    ctrl.update_approach(leg, int(pcu), 20.0, pcu, is_valid=True)
            ctrl.tick(dt=0.1)
            step += 1

        # EW_GREEN should now be adaptive: 3.5 + 20.0 / 1.0 = 23.5s
        self.assertAlmostEqual(ctrl.allocated_green_s, 23.5, places=5)

    def test_5_normal_adaptive_and_starvation_preserved(self):
        """Valid-sensor adaptive scheduling and anti-starvation ceiling remain intact."""
        ctrl = AdaptiveTrafficController(startup_all_red_s=2.0, sensor_timeout_s=5.0)
        for leg, pcu in (("N", 12.0), ("S", 12.0), ("E", 1.0), ("W", 1.0)):
            ctrl.update_approach(leg, int(pcu), 20.0, pcu, is_valid=True, timestamp_s=0.0)

        for _ in range(21):
            ctrl.tick(dt=0.1)
        self.assertEqual(ctrl.state, SignalState.STATE_NS_GREEN)
        # 3.5 + 24.0 / 1.0 = 27.5s
        self.assertAlmostEqual(ctrl.allocated_green_s, 27.5, places=5)


if __name__ == "__main__":
    unittest.main()
