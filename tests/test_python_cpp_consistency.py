from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

_ROOT = Path(__file__).resolve().parents[2]
_CONTROLLER = _ROOT / "controller"
_FIRMWARE_INC = _ROOT / "firmware" / "include"
if str(_CONTROLLER) not in sys.path:
    sys.path.insert(0, str(_CONTROLLER))

from adaptive_controller import AdaptiveTrafficController

_CPP_HARNESS = r"""
#include <iostream>
#include <string>
#include "TrafficController.hpp"

int main() {
  std::string profile;
  int mod_flag = 0;
  std::cin >> profile >> mod_flag;

  TrafficController ctrl;
  if (profile == "URBAN_ARTERIAL") {
    ctrl.applyTimingProfile(PROFILE_URBAN_ARTERIAL);
  } else if (profile == "HEAVY_METRO") {
    ctrl.applyTimingProfile(PROFILE_HEAVY_METRO);
  } else {
    ctrl.applyTimingProfile(PROFILE_BASELINE_45S);
  }
  ctrl.setModeratoEnabled(mod_flag != 0, 10.0f);

  std::string cmd;
  while (std::cin >> cmd) {
    if (cmd == "UPD") {
      char leg;
      int q, valid_i;
      float d_tail, pcu, inflow;
      std::cin >> leg >> q >> d_tail >> pcu >> valid_i >> inflow;
      ctrl.updateApproach(leg, q, d_tail, pcu, valid_i != 0, inflow);
    } else if (cmd == "TICK") {
      unsigned long t_ms;
      std::cin >> t_ms;
      ctrl.setMockTimeMs(t_ms);
      ctrl.tick();
      std::cout << ctrl.getStateName() << " "
                << ctrl.getAllocatedGreen() << " "
                << ctrl.getNsWaitTime() << " "
                << ctrl.getEwWaitTime() << "\n";
    }
  }
  return 0;
}
"""


@unittest.skipIf(shutil.which("g++") is None, "g++ required for Python/C++ lockstep trace test")
class TestPythonCppFourWayConsistency(unittest.TestCase):
    """Cross-language lockstep trace tests comparing Python AdaptiveTrafficController and C++ TrafficController."""

    @classmethod
    def setUpClass(cls):
        cls._tmp_dir = tempfile.TemporaryDirectory()
        src = Path(cls._tmp_dir.name) / "lockstep.cpp"
        cls._bin = Path(cls._tmp_dir.name) / "lockstep"
        src.write_text(_CPP_HARNESS)
        subprocess.run(
            ["g++", "-std=c++17", "-O2", f"-I{_FIRMWARE_INC}", str(src), "-o", str(cls._bin)],
            check=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp_dir.cleanup()

    def _run_lockstep(self, profile: str, enable_moderato: bool, total_steps: int, event_fn):
        py_ctrl = AdaptiveTrafficController(timing_profile=profile, enable_moderato=enable_moderato)
        cpp_lines = [f"{profile} {1 if enable_moderato else 0}"]

        py_trace = []
        for step in range(1, total_steps + 1):
            t_ms = step * 100
            for leg, q, d_tail, pcu, valid, inflow in event_fn(step):
                py_ctrl.update_approach(leg, q, d_tail, pcu, is_valid=valid, inflow_pcu=inflow)
                cpp_lines.append(f"UPD {leg} {q} {d_tail} {pcu} {1 if valid else 0} {inflow}")
            py_ctrl.tick(dt=0.1)
            cpp_lines.append(f"TICK {t_ms}")
            py_trace.append((
                py_ctrl.state.value,
                py_ctrl.allocated_green_s,
                py_ctrl.ns_wait_time_s,
                py_ctrl.ew_wait_time_s,
            ))

        proc = subprocess.run(
            [str(self._bin)],
            input="\n".join(cpp_lines) + "\n",
            text=True,
            capture_output=True,
            check=True,
        )
        cpp_out = [line.split() for line in proc.stdout.strip().splitlines()]
        self.assertEqual(len(cpp_out), len(py_trace))

        py_transitions = []
        for idx, (py_st, py_alloc, _, _) in enumerate(py_trace):
            if not py_transitions or py_transitions[-1][0] != py_st:
                py_transitions.append((py_st, py_alloc, (idx + 1) * 0.1))

        cpp_transitions = []
        for idx, cpp_row in enumerate(cpp_out):
            cpp_st = {"STARTUP_ALL_RED": "ALL_RED_STARTUP"}.get(cpp_row[0], cpp_row[0])
            cpp_alloc = float(cpp_row[1])
            if not cpp_transitions or cpp_transitions[-1][0] != cpp_st:
                cpp_transitions.append((cpp_st, cpp_alloc, (idx + 1) * 0.1))

        self.assertEqual(
            [t[0] for t in py_transitions],
            [t[0] for t in cpp_transitions],
            msg=f"State transition sequence mismatch:\npy={py_transitions}\ncpp={cpp_transitions}",
        )
        for idx, ((py_st, py_alloc, py_t), (cpp_st, cpp_alloc, cpp_t)) in enumerate(zip(py_transitions, cpp_transitions)):
            self.assertAlmostEqual(py_alloc, cpp_alloc, delta=0.05, msg=f"Alloc mismatch at {py_st} (t={py_t:.1f}s)")
            if idx + 1 < len(py_transitions):
                py_dur = py_transitions[idx + 1][2] - py_t
                cpp_dur = cpp_transitions[idx + 1][2] - cpp_t
                self.assertAlmostEqual(py_dur, cpp_dur, delta=0.15, msg=f"Phase duration mismatch at {py_st}")

    def test_1_adaptive_moderato_and_gap_out_lockstep(self):
        def events(step):
            if step % 10 != 1:
                return []
            if step < 400:
                return [(leg, 12 if leg in "NS" else 4, 20.0, 12.0 if leg in "NS" else 4.0, True, 3.0 if leg in "NS" else 0.0) for leg in "NSEW"]
            if step < 700:
                # NS clears -> gap-out test
                return [(leg, 0 if leg in "NS" else 14, 15.0, 0.0 if leg in "NS" else 14.0, True, 2.0 if leg in "EW" else 0.0) for leg in "NSEW"]
            # Gridlock -> MODERATO BALANCED_CONTROL
            return [(leg, 22, 60.0, 22.0, True, 4.0) for leg in "NSEW"]

        self._run_lockstep("URBAN_ARTERIAL", True, 1200, events)

    def test_2_partial_failure_blackout_fallback_and_recovery_lockstep(self):
        def events(step):
            if step < 150 and step % 10 == 1:
                return [(leg, 6, 15.0, 6.0, True, 1.0) for leg in "NSEW"]
            if 150 <= step < 300 and step % 10 == 1:
                # S drops out -> partial failure
                return [(leg, 6, 15.0, 6.0, True, 1.0) for leg in "NEW"]
            # 300..900: complete blackout (all 4 expire -> fixed fallback 25.0s)
            if step >= 900 and step % 10 == 1:
                # Full recovery with heavy EW
                return [(leg, 2 if leg in "NS" else 10, 20.0, 2.0 if leg in "NS" else 10.0, True, 0.0) for leg in "NSEW"]
            return []

        self._run_lockstep("BASELINE_45S", False, 1400, events)


if __name__ == "__main__":
    unittest.main()
