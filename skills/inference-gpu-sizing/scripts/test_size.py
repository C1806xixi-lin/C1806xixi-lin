"""Verify the documented stdin/stdout interface."""
import json
from pathlib import Path
import subprocess
import sys
import unittest


def run(payload):
    return subprocess.run([sys.executable, str(Path(__file__).with_name("size.py"))],
                          input=json.dumps(payload), capture_output=True, text=True, timeout=5)


def memory(**overrides):
    return dict(parameters=8589934592, weight_bits=16, layers=32, kv_heads=8,
                head_dim=128, kv_bits=16, live_tokens=4096, sequences=8, gpu_vram_gib=20,
                reserve_fraction=0.2, tp_candidates=[4, 2, 1]) | overrides


class SizingTests(unittest.TestCase):
    def check(self, mode, section):
        result = run({mode: section})
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)[mode]

    def test_peak_cards_and_whole_nodes_are_distinct(self):
        result = self.check("peak", dict(target_pflops=500, per_gpu_tflops=989.5, gpus_per_node=8))
        self.assertEqual(result, dict(minimum_cards=506, full_nodes=64, full_node_cards=512,
                                      full_node_peak_pflops=506.624))

    def test_memory_reserve_and_kv_sharding_change_the_result(self):
        result = self.check("memory", memory())
        self.assertEqual(result["kv_gib_per_replica"], 4)
        self.assertEqual(result["estimated_gib_per_card_by_tp"]["2"], 12)
        self.assertEqual(result["minimum_tested_tp"], 2)
        self.assertIsNone(self.check("memory", memory(gpu_vram_gib=8))["minimum_tested_tp"])
        self.assertEqual(self.check("memory", memory(gpu_vram_gib=8,
            kv_sharding="ideal"))["minimum_tested_tp"], 4)

    def test_reject_conflicting_kv_layouts(self):
        result = run({"memory": memory(kv_bytes_per_token=512)})
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("not both", result.stderr)

    def test_service_rounds_replicas_and_reserves_capacity(self):
        result = self.check("service", dict(peak_requests_per_second=100,
            qualified_requests_per_second_per_replica=20, cards_per_replica=4, capacity_fraction=0.8))
        self.assertEqual(result, dict(replicas=7, total_cards=28))


if __name__ == "__main__":
    unittest.main()
