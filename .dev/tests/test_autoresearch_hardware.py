import unittest

from nanoprotein.check_environment import autoresearch_hardware


def devices(name, memory=48, capability=(8, 9)):
    return [
        dict(name=name, memory_bytes=memory * 1024**3, capability=capability) for _ in range(4)
    ]


class AutoResearchHardwareTests(unittest.TestCase):
    def test_profiles_choose_their_kernel_and_round_duration(self):
        l40s = autoresearch_hardware(devices("NVIDIA L40S"))
        self.assertEqual(l40s["attention_backend"], "flash")
        self.assertEqual(l40s["training_walltime_seconds"], 3600)
        self.assertEqual(l40s["autoresearch_profile"], "l40s-60m")
        h100 = autoresearch_hardware(devices("NVIDIA H100 80GB HBM3", 80, (9, 0)))
        self.assertEqual(h100["attention_backend"], "flash3")
        self.assertEqual(h100["peak_bf16_tflops_per_gpu"], 989.5)
        self.assertEqual(h100["training_walltime_seconds"], 1200)
        self.assertEqual(h100["autoresearch_profile"], "h100-20m")

    def test_rejects_mixed_small_or_unqualified_allocations(self):
        mixed = devices("NVIDIA L40S")
        mixed[-1]["name"] = "NVIDIA H100"
        for rows in (
            mixed,
            devices("NVIDIA L40S", 24),
            devices("NVIDIA L40S")[:3],
            devices("NVIDIA L40"),
            devices("NVIDIA H200", 140, (9, 0)),
            devices("unqualified GPU"),
        ):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                autoresearch_hardware(rows)
