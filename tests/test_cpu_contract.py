"""The shell scripts, Dockerfile and README repeat the CPU and memory numbers
of nnunet_inference/cpu.py, because they are read before Python starts."""
from pathlib import Path
import re
import unittest

from nnunet_inference import cpu

REPOSITORY = Path(__file__).resolve().parents[1]
INTEGRATION = REPOSITORY / "nnunet_inference" / "integration"


class CpuContractTests(unittest.TestCase):
    def test_manifest_numbers(self):
        self.assertEqual(cpu.contract(), dict(max_threads=16, threads_per_fold=3, memory_limit_gib=64))
        # Five folds must fit, with one thread to spare for the coordinator.
        self.assertGreater(cpu.MAX_THREADS, 5 * cpu.THREADS_PER_FOLD)

    def test_entrypoint_exports_the_thread_count(self):
        text = (INTEGRATION / "entrypoint.sh").read_text()
        exported = dict(re.findall(r"\b(\w+)=(\d+)\b", " ".join(
            line for line in text.splitlines() if line.startswith("export "))))
        self.assertEqual(exported, {name: str(cpu.THREADS_PER_FOLD)
                                    for name in (*cpu.THREAD_VARIABLES, "nnUNet_def_n_proc")})

    def test_dockerfile_labels(self):
        text = (INTEGRATION / ".ror/virt/Dockerfile").read_text()
        self.assertIn(f'com.mmiv.cpu.threads="{cpu.MAX_THREADS}"', text)
        self.assertIn(f'com.mmiv.memory.gib="{cpu.MEMORY_LIMIT_GIB}"', text)

    def test_smoke_test_and_readme_limits(self):
        limits = (f"--cpus {cpu.MAX_THREADS}", f"--memory {cpu.MEMORY_LIMIT_GIB}g",
                  f"--memory-swap {cpu.MEMORY_LIMIT_GIB}g")
        for name in ("smoke-test.sh", "README.md"):
            text = (INTEGRATION / name).read_text()
            for limit in limits:
                self.assertIn(limit, text, name)
        self.assertIn(f"[:{cpu.MAX_THREADS}]", (INTEGRATION / "smoke-test.sh").read_text())


if __name__ == "__main__":
    unittest.main()
