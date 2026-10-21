import tempfile
import multiprocessing as mp
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from nnunet_inference.parallel import (ParallelFoldMixin, assign_worker_cpus, average_fold_logits,
                                       limit_worker_memory)


class RecordingLock:
    def __init__(self):
        self.held, self.events = False, []

    def acquire(self):
        self.held = True
        self.events.append("acquire")

    def release(self):
        self.held = False
        self.events.append("release")


def small_resenc_unet():
    from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
    torch.manual_seed(0)
    return ResidualEncoderUNet(
        input_channels=1, n_stages=3, features_per_stage=(4, 8, 16), conv_op=torch.nn.Conv3d,
        kernel_sizes=3, strides=(1, 2, 2), n_blocks_per_stage=(1, 1, 1), num_classes=2,
        n_conv_per_stage_decoder=(1, 1), conv_bias=True, norm_op=torch.nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True}, nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={"inplace": True}, deep_supervision=False).eval()


class ParallelTests(unittest.TestCase):
    def test_cpus_are_disjoint_capped_and_handle_noncontiguous_affinity(self):
        available = list(range(2, 80, 2))
        cpus = assign_worker_cpus(5, available)
        self.assertEqual(cpus, [[2, 4, 6], [8, 10, 12], [14, 16, 18], [20, 22, 24], [26, 28, 30]])
        self.assertEqual(len(set(sum(cpus, []))), 15)
        with self.assertRaisesRegex(RuntimeError, "require 15"):
            assign_worker_cpus(5, list(range(14)))

    def test_half_precision_aggregation_matches_native_fold_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            rng = np.random.default_rng(32)
            paths, expected = [], None
            for fold in range(5):
                logits = torch.from_numpy(rng.normal(size=(2, 3, 4, 5)).astype(np.float16))
                path = Path(tmp) / f"fold_{fold}.npy"
                np.save(path, logits.numpy())
                paths.append(path)
                if expected is None:
                    expected = logits.clone()
                else:
                    expected += logits
            expected /= 5
            torch.testing.assert_close(average_fold_logits(paths), expected, rtol=0, atol=0)

    def test_bad_logits_fail_before_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.npy", Path(tmp) / "b.npy"
            np.save(a, np.zeros((2, 3), dtype=np.float16))
            np.save(b, np.zeros((2, 4), dtype=np.float16))
            with self.assertRaisesRegex(RuntimeError, "shapes/dtypes"):
                average_fold_logits([a, b])
            np.save(a, np.full((2, 3), np.nan, dtype=np.float16))
            with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
                average_fold_logits([a])

    def test_worker_keeps_one_weight_copy_and_locks_only_the_last_decoder_stage(self):
        network = small_resenc_unet()
        stored = {k: v.clone() for k, v in network.state_dict().items()}
        predictor = SimpleNamespace(network=network, list_of_parameters=[stored])
        x = torch.randn(1, 1, 8, 16, 16)
        with torch.inference_mode():
            expected = network(x)
        lock = RecordingLock()
        lock_wait = limit_worker_memory(predictor, lock)
        params = network.state_dict()
        self.assertTrue(all(t.data_ptr() == params[k].data_ptr()
                            for k, t in predictor.list_of_parameters[0].items()))
        held_at = {}

        def record(name):
            def hook(module, args, output):
                held_at.setdefault(name, lock.held)
            return hook
        for i, module in enumerate(network.decoder.transpconvs):
            module.register_forward_hook(record(f"transpconv {i}"))
        with torch.inference_mode():
            network.load_state_dict(predictor.list_of_parameters[0])
            output = network(x)
            network(x)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        self.assertEqual(held_at, {"transpconv 0": False, "transpconv 1": True})
        self.assertEqual(lock.events, ["acquire", "release"] * 2)
        self.assertGreaterEqual(lock_wait(), 0.0)

    def test_worker_failure_propagates_and_cleans_children(self):
        class Harness(ParallelFoldMixin):
            parallel_folds = (0,)
            parallel_checkpoint_name = "checkpoint_best.pth"
            use_mirroring = False
        with tempfile.TemporaryDirectory() as tmp:
            predictor = Harness()
            predictor.parallel_model_dir = tmp
            before = {p.pid for p in mp.active_children()}
            with self.assertRaisesRegex(RuntimeError, "fold_0 failed"):
                predictor.predict_logits_from_preprocessed_data(torch.zeros((1, 2, 2, 2)))
            self.assertEqual({p.pid for p in mp.active_children()}, before)
